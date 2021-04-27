#%%
from pkg_resources import resource_filename as rscfn

import jmespath
import json
import datetime
import typing
import os
import tqdm
import itertools
import pandas as pd
import numpy as np

from ds_utils.db.connectors import HealthcareDW

NOW = datetime.datetime.now().date()
DAY = datetime.timedelta(days=1)

start_date = NOW - 180*DAY
end_date = NOW - 0*DAY
date_range = pd.date_range(start_date, end_date)

start_date_ymd = start_date.strftime("%Y%m%d")
end_date_ymd = end_date.strftime("%Y%m%d")
product = None
traffic_source = None
start_date_ymd, end_date_ymd

BING = "BING"
GOOG = "GOOGLE"
MA = "mediaalpha"
TABOOLA = "TABOOLA"
O65 = 'MEDICARE'

start_date = start_date_ymd
end_date = end_date_ymd
product = O65
traffic_source = TABOOLA
product_filter = "" if product is None else \
    f"AND UPPER(s.product) = UPPER('{product}')"
traffic_filter = "" if traffic_source is None else \
    f"AND UPPER(s.traffic_source) = UPPER('{traffic_source}')"
timeofday_query = f"""
    with
        rps as (
            SELECT
                session_id,
                sum(revenue) AS revenue
            FROM tron.session_revenue s
            WHERE session_creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
            {traffic_filter}
            GROUP BY 1
        ),
        rps_detail as (
            SELECT
                s.*,
                s.creation_date     AS utc_ts,
                r.revenue
            FROM 
                tracking.session_detail AS s
                JOIN rps as r
                    ON s.session_id = r.session_id
            WHERE s.creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
            {product_filter}
            {traffic_filter}
        )
    SELECT
        utc_ts::date                            as date,
        date_part(HOUR, utc_ts) +
        CASE 
            WHEN date_part(MINUTE, utc_ts)::INT BETWEEN 0 AND 14 THEN 0.0
            WHEN date_part(MINUTE, utc_ts)::INT BETWEEN 15 AND 29 THEN 0.25
            WHEN date_part(MINUTE, utc_ts)::INT BETWEEN 30 AND 44 THEN 0.5
            WHEN date_part(MINUTE, utc_ts)::INT BETWEEN 45 AND 59 THEN 0.75
        END                                     as hour,
        COUNT(*)                                as clicks_num,
        SUM((revenue>0)::int)                   as leads_num,
        SUM(revenue)                            as rev_sum,
        AVG(revenue)                            as rev_avg,
        STDDEV(revenue)                         as rev_std        
    FROM rps_detail r
    GROUP BY 
        date,hour
"""

bag_kpis_by_tod_sql = f"""
    with
        rps as (
            SELECT
                session_id,
                sum(revenue) AS revenue
            FROM tron.session_revenue s
            WHERE session_creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
            {traffic_filter}
            GROUP BY 1
        ),
        session_rps as (
            SELECT
                s.*,
                s.creation_date     AS creation_ts_utc,
                r.revenue
            FROM
                tracking.session_detail AS s
                JOIN rps as r
                    ON s.session_id = r.session_id
            WHERE s.creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
            {product_filter}
            {traffic_filter}
        ),
        session_cum_conv as (
            SELECT
                *,
                SUM((revenue>0)::INT) OVER (
                    ORDER BY creation_ts_utc
                    ROWS BETWEEN UNBOUNDED PRECEDING AND
                                    CURRENT ROW)    as conversions_cum
            FROM session_rps        
        ), 
        bag_rps as (
            SELECT
                *,
                conversions_cum                                                     as bag_id,
                COUNT(*) OVER (PARTITION BY conversions_cum)                        as bag_len,        
                SUM((revenue>0)::INT) OVER (PARTITION BY conversions_cum)           as bag_conv,
                AVG((revenue>0)::INT::float) OVER (PARTITION BY conversions_cum)    as bag_lpc,
                SUM(revenue) OVER (PARTITION BY conversions_cum)                    as bag_rpl,
                AVG(revenue) OVER (PARTITION BY conversions_cum)                    as bag_rpc
            FROM session_cum_conv
        )
    SELECT
        creation_ts_utc::date                   as date,
        date_part(HOUR, creation_ts_utc) +
        CASE 
            WHEN date_part(MINUTE, creation_ts_utc)::INT BETWEEN 0 AND 14 THEN 0.0
            WHEN date_part(MINUTE, creation_ts_utc)::INT BETWEEN 15 AND 29 THEN 0.25
            WHEN date_part(MINUTE, creation_ts_utc)::INT BETWEEN 30 AND 44 THEN 0.5
            WHEN date_part(MINUTE, creation_ts_utc)::INT BETWEEN 45 AND 59 THEN 0.75
        END                                     as hour,
        COUNT(*)                                as clicks_num,
        SUM(bag_lpc)                            as leads_num,
        SUM(bag_rpc)                            as rev_sum,
        AVG(bag_rpc)                            as rev_avg        
    FROM bag_rps
    GROUP BY 
        date,hour
"""
""" our version is PG 8 - range queries w/ definite bounds not supported 
    until PG 11
SUM((r.revenue>0)::INT) OVER (
            ORDER BY r.creation_ts_utc 
            RANGE BETWEEN '10 days'::INTERVAL PRECEDING AND
                            CURRENT ROW)    as conversions_past_day
"""

"""
ctr = clicks/impressions
lead/click = (rev > 0) / clicks
rev/lead = rev / (rev > 0)
rev/click = rev / clicks
"""

with HealthcareDW() as db:
    df = db.to_df(bag_kpis_by_tod_sql)

df["lpc"] = df["leads_num"] / df["clicks_num"]
df["rpl"] = df["rev_sum"] / df["leads_num"]
df["rpc"] = df["rev_avg"]

WEIGHT_COL = "clicks_num"
kpis = ["lpc", "rpl", "rpc"]
wkpis = [f"weighted_{kpi}" for kpi in kpis]
df[wkpis] = df[kpis] * df[WEIGHT_COL].values.reshape(-1, 1)

EPOCH = datetime.datetime(1970, 1, 1)
HOUR = datetime.timedelta(hours=1)
df['abs_hrs'] = (df["date"] - EPOCH.date()) / HOUR + df["hour"]
TEST_HR_INT = 2.25
N_GROUPS = 4
df['test_group'] = (df["abs_hrs"] // 2.5) % N_GROUPS
df = df.set_index("date")

df1 = df[df["test_group"] % 2 == 0].sum(level=["date"])
df2 = df[df["test_group"] % 2 == 1].sum(level=["date"])

def dfmult(df1,df2):
    intersection = {*df1.columns} & {*df2.index}
    return df1.loc[:, intersection].fillna(0) @ df2.loc[intersection, :].fillna(0)
def matnorm(X):
    return (X - X.mean())/X.std()
def gpcorr(df1,df2):
    X = df1[wkpis]
    Y = df2[wkpis]
    corrM = dfmult(matnorm(X).T, matnorm(Y)) / max(len(X), len(Y))
    return pd.Series(np.diag(corrM), index=wkpis)

gpcorr(df1,df2)
#%%
"""
GOAL:
- generate splits s.t. we can run A/B tests on kpis under some reasonable timeframe
    - generate timeframes for various effect sizes
- analysis
    1. run power test on traffic data - figure out experiment length for perfect split
        *NOTE*: define perfect split as having identical kpi behavior for A/A test 
    2. for range of testing window length and start/end dates
        - find split effect size in AA test 
        - assume worst case where split effect works against observed effect 
"""

from IPython.display import display as ipydisp

dfdtagg = df.sum(level="date")
mu = dfdtagg.mean()
std = dfdtagg.std()
lift_percentages = np.arange(10,100,10)
normalized_lifts = (lift_percentages / 100).reshape(-1,1) * (mu / std).values.reshape(1,-1)
normalized_lifts = pd.DataFrame(
    normalized_lifts,
    columns=dfdtagg.columns,
    index=lift_percentages
) 
from statsmodels.stats.power import TTestIndPower
exp_obs = normalized_lifts \
    .applymap(lambda lift: TTestIndPower().solve_power(effect_size=lift,nobs1=None,alpha=0.05,power=0.9,))
ipydisp(exp_obs[wkpis].astype(int))

dfgp1 = df[df["test_group"] % 2 == 0].sum(level=["date"])
dfgp2 = df[df["test_group"] % 2 == 1].sum(level=["date"])
AAeffects = (dfgp1.rolling(30).mean() - dfgp2.rolling(30).mean()).abs()
split_delta_mu = AAeffects.mean()
split_delta_sig = AAeffects.std()

for i in range(3):
    aa_normalized_effect = (split_delta_mu + split_delta_sig*i) / std
    normalized_lifts_worst_case = normalized_lifts - aa_normalized_effect
    exp_obs_worst_case = normalized_lifts_worst_case \
        .applymap(lambda lift: TTestIndPower().solve_power(effect_size=max(lift,1e-10), nobs1=None, alpha=0.05, power=0.9,))
    obs_added = (exp_obs_worst_case - exp_obs).astype(int)
    ipydisp(obs_added[wkpis])
#%%
def matnorm_fit_transform_half(X):
    # return X
    H,W = X.shape
    H = int(H/2)
    return ((X-X.iloc[:H].mean()) / X.iloc[:H].std()).iloc[H:]
import scipy.stats
from IPython.display import display as ipydisp

print("UNNORMALIZED SIGNIFICANCE TEST")
ind_test = scipy.stats.ttest_ind(df1, df2)
sigdf = pd.DataFrame(ind_test, index=["t", "p"], columns=df.columns)
ipydisp(sigdf[wkpis])

print("FIT-NORMALIZED SIGNIFICANCE TEST")
ind_test = scipy.stats.ttest_ind(matnorm_fit_transform_half(df1),matnorm_fit_transform_half(df2))
sigdf = pd.DataFrame(ind_test, index=["t", "p"], columns=df.columns)
ipydisp(sigdf[wkpis])

from matplotlib import pyplot as plt
def matnorm(X):
    return (X - X.mean())/X.std()
for c in wkpis:
    ax = plt.gca()
    for i, dfi in enumerate([df1,df2]):
        # gpsum(df, gp)[c].plot(figsize=(20, 10), label=f"gp {i}")
        # matnorm(dfi)[c].plot(figsize=(20, 10), label=f"gp {i}")
        dfi[c].plot(figsize=(20, 10), label=f"gp {i}")
    plt.title(f"Raw {c} by group")
    plt.legend()
    plt.show()

    ax = plt.gca()
    for i, dfi in enumerate([df1, df2]):
        # gpsum(df, gp)[c].plot(figsize=(20, 10), label=f"gp {i}")
        # matnorm(dfi)[c].plot(figsize=(20, 10), label=f"gp {i}")
        matnorm_fit_transform_half(dfi[[c]]).plot(figsize=(20, 10), label=f"gp {i}",ax=ax)
    plt.title(f"2nd half normalized by 1st half {c} by group")
    plt.legend()
    plt.show()
#%%
"""
AA test
- null hypo true
- p-val = prob(observe result >= observed result | no effect)
    => AA test - so want small observed result
    => want p-val to be large > 0.9
- power = prob(test rejects null hypo correctly | null hypothesis false)
- want negative power: prob(test rejects null)
    = prob(test configms null hypo | null hypothesis true)
"""
# %%
