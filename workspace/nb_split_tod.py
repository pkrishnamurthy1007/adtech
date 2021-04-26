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

start_date = NOW - 60*DAY
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
traffic_source = BING
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
        rps_tz_adj as (
            SELECT
                s.*,
                s.creation_date                                         AS utc_ts,
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
    FROM rps_tz_adj
    GROUP BY 
        date,hour
"""
"""
ctr = clicks/impressions
lead/click = (rev > 0) / clicks
rev/lead = rev / (rev > 0)
rev/click = rev / clicks
"""

with HealthcareDW() as db:
    df = db.to_df(timeofday_query)

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
#%%
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
