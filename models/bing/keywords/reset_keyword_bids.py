#%%
#### LOAD COMMON ####
import uniplot
from models.bing.keywords.common import *

#### LOAD PACKAGES ####
import sys
import numpy as np
import pandas as pd
import pytz
import datetime
import boto3

from IPython.display import display as ipydisp
from ds_utils.db.connectors import HealthcareDW

#### DEFINE GLOBAL VARIABLES ####

CLICKS = 120  # click threshold. level at which kw uses all of its own data
MAX_PUSH = 0.2
MAX_CUT = -0.3
CPC_MIN = 0.05

perf_reporting_sql = """
SELECT
    transaction_date        AS date,
    transaction_hour,
    transaction_date_time,
    data_source_type,
    account_id              AS account_num,
    campaign_id,
    ad_group_id             AS adgroup_id,
    keyword_id,
    campaign_name           AS campaign,
    ad_group                AS adgroup,
    keyword,
    paid_clicks             AS clicks,
    cost,
    revenue                 AS rev,
    match_type              AS match,
    max_cpc                 AS max_cpc
FROM hc.tron.intraday_profitability
WHERE
    date >= current_date - 180 AND
    channel = 'SEM' AND
    traffic_source = 'BING'
"""
kw_sql = f"""
WITH
    perf_reporting as ({perf_reporting_sql})
SELECT 
    perf_reporting.*,
    kw.account_id::INT      AS account_id,
    kw.bid                  AS latest_max_cpc
FROM 
    perf_reporting
LEFT JOIN dl_gold.adtech_bingads_keyword AS kw
ON perf_reporting.keyword_id = kw.keyword_id
"""
with HealthcareDW(database="adtech") as db:
    reporting_df = db.to_df(kw_sql)
reporting_df_bkp = reporting_df

# TODO: pull down keywords for all active campaigns in data window as kw df
with HealthcareDW(database="adtech") as db:
    accnt_df = db.to_df("select * from dl_gold.adtech_bingads_account")
# with HealthcareDW(database="adtech") as db:
#     kw_df = db.to_df("select * from dl_gold.adtech_bingads_keyword")

# # TODO: address roughly 3k in "un-accounted" revenue
# reporting_df[reporting_df["account_id"].isna()]["rev"].sum()
# reporting_df[reporting_df["keyword_id"].isna()]["rev"].sum()
#%%
reporting_df = reporting_df_bkp

#### DATA MUNGING AND CLEANING ####
"""
CHANGED:
- used to run script for each account_id - now computes updates for accounts in batch
"""

# set clicks to numeric
reporting_df['clicks'] = reporting_df['clicks'].astype(float)

# set nans for cost in rev reporting rows - and nans for rev in cost rerpotign rows
costI = reporting_df["data_source_type"] == "COST"
revI = reporting_df["data_source_type"] == "REVENUE"
reporting_df.loc[costI, "rev"] = np.NaN
reporting_df.loc[revI, "max_cpc"] = np.NaN

reporting_df["costI"] = costI
reporting_df["revI"] = revI
assert reporting_df["costI"].sum(
) + reporting_df["revI"].sum() == len(reporting_df)

#### PROCESS DATE ####
reporting_df['date'] = pd.to_datetime(reporting_df['date']) \
    .dt.tz_localize("UTC") \
    .dt.tz_convert("EST") \
    .dt.date
reporting_df["date"] = pd.to_datetime(reporting_df["date"])
reporting_df['today'] = TODAY
#%%
from matplotlib import pyplot as plt
### PLOT ROLLING KPIS ###
kpiC = ["rev", "cost", "clicks", "max_cpc"]  
kpi_agg_d = {
    "rev": sum,
    "cost": sum,
    "clicks": sum,
    "max_cpc": "mean",
}
def plot_performance(reporting_df):
    date_kpi_df = reporting_df .groupby("date").agg(kpi_agg_d)
    date_kpi_df["ROAS"] = date_kpi_df["rev"] / date_kpi_df["cost"]
    kpiC += ["ROAS"]
    for n in [3, 7, 14, 30]:
        rolling_kpi_C = [f"{c}_{n}d_avg" for c in kpiC]
        date_kpi_df[rolling_kpi_C] = date_kpi_df[kpiC].rolling(n).mean()
        (date_kpi_df[rolling_kpi_C] /
        date_kpi_df[rolling_kpi_C].mean()).iloc[:, :-1].plot()

    roasC = [c for c in date_kpi_df.columns if c.startswith("ROAS")]
    date_kpi_df[roasC[1:]].plot()

    # # requires `gnuplot` which may not be available on gh action server
    # import termplotlib
    # termplotlib.plot.plot(bucket_rps_df["rps_avg_true"].fillna(0),bucket_rps_df["rps_avg_true"].index)

    # # doesnt work in vscode
    # import plotext
    # plotext.plot([dfagg_norm["max_cpc"], dfagg_norm["rev_7day"]], dfagg_norm.index)
    # plotext.title(f"mean normalized 7-day rolling rev and reported max_cpc")
    # plotext.plotsize(100,30)
    # plotext.show()

    uniplot.plot([*date_kpi_df[roasC[1:]].fillna(1).values.T],
                legend_labels=roasC[1:],
                title=f"rolling ROAS",
                width=90, height=15)

date_kpi_accnt_df = reporting_df \
    .groupby(["account_num","date"]) .agg(kpi_agg_d) \
    .unstack(level=0)
accnts = reporting_df["account_num"].dropna().unique()
date_kpi_accnt_df[[("ROAS",accnt) for accnt in accnts]] = \
    (date_kpi_accnt_df["rev"] / date_kpi_accnt_df["cost"])[accnts]
#%%
for kpi in [*kpiC,"ROAS"]:
    date_kpi_accnt_df[kpi][['X000EWRC', 'B013D68T', 'B013P57C']] \
        .rolling(7).mean().plot()
    plt.title(kpi)
    plt.show()
# %%
