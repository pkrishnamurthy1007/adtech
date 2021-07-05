#%%
#### LOAD COMMON ####
from api.bingads.bingapi.client import *
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
#%%
# # TODO: address roughly 3k in "un-accounted" revenue
# reporting_df[reporting_df["account_id"].isna()]["rev"].sum()
# reporting_df[reporting_df["keyword_id"].isna()]["rev"].sum()
reporting_df = pd.merge(
    reporting_df,
    accnt_df[["account_id","account_name"]],
    on="account_id")

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
assert reporting_df["costI"].sum() + reporting_df["revI"].sum() == len(reporting_df)

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
    .groupby(["account_name","date"]) .agg(kpi_agg_d) \
    .unstack(level=0)
accnts = reporting_df["account_name"].dropna().unique()
date_kpi_accnt_df[[("ROAS",accnt) for accnt in accnts]] = \
    (date_kpi_accnt_df["rev"] / date_kpi_accnt_df["cost"])[accnts]
#%%
for kpi in [*kpiC,"ROAS"]:
    date_kpi_accnt_df[kpi][accnts] \
        .rolling(7).mean().plot()
    plt.title(kpi)
    plt.show()
#%%
# bid_hist_sql = f"""
# select account_id, id keyword_id, parent_id adgroup_id, modified_time, keyword, match_type, bid
# from dl_landing.external_bingads_entity_detail
# where type='Keyword'
# and process_date='2021-07-02'
# """
# with HealthcareDW(database="adtech") as db:
#     bid_hist_df = db.to_df(bid_hist_sql)
# bid_hist_df
#%%
import json,logging
bing_creds = json.loads(os.getenv("BING_CREDS"))
LOGLEVEL = logging.WARN
from api.bingads.bingapi.client import BingClient
#%%
1/0
DAY = datetime.timedelta(days=1)
accnt2reset_date = {
    # 'HealthCare.com O65': 1/0,
    'HealthCare.com U65':   datetime.date(2021,3,2),
    'HealthCare.org U65':   datetime.date(2021,5,4),
    # 'MedicareGuide.com':    datetime.date(2021,6,29),
}
accnt_nm,dt = [*accnt2reset_date.items()][1]
# accnt_nm, dt = accnt2reset_date.pop("HealthCare.org U65")
for accnt_nm,dt in accnt2reset_date.items():
    print(accnt_nm,dt)

    accntI = reporting_df["account_name"] == accnt_nm
    date = reporting_df["date"].dt.date
    dtI = (dt-7*DAY < date) & (date <= dt)
    # kws = reporting_df.loc[accntI,"keyword_id"].unique()

    kw_attrs = reporting_df \
        [accntI] \
        [["account_id", "campaign_id","adgroup_id", "keyword_id"]] \
        .dropna().groupby("keyword_id").last()

    from models.utils import get_wavg_by,wavg
    kw_cpcs = reporting_df \
        [accntI & dtI] \
        .groupby("keyword_id") \
        .agg({
            "max_cpc": "mean", 
            "clicks": sum,
        }) \
        .reindex(kw_attrs.index)
    kw_attrs = pd.concat((kw_attrs,kw_cpcs),axis=1)

    adgp_cpc = kw_attrs.groupby("adgroup_id")["max_cpc"].transform(get_wavg_by(kw_attrs,"clicks"))
    camp_cpc = kw_attrs.groupby("campaign_id")["max_cpc"].transform(get_wavg_by(kw_attrs,"clicks"))
    accnt_cpc = kw_attrs.groupby("account_id")["max_cpc"].transform(get_wavg_by(kw_attrs,"clicks"))
    kw_attrs["max_cpc"] = kw_attrs["max_cpc"] \
        .combine_first(adgp_cpc) \
        .combine_first(camp_cpc) \
        .combine_first(accnt_cpc)

    accnt_id = kw_attrs["account_id"].unique().astype(int)[0]
    campaign_ids,adgroup_ids,keyword_ids = kw_attrs.reset_index() \
        [["campaign_id","adgroup_id","keyword_id"]].astype(int).values.T
    keyword_bids = kw_attrs["max_cpc"].astype(float).round(2).values
    accnt_client = BingClient(
        account_id=accnt_id,
        customer_id=bing_creds['BING_CUSTOMER_ID'],
        dev_token=bing_creds['BING_DEVELOPER_TOKEN'],
        client_id=bing_creds['BING_CLIENT_ID'],
        refresh_token=bing_creds['BING_REFRESH_TOKEN'],
        loglevel=LOGLEVEL,
    )
    keyword_updates = accnt_client.bulk_update_keyword_bids(
        adgroup_ids, keyword_ids, keyword_bids)
    keyword_bid_updates = [kwu.keyword.Bid.Amount for kwu in keyword_updates]
    assert all(np.array(sorted(keyword_bid_updates))
                == np.array(sorted(keyword_bids)))
# %%
