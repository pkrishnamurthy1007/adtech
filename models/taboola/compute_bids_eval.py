#%%
from utils.env import load_env_from_aws
load_env_from_aws()

import importlib
import datetime
import itertools
import collections
import pprint
import sys
import re
import os

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from notebooks.aduriseti_shared.utils import *
from models.utils import wavg, get_wavg_by, wstd
from IPython.display import display as ipydisp
from models.utils.rpc_est import get_split_factor
import models.utils.rpc_est


NOW = datetime.datetime.now()
TODAY = NOW.date()
DAY = datetime.timedelta(days=1)

start_date = TODAY - 90*DAY
eval_date = TODAY - 30*DAY
end_date = TODAY

split_cols = ["state", "device", "keyword"]
rps_df = agg_rps(start_date, end_date, None, traffic_source=TABOOLA,
                 agg_columns=tuple(["campaign_id", *split_cols, "utc_dt"]))
rps_df = translate_taboola_vals(rps_df)
rps_df_bkp = rps_df.copy()
#%%
rps_df = rps_df_bkp.copy()
importlib.reload(models.utils.rpc_est)
AggRPSClust = models.utils.rpc_est.AggRPSClust
TreeRPSClust = models.utils.rpc_est.TreeRPSClust
KpiSimClust = models.utils.rpc_est.KpiSimClust
HybridCorrTreeClust = models.utils.rpc_est.HybridCorrTreeClust

rps_df = rps_df.reset_index()
rps_df["leads"] = rps_df["num_leads"].fillna(0)
rps_df["lps"] = rps_df["leads"] / rps_df["sessions"]
rps_df["rpl"] = rps_df["revenue"] / rps_df["leads"]
rps_df["score"] = rps_df[["score_null_avg",
                          "score_adv_avg", "score_supp_avg"]].sum(axis=1)
rps_df["rps"] = rps_df["rps_avg"]
fitI = rps_df['utc_dt'].dt.date < eval_date
fitI.index = rps_df.index
rps_df["rps_"] = rps_df["revenue"] / rps_df["sessions"]
delta = rps_df["rps"] - rps_df["rps_"]
assert delta.abs().max() < 1e-10
assert abs(rps_df["revenue"].sum() / rps_df["sessions"].sum() -
           wavg(rps_df["rps"], rps_df["sessions"])) < 1e-10

clusterer = TreeRPSClust(clusts=32,enc_min_cnt=30).fit(
    rps_df[fitI].set_index([*split_cols, "utc_dt"]), None)
rps_df.loc[fitI, "clust"] = clusterer.transform(
    rps_df[fitI].set_index([*split_cols, "utc_dt"]))
rps_df.loc[~fitI, "clust"] = clusterer.transform(
    rps_df[~fitI].set_index([*split_cols, "utc_dt"]))
rps_df["clust"] = rps_df["clust"].fillna(-1)
rps_df["rps_clust"] = rps_df \
    .groupby(["clust", "utc_dt"])["rps"].transform(get_wavg_by(rps_df, "sessions"))
daily_rps_mae = (rps_df["rps"] - rps_df["rps_clust"]).abs()
assert abs(
    wavg(rps_df["rps_clust"], rps_df["sessions"]) -
    wavg(rps_df["rps"], rps_df["sessions"])) < 1e-10

kpis_agg = ["revenue", "sessions", "leads"]
kpis_session = ["rps", "lps"]
kpis_lead = ["rpl"]
clust_rps_df = rps_df[~fitI].groupby("clust")[kpis_agg].sum()
clust_rps_df[kpis_session] = rps_df[~fitI].groupby("clust") \
    .apply(lambda df: wavg(df[kpis_session], df['sessions']))
clust_rps_df[kpis_lead] = rps_df[~fitI].groupby("clust") \
    .apply(lambda df: wavg(df[kpis_lead].fillna(0).values, df['leads']))
# clust_rps_df[split_cols]
clust_rps_df["rps_"] = clust_rps_df["revenue"] / clust_rps_df["sessions"]
clust_rps_df["rpl_"] = clust_rps_df["revenue"] / clust_rps_df['leads']
# agg_rps_df = rps_df[~fitI].groupby(rps_df.index.names[:-1]).agg({
#         "sessions": sum,
#         "rps": get_wavg_by(rps_df[~fitI],"sessions")
#     })
ipydisp(clust_rps_df)
# assert clust_rps_df["rps"].max() <= agg_rps_df["rps"].max()
# rps_wavg = wavg(agg_rps_df[["rps"]], agg_rps_df["sessions"])
rps_wavg = wavg(rps_df[~fitI]["rps"], rps_df[~fitI]["sessions"])
rps_clust_wavg = wavg(clust_rps_df[["rps"]], clust_rps_df["sessions"])
assert all((rps_wavg - rps_clust_wavg).abs()
           < 1e-3), (rps_wavg, rps_clust_wavg)
rps_wavg, rps_clust_wavg

perfd = {
    "clusterer": clusterer,
    # "fit_shape": agg_rps_df.shape,
    "clust_shape": clust_rps_df.shape,
    # "split_variance": wstd(agg_rps_df["rps"], agg_rps_df["sessions"]),
    "cluster_variance": wstd(clust_rps_df["rps"], clust_rps_df["sessions"]),
    # wstd(rps_df["rps_avg"],rps_df["sessions"])
    # "clustered_split_factor": get_split_factor(rps_df),
    "rps_mae": wavg(daily_rps_mae, rps_df["sessions"]),
}
pprint.pprint(perfd)
#%%
clust_dt_rps_df = rps_df.groupby(["clust", "utc_dt"])[kpis_agg].sum()
clust_dt_rps_df[kpis_session] = rps_df.groupby(["clust", "utc_dt"]) \
    .apply(lambda df: wavg(df[kpis_session], df['sessions']))
clust_dt_rps_df[kpis_lead] = rps_df[~fitI].groupby(["clust", "utc_dt"]) \
    .apply(lambda df: wavg(df[kpis_lead].fillna(0).values, df['leads']))

# 30 is a good breakpt for using bag mtd
clust_dt_rps_df = clust_dt_rps_df.groupby("clust") \
    .apply(lambda df:
           df
           .reset_index("clust", drop=True)
           .reindex(pd.date_range(start_date, end_date)).fillna(0))
clust_dt_rps_df.index.names = ["clust", "utc_dt"]

def get_nday_sum(c,n):
    def f(df):
        return df.groupby("clust") \
            .apply(lambda df:
                df
                .reset_index("clust", drop=True)
                [[c]].rolling(n).sum()) \
            [c]
    return f

rpl = get_nday_sum("revenue", 7)(clust_dt_rps_df).groupby("utc_dt").transform(sum) / \
    get_nday_sum("leads", 7)(clust_dt_rps_df).groupby("utc_dt").transform(sum)
lps = get_nday_sum("leads", 60)(clust_dt_rps_df) / get_nday_sum("sessions", 60)(clust_dt_rps_df)
clust_dt_rps_df["rps_est"] = rpl * lps

for ci in clust_dt_rps_df.index.unique("clust"):
    clust_dt_rps_df.loc[ci, "rps_est"].plot(label=ci)
plt.legend()
plt.show()

rps_df = rps_df.sort_values(["clust","utc_dt"])
rps_df["rps_est"] = rps_df.set_index(["clust","utc_dt"])[[]].join(clust_dt_rps_df["rps_est"]).values
# rps_df.groupby(["utc_dt","campaign_id"])[["rps_est"]] \
#     .agg(get_wavg_by(rps_df,"sessions"))
rps_df_campaign = rps_df[rps_df["utc_dt"].dt.date > TODAY - 7*DAY] \
    .groupby(["campaign_id"])[["rps_est"]] \
    .agg(get_wavg_by(rps_df, "sessions"))
rps_df_publisher = rps_df \
        [rps_df["utc_dt"].dt.date > TODAY - 7*DAY] \
        .groupby(["campaign_id","keyword"])[["rps_est"]] \
        .agg(get_wavg_by(rps_df, "sessions")) \
        .unstack()
#%%
for ci in lps.index.unique("clust"):
    lps.loc[ci].plot()
#%%
rpl[ci].plot()
#%%
import json
TABOOLA_HC_CREDS = json.loads(os.getenv("TABOOLA_HC_CREDS"))
TABOOLA_PIVOT_CREDS = json.loads(os.getenv("TABOOLA_PIVOT_CREDS"))

from pytaboola import TaboolaClient
from pytaboola.services import AccountService,CampaignService,CampaignSummaryReport
# d = CampaignSummaryReport(client, O65_ACCNT_ID).fetch(
#     dimension="campaign_day_breakdown",start_date=TODAY-7*DAY, end_date=TODAY)
# import jmespath
# jmespath.search("results[?cpc > `0`].{cpc: cpc,campaign_id: campaign, utc_dt: date}",d)

client = TaboolaClient(**TABOOLA_HC_CREDS)
acct_service = AccountService(client)
accnts = acct_service.list()["results"]
NETWORK_ACCNT_ID = "healthcareinc-network"
TEST_ACCNT_ID = "healthcareinc-sc2"
O65_ACCNT_ID = "taboolaaccount-rangaritahealthcarecom"
id2accnt = {a["account_id"]: a for a in accnts}

camps = []
for aid in [TEST_ACCNT_ID,O65_ACCNT_ID]:
    camp_service = CampaignService(client, aid)
    camps += camp_service.list()

import itertools
cross = itertools.product
import jmespath
get = jmespath.search

def yield_from_type_pivot(camp,k):
    for t,v in cross(
            get(f"[{k}.type]", camp),
            get(f"{k}.value", camp) or [""]):
        yield [(k,(t,v)),1]

def yield_time_rules(camp):
    DAYS = [
        'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY',
        'FRIDAY', 'SATURDAY', 'SUNDAY', ]
    HRS = range(24)
    k = "activity_schedule"
    time_mode = get(f"{k}.mode", camp)
    if time_mode == "ALWAYS":
        included_hrs = {
            day: {hr: 1 for hr in HRS}
            for day in DAYS
        }
    else:
        included_hrs = {
            day: {hr: 0 for hr in HRS}
            for day in DAYS
        }
        for rule, day, st, end in \
                get(f"{k}.rules[*].[type,day,from_hour,until_hour]", camp):
            for hr in range(st, end):
                included_hrs[day.upper()][hr] = int(rule == "INCLUDE")
    for day, hr2v in included_hrs.items():
        for hr, v in hr2v.items():
            yield [(k, (day, hr)), v]

def yield_bid_strategy_rules(camp):
    k = "publisher_bid_strategy_modifiers"
    for r in get(f"{k}.values[*].*",camp):
        yield [(k,tuple(r)),1]

def yield_bid_modifiers(camp):
    k = "publisher_bid_modifier"
    for site,mod in get(f"{k}.values[*].*", camp):
        yield [(k, site), mod]

flat_K = [
    "advertiser_id",
    "id",
    "cpc",
    "safety_rating",
    "daily_cap",
    "daily_ad_delivery_model",
    "bid_type",
    "bid_strategy",
    "traffic_allocation_mode",
    "marketing_objective",
    "is_active",
]
type_pivoted_K = [
    "country_targeting",
    "sub_country_targeting",
    "dma_country_targeting",
    "region_country_targeting",
    "city_targeting",
    "postal_code_targeting",
    "contextual_targeting",
    "platform_targeting",
    "publisher_targeting",
    "auto_publisher_targeting",
    # "os_targeting",
    "connection_type_targeting",
    "browser_targeting",
]

def flatten_camp(camp):
    campd = {("attrs",k): camp[k] for k in flat_K}
    for k in type_pivoted_K:
        campd.update(dict(yield_from_type_pivot(camp,k)))
    campd.update(dict(yield_time_rules(camp)))
    campd.update(dict(yield_bid_strategy_rules(camp)))
    campd.update(dict(yield_bid_modifiers(camp)))
    return campd

import pandas as pd
campdf = pd.DataFrame([flatten_camp(camp) for camp in camps])
campdf = campdf.set_index(("attrs", "id"))
campdf.columns = pd.MultiIndex.from_tuples(campdf.columns)
# bid_mod_C = [c for c in campdf.columns if "publisher_bid_modifier" == c[0]]
# campdf[bid_mod_C].fillna(1)
# campdf = campdf.fillna(0)
print("|campdf|", campdf.shape)

import numpy as np
print("campaign df sparsity:",((campdf == 0) | campdf.isna()).sum().sum() / np.prod(campdf.shape))
strC = campdf.dtypes[campdf.dtypes == object].index
floatC = campdf.dtypes[campdf.dtypes == np.float64].index
#%%
# %%
# active_camp_df = pd.read_csv(rscfn(__name__,"active_campaigns.csv"))
# active_camps = active_camp_df["id"]

activeI = campdf["attrs"]["is_active"]
active_camps = campdf.loc[activeI].index
cpc_df_campaign_new = np.clip(
    rps_df_campaign["rps_est"].reindex(active_camps) / ROI_TARGET,
    (1-MAX_CUT)*campdf["attrs"].loc[active_camps,"cpc"],
    (1+MAX_PUSH)*campdf["attrs"].loc[active_camps,"cpc"])
cpc_df_campaign_new = cpc_df_campaign_new \
                        .combine_first(campdf["attrs"].loc[active_camps,"cpc"])

import requests
resp = requests.get(
    f"{TABOOLA_BASE}/{O65_ACCNT_ID}/allowed-publishers/",
    headers=client.authorization_header)
taboola_publishers = jmespath.search('results[].account_id', resp.json())

bid_mod_df = campdf["publisher_bid_modifier"] \
    .reindex(active_camps) \
    .T.reindex(taboola_publishers).T
cpc_df_publisher = bid_mod_df.fillna(1) * \
    campdf["attrs"].loc[active_camps, ["cpc"]].values
cpc_df_publisher_new = np.clip(
    rps_df_publisher["rps_est"] \
        .reindex(active_camps) \
        .T.reindex(taboola_publishers).T / ROI_TARGET,
    (1-MAX_CUT)*cpc_df_publisher,
    (1+MAX_PUSH)*cpc_df_publisher,)
bid_mod_df_new = cpc_df_publisher_new / cpc_df_campaign_new.values.reshape(-1,1)
approx1 = (bid_mod_df_new - 1).abs() < 1e-2
bid_mod_df_new = bid_mod_df_new.loc[:,~(bid_mod_df_new.isna() | approx1).all(axis=0)]
bid_mod_df_new = bid_mod_df_new \
    .combine_first(bid_mod_df.loc[:,~bid_mod_df.isna().any()])

campdf.loc[active_camps,("updates","cpc")] = cpc_df_campaign_new
campdf.loc[active_camps,("updates","publisher_bid_modifier")] = \
    bid_mod_df_new.apply(
        lambda r: {
                "values": [{'target': c, "bid_modification": v} for c,v in r.items()]
            },
        axis=1)
campdf.loc[active_camps,"updates"].apply(dict,axis=1)
#%%

rps_df = rps_df.join(campdf["attrs"][["cpc","is_active"]],on="campaign_id",rsuffix="_")
rps_df = rps_df.loc[:,~rps_df.columns.duplicated()]
rps_df["cost"] = rps_df["cpc"] * rps_df["sessions"]
df = rps_df \
    [rps_df["is_active"].fillna(False)] \
    .groupby(["utc_dt","campaign_id"])\
    [["revenue","cost"]].sum().unstack()
(df["revenue"]/df["cost"]).plot()
#%%
#%%
campdf["publisher_bid_modifier"]
s1 = df["rps_est"].unstack().columns
s2 = campdf["publisher_bid_modifier"].columns
s1 = {*s1}; s2 = {*s2}
len(s1-s2),len(s2-s1),len(s1&s2)
#%%
{f[3:] for f in clusterer.enc_features} - s2
#%%
import requests
TABOOLA_BASE = "https://backstage.taboola.com/backstage/api/1.0"
resp = requests.get(
    f"{TABOOLA_BASE}/resources/campaigns_properties/operating_systems",
    headers=client.authorization_header)
taboola_os = jmespath.search('results[].name', resp.json(),)

resp = requests.get(
    f"{TABOOLA_BASE}/resources/campaigns_properties/platforms",
    headers=client.authorization_header)
taboola_platforms = jmespath.search('results[].name', resp.json(),)

resp = requests.get(
    f"{TABOOLA_BASE}/resources/countries/us/dma",
    headers=client.authorization_header)
taboola_dmas = jmespath.search('results[].name', resp.json(),)

resp = requests.get(
    f"{TABOOLA_BASE}/resources/countries/us/regions",
    headers=client.authorization_header)
taboola_states = jmespath.search('results[].name', resp.json(),)

resp = requests.get(
    f"{TABOOLA_BASE}/{O65_ACCNT_ID}/allowed-publishers/",
    headers=client.authorization_header)
taboola_publishers = jmespath.search('results[].account_id',resp.json())
#%%
resp = requests.get(
    f"{TABOOLA_BASE}/{O65_ACCNT_ID}/dictionary/audience_segments/",
    headers=client.authorization_header)
taboola_audiences = resp.json()["results"]

resp = requests.get(
    f"{TABOOLA_BASE}/{O65_ACCNT_ID}/dictionary/lookalike_audiences/",
    headers=client.authorization_header)
resp.json()

resp = requests.get(
    f"{TABOOLA_BASE}/{O65_ACCNT_ID}/dictionary/contextual_segments/",
    headers=client.authorization_header)
len(resp.json()["results"])
# %%
s3 = {*taboola_publishers}
len(s1-s3),len(s2-s3)
# %%
{f[3:] for f in clusterer.enc_features} - s3
# %%
