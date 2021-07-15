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
from IPython.display import display as ipydisp

from notebooks.aduriseti_shared.utils import *
from models.utils import wavg, get_wavg_by, wstd

from models.taboola.common import *
from models.taboola.utils import *

start_date = TODAY - 90*DAY
eval_date = TODAY - 30*DAY
end_date = TODAY

split_cols = ["state", "device", "keyword"]
rps_df = agg_rps(start_date, end_date, None, traffic_source=TABOOLA,
                 agg_columns=tuple(["campaign_id", *split_cols, "utc_dt"]))
rps_df = translate_taboola_vals(rps_df)
rps_df = rps_df_postprocess(rps_df)
rps_df_bkp = rps_df.copy()
#%%
rps_df = rps_df_bkp.copy()
rps_df = rps_df.reset_index()
fitI = rps_df['utc_dt'].dt.date < eval_date
fitI.index = rps_df.index

import models.taboola.utils
importlib.reload(models.taboola.utils)
TaboolaRPSEst = models.taboola.utils.TaboolaRPSEst
clusterer = TaboolaRPSEst(clusts=32,enc_min_cnt=30).fit(
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
rps_df["rps_est"] = clusterer.predict(rps_df.set_index([*split_cols,"utc_dt"]))

rps_df_campaign = rps_df[rps_df["utc_dt"].dt.date > TODAY - 7*DAY] \
    .groupby(["campaign_id"])[["rps_est"]] \
    .agg(get_wavg_by(rps_df, "sessions"))
rps_df_publisher = rps_df \
        [rps_df["utc_dt"].dt.date > TODAY - 7*DAY] \
        .groupby(["campaign_id","keyword"])[["rps_est"]] \
        .agg(get_wavg_by(rps_df, "sessions")) \
        .unstack()
#%%
from pytaboola import TaboolaClient
from pytaboola.services import AccountService,CampaignService,CampaignSummaryReport
# d = CampaignSummaryReport(client, O65_ACCNT_ID).fetch(
#     dimension="campaign_day_breakdown",start_date=TODAY-7*DAY, end_date=TODAY)
# import jmespath
# jmespath.search("results[?cpc > `0`].{cpc: cpc,campaign_id: campaign, utc_dt: date}",d)

client = TaboolaClient(**TABOOLA_HC_CREDS)
acct_service = AccountService(client)
accnts = acct_service.list()["results"]
id2accnt = {a["account_id"]: a for a in accnts}

camps = []
for aid in [TEST_ACCNT_ID,O65_ACCNT_ID]:
    camp_service = CampaignService(client, aid)
    camps += camp_service.list()

campdf = pd.DataFrame([flatten_camp(camp) for camp in camps])
campdf = campdf.set_index(("attrs", "id"))
campdf.columns = pd.MultiIndex.from_tuples(campdf.columns)
print("|campdf|", campdf.shape)

print("campaign df sparsity:",((campdf == 0) | campdf.isna()).sum().sum() / np.prod(campdf.shape))
strC = campdf.dtypes[campdf.dtypes == object].index
floatC = campdf.dtypes[campdf.dtypes == np.float64].index

# campdf = campdf.reindex({*campdf.index,*active_camps})
active_camps = {*active_camps} & {*campdf.index}
active_camps
#%%
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
#%%
campdf.loc[active_camps,("updates","cpc")] = cpc_df_campaign_new.round(2)
campdf.loc[active_camps,("updates","publisher_bid_modifier")] = \
    bid_mod_df_new.round(2).apply(
        lambda r: {
                "values": [{'target': c, "bid_modification": v} for c,v in r[~r.isna()].items()]
            },
        axis=1)
campdf["updates"] = campdf["updates"].where(
    pd.notnull(campdf["updates"]), None)
updatedf = pd.concat((
    campdf.loc[active_camps,"attrs"]["advertiser_id"],
    campdf.loc[active_camps,"updates"].apply(dict,axis=1).apply(json.dumps),
),axis=1) \
    .reset_index()
updatedf.columns = ["campaign_id","account_id","update"]
updatedf["date"] = TODAY
updatedf["datetime"] = NOW

upload_taboola_updates_to_redshift(updatedf)
#%%
sql = f"""
    SELECT 
        *
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY account_id,campaign_id,schedule
                ORDER BY datetime DESC
            ) as rn
        FROM 
            {DS_SCHEMA}.{TABOOLA_CAMPAIGN_UPDATE_TABLE}
    )
    WHERE 
        rn = 1
    ;
"""
with HealthcareDW() as db:
    updatedf = db.to_df(sql)
updatedf['update'] = updatedf['update'].apply(json.loads)
#%%
for _,r in updatedf.iterrows():
    client = TaboolaClient(**TABOOLA_HC_CREDS)
    camp_service = CampaignService(client, r["account_id"])
    camp_service.update(r["campaign_id"],**r["update"])
#%%
campdf.loc[active_camps]
#%%
sql = f"""
select d.query, substring(d.filename,14,20), 
d.line_number as line, 
substring(d.value,1,16) as value,
substring(le.err_reason,1,48) as err_reason
from stl_loaderror_detail d, stl_load_errors le
where d.query = le.query
and d.query = pg_last_copy_id(); 
"""
sql = f"""
select *
from stl_load_errors
order by starttime desc
limit 100 
"""
with HealthcareDW() as db:
    df = db.to_df(sql)
df
#%%
df.iloc[0,-1]
#%%
df.iloc[0, -3]
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
