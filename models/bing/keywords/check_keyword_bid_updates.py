#%%
from models.bing.keywords.common import *
import glob
import datetime
import pandas as pd
import numpy as np

import boto3
ls_resp = boto3.client("s3").list_objects(Bucket=S3_OUTPUT_BUCKET,Prefix=S3_OUTPUT_PREFIX)
prev_output_keys = [o["Key"] for o in ls_resp["Contents"]]
todays_output_keys = [k for k in prev_output_keys if k.endswith(f"{TODAY}.csv")]

todays_output = [pd.read_csv(f"s3://{S3_OUTPUT_BUCKET}/{k}") for k in todays_output_keys]
df_out = pd.concat(todays_output)
old_len = df_out.__len__()
df_out = df_out.drop_duplicates() 
assert df_out.__len__() * 2 == old_len, """
We should have 2 records for each kw b/c we break bids out by account_id and write them,
but we also write the bids for all accounts.
"""
#%%
from IPython.display import display as ipydisp

df_check = df_out

windows = [1,3,7,14,30]
clicksC = [f"clicks_sum_{n}day_raw" for n in windows]
revC = [f"rev_sum_{n}day_raw" for n in windows]
costC = [f"cost_sum_{n}day_raw" for n in windows]
roasC = [f"roas_{n}day" for n in windows]
df_check[roasC] = df_check[revC].abs() / (df_check[costC].abs().values + 1e-10)
for c,d in zip(roasC,costC):
    df_check.loc[df_check[d].abs() < 1e-2,c] = np.NaN
ipydisp(df_check[clicksC+revC+costC+roasC].agg(["sum","mean"]).round(2).T)

ipydisp(
df_check[[
        # 'account_id', 'account_num', 'match', 'campaign_id',
        # 'adgroup_id', 'keyword_id', 
        # 'campaign', 'adgroup', 'keyword',
        # 'adgroup_norm', 'keyword_norm', 'geoI', 
        # 'max_cpc',
        "clicks_sum_7day_raw", "rev_sum_7day_raw", "cost_sum_7day_raw","roas_7day",
        "rpc_est", "cpc_target","cpc_observed",
        "max_cpc_old", "max_cpc_new"]] \
    .rename(columns={
        "clicks_sum_7day_raw":  "clicks",
        "rev_sum_7day_raw":     "rev",
        "cost_sum_7day_raw":    "cost",
        "roas_7day":            "roas",
        "rpc_est":              "rpc",
        "cpc_target":           "cpc_goal",
        "cpc_observed":         "cpc_obs",
        "max_cpc_old":          "cpc_old",
        "max_cpc_new":          "cpc_new",
    }) \
    .sort_values(by="cost",ascending=False) \
    .round(2) \
    .iloc[:20])
#%%
df_check["max_cpc_old"] = np.maximum(df_check["max_cpc_old"],0.05)
df_check["change"] = df_check["max_cpc_new"]/df_check["max_cpc_old"] - 1
df_check["cost_t+1_est"] = (df_check["change"]+1) * df_check["cost_sum_7day_raw"]
df_check["rev_t+1_est"] = (df_check["change"]+1) * df_check["rev_sum_7day_raw"]
df_check["cost_delta_est"] = df_check["cost_sum_7day_raw"] * df_check["change"]
# assume rpc will be unchanged - but volume will increase/decrease proportional
#   to bid change
df_check["rev_delta_est"] = df_check["rev_sum_7day_raw"] * df_check["change"]

U = df_check[["cost_t+1_est","rev_t+1_est"]]
V = df_check[["cost_sum_7day","rev_sum_7day"]] + df_check[["cost_delta_est","rev_delta_est"]].values
assert all((U - V.values).abs() < 1e-10)

total_roas_windows = df_check[revC].sum() / df_check[costC].sum().values
total_roas_windows = pd.Series(data=total_roas_windows.values, index=roasC)
ipydisp(total_roas_windows)

total_roas = total_roas_windows["roas_7day"]
total_roas_delta = df_check["rev_t+1_est"].sum() / df_check["cost_t+1_est"].sum() - total_roas

roas_miss =  ROI_TARGET - total_roas
roas_miss_delta = roas_miss - total_roas_delta
rel_roas_miss_delta = abs(roas_miss_delta) / min(abs(roas_miss),abs(total_roas_delta))

ipydisp(pd.Series({
    "total_roas": total_roas,
    "total_cost_delta_est": df_check["cost_delta_est"].sum(),
    "total_rev_delta_est": df_check["rev_delta_est"].sum(),
    "total_roas_delta_est": total_roas_delta,
    "":"",
    "ROI_TARGET": ROI_TARGET,
    "roas_miss": roas_miss,
    "roas_miss_delta": roas_miss_delta,
    "rel_roas_miss_delta": rel_roas_miss_delta,
}))
#%%
"""
WTS
- under ROAS (ROAS < ROI_TARGET) => VOL & REV go down
- over ROAS (ROAS > ROI_TARGET) => VOL & REV go up
- want |ROAS miss| ~= |REV relative change|
- want ROAS_MISS == (ROAS - ROI_TARGET) ~= REV_delta_rel
- ROAS_MISS = ROAS - ROI_TARGET
- rel_delta_delta = |REV_delta_rel - ROAS_MISS| / min(|REV_delta_rel|,|ROAS_MISS|)

WED - using tues data
- ROAS 1.07 - TARGET 1.15
- project REV change was +4%, 0.04
- ROAS_MISS = 0.08
- 0.12
~3




CHECKS
- 
- rel_delta_delta 
    [0,0.25) => PASS
    [0.25,1) => WARN
    [1,inf)  => ERROR
- costs should change more then rev 
    - |COST_delta| > |REV_delta|    ELSE   => WARN
- costs always decrease more than rev
    - COST_delta < REV_delta        ELSE   => ERROR
- TODO
    - want COST_delta ~= REV_delta
    - want in gh action readout:
        - ROAS stuff split out by account_id
        - yesterday ROAS - 7 day ,30 day 
    - slack notif
        - gsheets
        - s3 link to unified dump
    - [x] take curtis off success notifications 
    - [x] take dan off error and success notifications
"""

WARN = False
ERROR = False

# definitely warn us if we are losing money 
if total_roas < 1: WARN |= True

if 0    <= rel_roas_miss_delta < 0.25:  pass
if 0.25 <= rel_roas_miss_delta:     WARN |= True
# if 0.25 <= rel_roas_miss_delta < 1:     WARN |= True
# if 1 <= rel_roas_miss_delta:            ERROR |= True

import sys
if ERROR:   sys.exit(1)
if WARN:    sys.exit(2)
# %%
