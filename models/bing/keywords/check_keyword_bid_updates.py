#%%
from models.bing.keywords.common import *
import glob
import datetime
import pandas as pd

TODAY = datetime.datetime.now().date()
# todays_output = glob.glob(f"{OUTPUT_DIR}/**/*{TODAY}.csv")
# df_out = pd.concat((pd.read_csv(fpth) for fpth in todays_output))
import boto3
ls_resp = boto3.client("s3").list_objects(Bucket=S3_OUTPUT_BUCKET,Prefix=S3_OUTPUT_PREFIX)
prev_output_keys = [o["Key"] for o in ls_resp["Contents"]]
todays_output_keys = [k for k in prev_output_keys if k.endswith(f"{TODAY}.csv")]

todays_output = [pd.read_csv(f"s3://{S3_OUTPUT_BUCKET}/{k}") for k in todays_output_keys]
df_out = pd.concat(todays_output)
old_len = df_out.__len__()
df_out = df_out.drop_duplicates() 
assert df_out.__len__() * 2 == old_len, """
We should have 2 records for each kw b/c we break bids out by account and write them,
but we also write the bids for all accounts.
"""
#%%
df_check = df_out
df_check["change"] = df_check["max_cpc_new"]/df_check["max_cpc_old"] - 1
df_check["cost_delta_est"] = df_check["cost_raw"] * df_check["change"]
# assume rpc will be unchanged - but volume will decrease proportional
#   to bid change
df_check["rev_delta_est"] = df_check["rev_raw"] * df_check["change"]
df_check["roas"] = df_check["rev_raw"] / df_check["cost_raw"]

total_roas = df_check["rev_raw"].sum() / df_check["cost_raw"].sum()
total_cost_delta_est = df_check["cost_delta_est"].sum()
total_rev_delta_est = df_check["rev_delta_est"].sum()

total_rel_delta = df_check[["rev_delta_est","cost_delta_est"]].sum() / \
    df_check[["rev_raw", "cost_raw"]].sum().values
roas_miss = total_roas - ROI_TARGET
roas_miss_delta = roas_miss - total_rel_delta["rev_delta_est"]
rel_roas_miss_delta = abs(roas_miss_delta) / min(abs(roas_miss),abs(total_rel_delta["rev_delta_est"]))

print("AGGREGATE:\n",df_check[["rev_raw", "cost_raw", "cost_delta_est", "rev_delta_est"]].sum())
print("RELATIVE:\n", total_rel_delta)
import pprint
pprint.pprint({
    "total_roas": total_roas,
    "total_cost_delta_est": total_cost_delta_est,
    "total_rev_delta_est": total_rev_delta_est,
    "":"",
    "ROI_TARGET": ROI_TARGET,
    "roas_miss": roas_miss,
    "roas_miss_delta": roas_miss_delta,
    "rel_roas_miss_delta": rel_roas_miss_delta,
})
#%%
"""
# TODO: make sure >= 51 state kws w/ in each geo gran gp - once we start using Dans catalog tables 

kw_idx_C = ["account", "geoI", "campaign_id", "adgroup_norm", "keyword_norm"]
bid_idx_C = ["account", "geoI", "campaign_id", "adgroup_id", "keyword_id"]
match_idx_C = ["account", "geoI", "campaign_id", "adgroup_id", "keyword_id", "match"]
df[match_idx_C].drop_duplicates().shape
#%%
df[kw_idx_C].drop_duplicates().shape
#%%
df["cnt"] = 1
geo_gran_cnts = df \
    [geoI] \
    .drop_duplicates(bid_idx_C) \
    .groupby(kw_idx_C) ["cnt"] .count() \
    .reset_index()
geo_gran_cnts
#%%
[*geo_gran_cnts["keyword_norm"]]
#%%
df[geoI]["adgroup_norm"].drop_duplicates()
"""
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
        - ROAS stuff split out by account
        - yesterday ROAS - 7 day ,30 day 
    - slack notif
        - gsheets
        - s3 link to unified dump
    - [x] take curtis off success notifications 
    - [x] take dan off error and success notifications
"""

import sys
def PASS():     pass
def WARN():     sys.exit(2)
def ERROR():    sys.exit(1)
if 0    <= rel_roas_miss_delta < 0.25:  PASS()
if 0.25 <= rel_roas_miss_delta < 1:     WARN()
if 1 <= rel_roas_miss_delta:            ERROR()

# if abs(total_cost_delta_est) < abs(total_rev_delta_est): WARN()

# if total_cost_delta_est < total_rev_delta_est: ERROR()
# %%
