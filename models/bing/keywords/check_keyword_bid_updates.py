#%%
from models.bing.keywords.common import *
import glob
import datetime
import pandas as pd
TODAY = datetime.datetime.now().date() 
todays_output = glob.glob(f"{OUTPUT_DIR}/**/*{TODAY}.csv")
df_out = pd.concat((pd.read_csv(fpth) for fpth in todays_output))
#%%
df_check = df_out
df_check["change"] = df_check["max_cpc_new"]/df_check["max_cpc_old"] - 1
df_check["cost_delta_est"] = df_check["cost_y"] * df_check["change"]
# assume rpc will be unchanged - but volume will decrease proportional
#   to bid change
df_check["rev_delta_est"] = df_check["rev_y"] * df_check["change"]
df_check["roas"] = df_check["rev_y"] / df_check["cost_y"]

total_roas = df_check["rev_y"].sum() / df_check["cost_y"].sum()
total_cost_delta_est = df_check["cost_delta_est"].sum()
total_rev_delta_est = df_check["rev_delta_est"].sum()

total_rel_delta = df_check[["rev_delta_est","cost_delta_est"]].sum() / \
    df_check[["rev_y", "cost_y"]].sum().values
roas_miss = total_roas - ROI_TARGET
roas_miss_delta = roas_miss - total_rel_delta["rev_delta_est"]
rel_roas_miss_delta = abs(roas_miss_delta) / min(abs(roas_miss),abs(total_rel_delta["rev_delta_est"]))

print("AGGREGATE:\n",df_check[["rev_y", "cost_y", "cost_delta_est", "rev_delta_est"]].sum())
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
