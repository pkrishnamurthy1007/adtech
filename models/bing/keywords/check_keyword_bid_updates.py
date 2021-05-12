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
# %%
total_roas = df_check["rev_y"].sum() / df_check["cost_y"].sum()
total_cost_delta_est = df_check["cost_delta_est"].sum()
total_rev_delta_est = df_check["rev_delta_est"].sum()
total_roas, total_cost_delta_est, total_rev_delta_est
# %%
df_check[["rev_y", "cost_y", "cost_delta_est", "rev_delta_est"]].sum()
# %%
df_check[["cost_delta_est", "rev_delta_est"]].sum() / \
    df_check[["rev_y", "cost_y"]].sum().values
# %%
total_roas
#%%
ROI_TARGET
#%%
"""
WTS
- under ROAS (ROAS < ROI_TARGET) => REV go down
- over ROAS (ROAS > ROI_TARGET) => REV go up
- want |ROAS miss| ~= |REV relative change|

CHECKS
- 
- IF REV_delta_rel \in (ROAS - ROI_TARGET) * [0.8,1.25] => OK
    ELIF REV_delta_rel \in (ROAS - ROI_TARGET) * [0.5,2] => WARN
    ELSE => ERROR
- |COST_delta| > |REV_delta| ELSE   => WARN
- |COST_delta| < |REV_delta| ELSE   => ERROR

"""
#%%
if total_roas > ROI_TARGET:
    pass
#%%
