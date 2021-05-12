#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAILY KEYWORD BIDDING ALGORITHM BING
"""

#### LOAD PACKAGES ####
import os
import numpy as np
import pandas as pd
import pytz
import datetime
import boto3
#import bingads

#### LOAD COMMON ####
from models.bing.keywords.common import * 

import sys; sys.exit(-2)

#### DEFINE GLOBAL VARIABLES ####

CLICKS = 120 #click threshold. level at which kw uses all of its own data
ROI_TARGET = 1.15 # target we are aiming
MAX_PUSH = 0.2
MAX_CUT = -0.3
CPC_MIN = 0.05
# data is reported in EST 
NOW = datetime.datetime.now(pytz.timezone('EST'))
TODAY = NOW.date()
WEEKDAY = TODAY.weekday()
DAYS_BACK = 1

query = f"""
SELECT 
    transaction_date as date,
    transaction_hour,
    transaction_date_time,
    data_source_type,
    account_id as account,
    campaign_id,
    ad_group_id as adgroup_id, 
    keyword_id,
    campaign_name as campaign,
    ad_group as adgroup,
    keyword, 
    paid_clicks as clicks,
    cost,
    revenue as rev,
    match_type as match,
    max_cpc
FROM tron.intraday_profitability
WHERE 
    date >= current_date - 84 AND 
    channel = 'SEM' AND
    traffic_source = 'BING'
"""
from ds_utils.db.connectors import HealthcareDW
# TODO: submit a PR where env vars are not accessed on connector class def
# TODO: https://healthcareinc.slack.com/archives/C01VBRTJ4R5/p1619728061020600
with HealthcareDW() as db:
    df = db.to_df(query)
df_bkp = df

# TODO: address roughly 3k in "un-accounted" revenue
# df = df_bkp
# df[df["account"].isna()]["rev"].sum()
#%%
df = df_bkp

#### DATA MUNGING AND CLEANING ####
"""
CHANGED:
- used to run script for each account - now computes updates for accounts in batch
"""

# set clicks to numeric
df['clicks'] = df['clicks'].astype(float)

# set nans for cost in rev reporting rows - and nans for rev in cost rerpotign rows
costI = df["data_source_type"] == "COST"
revI = df["data_source_type"] == "REVENUE"
df.loc[costI, "rev"] = np.NaN
df.loc[revI, "max_cpc"] = np.NaN

df["costI"] = costI
df["revI"] = revI
assert df["costI"].sum() + df["revI"].sum() == len(df)

#### PROCESS DATE ####
df['date'] = pd.to_datetime(df['date'])
df["weekday"] = df['date'].dt.weekday
df['today'] = TODAY
df['days_back'] = (NOW - df['date'].dt.tz_localize("EST")).dt.days

# #### FIX DUPLICATE DATA ISSUES ####
# """
# CHANGED:
# - the way the indices and joins are applied to deduplicate rev
# - previously this expanded df from 500k => 700k records b/c of outer join 
#     behavior for duplicated column tuples
#         - this resulted in overreporting clicks by 50%
# - still not 100% this is correct behavior - but at least df size is cut down
#     instead of increased
# """
# df_match = df\
#     [["account","campaign_id","adgroup_id","keyword_id","match"]]\
#     [df["data_source_type"]=="COST"] \
#     .drop_duplicates()
# df_match["unique_match_types"] = df_match \
#     .groupby(["account","campaign_id","adgroup_id","keyword_id"]) \
#     .transform("count")["match"]
# dup_match_I = df_match["unique_match_types"] > 1
# phrase_match_I = df_match["match"] == "Phrase"
# join_key = ["account","campaign_id","adgroup_id","keyword_id","match"]
# df = df.merge(
#     df_match.loc[~(dup_match_I & phrase_match_I),join_key],
#     on=join_key, how='right')

# print("|df|", df.shape)

# """
# CHANGED
# - whatever was going on here
# - couldnt figure out what was being done
# - but had the same problem where df was getting blown up b/c of outer join 

# TODO: 
# - verify 
# """
# #%%
# #### FIX ISSUE WITH MATCH TYPE #####
# print("|df|", df.shape)
# join_key = ["account","campaign_id","adgroup_id","keyword_id"]
# df = pd.merge(df,df_match,how="left", on=join_key,suffixes=("_l",""))
# print("|df|", df.shape)

# I = df["match"] == df['match_l']
# I.mean(),I.sum()


### CREATE SEPARATE GROUPINGS FOR GEO-GRANULAR ADGROUPS AND KEYWORDS ###
"""
CHANGED:
- previously geo grandular keywords were handled in separate script
- now they are grouped and handled along w/ the rest of the campaign keywords
"""
"""
Normalize keywords and adgroups for geo-granular campaigns 
so that data grouped on these adgroups/keywords may be shared w/in campaigns
"""
geoI = df['campaign'].str.contains("Geo-Granular")
print("geoI.isna()", geoI.isna().mean(), geoI.isna().sum())
geoI = geoI.fillna(False)
print("geoI", geoI.mean(), geoI.sum())
df[["adgroup_norm","keyword_norm"]] = df[["adgroup","keyword"]]
df.loc[geoI,"keyword_norm"] = df.loc[geoI,"keyword"].str.replace(r"+","")

# find keyws for granular geo data
states_to_clean = [
    "[Aa]labama","[Aa]laska","[Aa]rizona", "[Aa]rkansas","[Cc]alifornia","[Cc]olorado","[Cc]onnecticut",
    "[Dd]elaware","DC","dc","[Ff]lorida","[Gg]eorgia","[Hh]awaii", "[Ii]daho", "[Ii]llinois", "[Ii]ndiana",
    "[Ii]owa", "[Kk]ansas", "[Kk]entucky","[Ll]ouisiana", "[Mm]aine", "[Mm]aryland", "[Mm]assachusetts", 
    "[Mm]ichigan", "[Mm]innesota", "[Mm]ississippi", "[Mm]issouri", "[Mm]ontana", "[Nn]ebraska",
    "[Nn]evada", "[Nn]ew[ -][Yy]ork", "\\+[Nn]ew \\+[Yy]ork", "[Oo]hio", "[Oo]klahoma", "[Oo]regon", "[Pp]ennsylvania", 
    "\\+?[Rr]hode[- ]\\+?[Ii]sland","\\+?[Ss]outh[- ]\\+?[Dd]akota",
    "\\+?[Nn]orth[ -]\\+?[Dd]akota","\\+?[Nn]ew[ -]\\+?[Hh]ampshire",
    "\\+?[Nn]ew[ -]\\+?[Jj]ersey","\\+?[Nn]ew[ -]\\+?[Mm]exico","\\+?[Nn]orth[ -]\\+?[Cc]arolina","\\+?[Ss]outh[ -]\\+?[Cc]arolina",
    "[Tt]ennessee", "\\+?[Ww]est[ -]\\+?[Vv]irginia",
    "[Tt]exas", "[Uu]tah", "[Vv]irginia", "[Vv]ermont",
    "[Ww]ashington", "[Ww]isconsin", "[Ww]est[ -][Vv]irginia","\\+[Ww]est \\+[Vv]irginia","[Ww]yoming"]

import tqdm
#replace the state names in adg and kws
for state in tqdm.tqdm(states_to_clean):
    df.loc[geoI, "adgroup_norm"] = df.loc[geoI, "adgroup_norm"] \
        .str.replace(state, "state", regex=True)
    df.loc[geoI, "keyword_norm"] = df.loc[geoI, "keyword_norm"] \
        .str.replace(state, "state", regex=True)

I = df[["keyword","adgroup"]].values == df[["keyword_norm","adgroup_norm"]].values
assert I[~geoI].mean() == 1
assert I[geoI].mean() < 1e-3 # i think all of these adgps and keywords should be modified
# TODO: add a check for the regex replace and the `+` replace

# NOTE: grouping by campaign/adgroup/keyword strs creates more rows
#       than grouping by their ids
#   - maybe this is b/c the strs may be changed over time?
#   - means that grouping on str may be artificially restricting data
# TODO: assign most recent campaign/adgroup/kw str to each respective id
#   - write check to determine if that then makes grouping by str less
#       granular than grouping by id
camp_idx_C = ["account", "campaign_id"]
adgp_idx_C = ["account", "campaign_id", "adgroup_norm"]
kw_idx_C = ["account", "campaign_id", "adgroup_norm", "keyword_norm"]
# create a key that uniquely specifies the bid values we want to write to bing
bid_idx_C = ["account","campaign_id","adgroup_id","keyword_id","match"]
df["bid_key"] = df.groupby(bid_idx_C).ngroup()
assert df[bid_idx_C].drop_duplicates().__len__() == \
    df[[*bid_idx_C, "bid_key"]].drop_duplicates().__len__()
# """
# CPC field max_cpc is populated from CurrentMaxCpc from the raw table "bing_keyword" (edited) 
# 1:29
# if you need to se bing by hour during the day you can use the table tron_profitability_by_hour
# 1:29
# for bing the raw table is "bing_campaignbyhour_performance"
# 1:30
# both tables are in aurora data base log_upload
# """
# #%%
# from ds_utils.db.connectors import AnalyticsDB
# with AnalyticsDB() as db:
#     sql_df = db.to_df(
#         "select * from log_upload.bing_keyword limit 1000"
#     )
# sql_df.shape
# #%%
# costI = df["data_source_type"] == "COST"
# revI = df["data_source_type"] == "REVENUE"
# df.loc[costI,"rev"] = np.NaN
# df.loc[revI,"max_cpc"] = np.NaN
# for c in ["date","weekday"]:
#     dfagg = df \
#         [[c, "max_cpc", "rev"]] \
#         .groupby(c) \
#         [["max_cpc","rev"]] .mean()
#     (dfagg / dfagg.mean()).plot()
#     from matplotlib import pyplot as plt
#     plt.title(f"mean normalized rev and reported max_cpc against {c}")
#     plt.show()
# #%%
# c = "date"
# dfagg = df \
#     [[c, "max_cpc", "rev"]] \
#     .groupby(c) \
#     [["max_cpc","rev"]] \
#     .mean() \
#     .sort_index()
# dfagg["rev_7day"] = dfagg["rev"].rolling(7).mean()
# (dfagg / dfagg.mean())[["max_cpc","rev_7day"]].plot()
# from matplotlib import pyplot as plt
# plt.title(f"mean normalized 7-day rolling rev and reported max_cpc against {c}")
# plt.show()
# 
# df \
#     .groupby(bid_idx_C+adgp_idx_C)["max_cpc"].mean() \
#     .groupby(adgp_idx_C).std()
#%%
#### PREPARED YESTERDAY DATA FOR USE IN BID CHANGES ####

#store keyword attributes including match type
df_attributes = df \
    [[  "bid_key",
        "account","match",
        "campaign_id", 'adgroup_id', "keyword_id",
        "campaign", "adgroup", "keyword",
        "adgroup_norm", "keyword_norm", ]] \
    .drop_duplicates(subset=["bid_key"])

"""
CHANGED:
- previously we only used 1 day of past data to estimate cpc
- now we use 7 days of past data to estimate cpc
- we update bids for kws with data in the lookback window - this means 
    we update many more kws as lookback window is 7x greater
- NOTE: consider scenario where we have on day of clicks for a particular keyword
    - we will then update that keyword for 7 days based on cpc data for just 1 day
- NOTE: why do this? b/c our mtd of computing cpc: cost/clicks
        is going to be heavily influenced by our TOD modifiers
    - specifically - we dont want monday cpc updates to use sunday cpc as baseline
    - and vice versa for Saturday & Friday
"""
DATE_WINDOW = 7
# get rev,click,cost sums for past week
df_bid_perf = df \
    [(df["days_back"] <= DATE_WINDOW)] \
    .groupby(["bid_key",*kw_idx_C]) [["clicks",'rev','cost']] .agg(sum) \
    .groupby(kw_idx_C) .transform(sum)
assert df_bid_perf.__len__() == \
    df_bid_perf.reset_index().drop_duplicates(subset="bid_key").__len__()

# get latest bids for last week kws
"""
Q: what is the `max_cpc` col in tron.intraday_profitability?
    - the `max_cpc` level for taboolas GSP auction
Q: does `max_cpc` apply bid modifiers => NO
Q: is it an aggreagtion? 
    => NO - it is the actual val for this kw at the time of the reporting
Q: how is it populated?
    =>
    Aurora.log_upload.bing_keyword => 
    Aurora.log_upload.bing_campaignbyhour_performance =>
    Aurora.log_upload.tron_profitability_by_hour => 
    RedshiftHC.tron.intraday_profitability
Q: what is the frequency of each of the above ETLs?

Q: Does it refresh for every keyword or just ones with impressions the day before? I think that may be why he didn't use it before
https://healthcareinc.slack.com/archives/C020R24TXL1/p1620413974069700

TODO:
- pull in geo-granular bids from other locations if data sparse 
"""
df_bid = df \
    [(df["days_back"] <= DATE_WINDOW) & (df["max_cpc"] > 0)] \
    [["transaction_date_time", "max_cpc", "bid_key"]] \
    .drop_duplicates()
latest_bid_I = df_bid["transaction_date_time"] == df_bid.groupby("bid_key")['transaction_date_time'].transform(max)
df_bid = df_bid[latest_bid_I]

#join bids with perf data
df_bid = pd.merge(df_bid_perf, df_bid, how="left", on=["bid_key"])

#jon kw attributes
df_bid = pd.merge(df_bid,df_attributes, how="left", on=["bid_key"])

#keep only kws with clicks last week
df_bid = df_bid[df_bid["clicks"] > 0]

# """
# Q: 
# - so - cpc unique by bid
# - multiple bids in kw grouping
# - is cpc ever diff w/in a kw grouping?
# => A: yes but this could happen b/c of ...?
# @Curtis said its probably fine
# """
# df_bid.groupby(kw_idx_C)["max_cpc"].std()
#%%
#### CREATE AGGREGATIONS FOR USE IN RPC ###
# simplify df
df_rpc = df \
    .groupby(["days_back","match",*kw_idx_C]) \
    ["clicks","rev","cost"] \
    .sum()
#find adg, campaign level data for rpc estimation
df_rpc[["clicks_adg", "rev_adg"]] = df_rpc \
    .groupby(["days_back","match",*adgp_idx_C]) \
    [["clicks","rev"]] .transform(sum)
df_rpc[['clicks_cmp', 'rev_cmp']] = df_rpc \
    .groupby(["days_back","match",*camp_idx_C]) \
    [["clicks", "rev"]] .transform(sum)
df_rpc[["clicks_act","rev_act"]] = df_rpc \
    .groupby(["days_back","match","account"]) \
    [["clicks", "rev"]] .transform(sum)

#### FIND DECAY MULTIPLIER ####
decay_factor = 0.03 
df_rpc["decay_multiplier"] = np.exp(-decay_factor * df_rpc.index.get_level_values("days_back"))

"""
TODO:
Q: how to deal w/ bid trap prpbolem?
- initla soln: 
    - validate this data cycling and rollup mtd for escaping bid trap
- ocntext
    - bid trap scenarios
        1. bid down little too aggressive - could nudge bid back up and make kw profitable
        2. correctly identifies a poor kw
    - only soln is explore/exploit split
- down the line we can do that explot/explore split
"""
def find_rpc(df_kw):
    df_kw = df_kw * df_kw["decay_multiplier"].values.reshape(-1,1)

    #remove clicks and rev from successive levels
    df_kw["clicks_act"] = df_kw["clicks_act"]-df_kw["clicks_cmp"]
    df_kw["clicks_cmp"] = df_kw["clicks_cmp"]-df_kw["clicks_adg"]
    df_kw["clicks_adg"] = df_kw["clicks_adg"]-df_kw["clicks"]

    df_kw["rev_act"] = df_kw["rev_act"]-df_kw["rev_cmp"]
    df_kw["rev_cmp"] = df_kw["rev_cmp"]-df_kw["rev_adg"]
    df_kw["rev_adg"] = df_kw["rev_adg"]-df_kw["rev"]

    clickC  = ["clicks","clicks_adg","clicks_cmp","clicks_act",]
    revC    = ["rev","rev_adg","rev_cmp","rev_act"]
    clicks_arr = np.concatenate(df_kw[clickC].values.T)
    rev_arr = np.concatenate(df_kw[revC].values.T)

    # add a final fallback grouping of 0.5 * rpc
    # clicks_arr = np.append(clicks_arr, df_kw["clicks"].sum())
    # rev_arr = np.append(rev_arr, df_kw["rev"].sum()/2)

    df3 = pd.DataFrame(np.stack([clicks_arr,rev_arr]),index=["clicks","rev"]).T
    df3["clicks_cum"] = df3["clicks"].cumsum()
    # #choose what data to use based on click thresold
    max_row = np.searchsorted(df3["clicks_cum"],CLICKS)
    # TODO: unsure about this mtd of restricting influence of less granular aggrgations
    # limit number of clicks in last row
    if(max_row < len(df3)):
        clicks_to_backfill = CLICKS - df3["clicks"].iloc[:max_row].sum()
        backfill_portion = min(1, clicks_to_backfill / df3.loc[max_row, "clicks"])
        df3.loc[max_row,["rev","clicks"]] *= backfill_portion
    # NOTE: this method wont stop u from making adjustments if there is no data at 
    #       highest level - accnt level - but --- I mean - would that ever happen?
    df3 = df3.iloc[:max_row+1]
    #find and output rpc
    rpc = df3["rev"].sum()/df3["clicks"].sum()
    return rpc

#### FIND RPC ####
df_bid["rpc_est"] = df_bid[kw_idx_C] \
    .apply(lambda r: (slice(None), slice(None), *r), axis=1) \
    .apply(lambda kw_slc: find_rpc(df_rpc.loc[kw_slc]))
#%%
"""
cpc_t = cost_t/clicks_t
roi_t = rpc_t / cpc_t
ASSUME RPC IS CONSTANT
cpc_target = rpc_constant / roi_target
TODO: 
- what if rpc == 0? - do we really want to set cpc == 0
- wont that guarantee we never give this kw another change?
- NOTE: actually no - over time we will eventuall use the adgroup,campaign,account
        level stats instead of the kw level stats - so assuming at least one of those
        levels makes some revenue we will eventually start bidding >0 for this kw again

"""
#### FINALIZE BIDS ####
df_bid["cpc_observed"] = df_bid["cost"]/df_bid["clicks"]
#find cpc target
df_bid["cpc_target"] = df_bid["rpc_est"]/ROI_TARGET
(df_bid["cpc_observed"] > df_bid["max_cpc"]).mean()
#apply bids change rules
df_bid = df_bid.rename(columns={'max_cpc': 'max_cpc_old'})

"""
NOTE:
- 
"""
df_bid["bid_change"] = df_bid["cpc_target"] -  df_bid["cpc_observed"]
df_bid["max_cpc_new"] = df_bid["max_cpc_old"] + df_bid["bid_change"]
df_bid["perc_change"] = df_bid["max_cpc_new"] / df_bid["max_cpc_old"] - 1
df_bid.loc[df_bid["perc_change"] > MAX_PUSH,"perc_change"] = MAX_PUSH
df_bid.loc[df_bid["perc_change"] < MAX_CUT, "perc_change"] = MAX_CUT
df_bid["max_cpc_new"] = df_bid["max_cpc_old"] * (1 + df_bid["perc_change"])
df_bid.loc[df_bid["max_cpc_new"] < CPC_MIN, "max_cpc_new"] = CPC_MIN

#round bids
df_bid["max_cpc_new"] = df_bid["max_cpc_new"].round(2) 

#record change
df_bid["bid_change"] = df_bid["max_cpc_new"] - df_bid["max_cpc_old"]

#prepare output
df_out = df_bid[["account","campaign_id","adgroup_id","keyword_id","campaign","adgroup","keyword","match","clicks","rev","cost","max_cpc_old","max_cpc_new","rpc_est","cpc_target","bid_change","perc_change"]]
df_out = df_out.rename(columns={"clicks": "clicks_y"})
df_out = df_out.rename(columns={"rev": "rev_y"})
df_out = df_out.rename(columns={"cost": "cost_y"})

df_out = df_out.sort_values("clicks_y",ascending = False)

df_out["rev_y"] = df_out["rev_y"].round(2)  
df_out["cost_y"] = df_out["cost_y"].round(2)
#%%
#write csv to local/s3
df_out = df_out.set_index("account",drop=False)
for accnt in df_out.index.unique("account"):
    accnt_dir = f"{BIDS_DIR}/{accnt}"
    os.makedirs(accnt_dir,exist_ok=True)
    bids_fnm = f"BIDS_{TODAY}.csv"
    bids_fpth = f"{BIDS_DIR}/{accnt}/{bids_fnm}"
    df_out.to_csv(bids_fpth,index=False,encoding='utf-8')

    #### WRITE OUTPUT TO S3 ####
    s3_resource = boto3.resource('s3')

    s3_client = boto3.client('s3')
    response = s3_client.upload_file(
        bids_fpth,
        S3_OUTPUT_BUCKET,
        f"{S3_OUTPUT_PREFIX}/{accnt}/{bids_fnm}")
#%%
