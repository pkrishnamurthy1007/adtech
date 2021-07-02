#%%
"""
DAILY KEYWORD BIDDING ALGORITHM BING
"""

#### LOAD COMMON ####
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

#### Get reporting data for bidder ####

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
    date >= current_date - 84 AND
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
assert reporting_df["costI"].sum() + reporting_df["revI"].sum() == len(reporting_df)

#### PROCESS DATE ####
reporting_df['date'] = pd.to_datetime(reporting_df['date']) \
    .dt.tz_localize("UTC") \
    .dt.tz_convert("EST") \
    .dt.date
reporting_df["date"] = pd.to_datetime(reporting_df["date"])
reporting_df['today'] = TODAY

### PLOT ROLLING KPIS ###
kpiC = ["rev","cost","clicks","max_cpc"]
date_kpi_df = reporting_df .groupby("date") [kpiC].sum()
date_kpi_df["ROAS"] = date_kpi_df["rev"] / date_kpi_df["cost"]
kpiC += ["ROAS"]
for n in [3,7,14,30]:
    rolling_kpi_C = [f"{c}_{n}d_avg" for c in kpiC]
    date_kpi_df[rolling_kpi_C] = date_kpi_df[kpiC].rolling(n).mean()
    (date_kpi_df[rolling_kpi_C] / date_kpi_df[rolling_kpi_C].mean()).iloc[:,:-1].plot()

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

import uniplot
uniplot.plot([*date_kpi_df[roasC[1:]].fillna(1).values.T],
             legend_labels=roasC[1:],
             title=f"rolling ROAS",
             width=90,height=15)
#%%
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
geoI = reporting_df['campaign'].str.contains("Geo-Granular")
print("geoI.isna()", geoI.isna().mean(), geoI.isna().sum())
geoI = geoI.fillna(False)
print("geoI", geoI.mean(), geoI.sum())
reporting_df["geoI"] = geoI

# find keyws for granular geo data
STATES = [
    "Alaska", "Alabama", "Arkansas", 
    "American Samoa", "Samoa", 
    "Arizona", "California", "Colorado", "Connecticut", 
    "District of Columbia", "DC", "District Columbia",
    "Delaware", "Florida", "Georgia", 
    "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", 
    "Kansas", "Kentucky", "Louisiana", "Massachusetts", 
    "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", 
    "Mississippi", "Montana", "North Carolina", "North Dakota",
    "Nebraska", "New Hampshire", "New Jersey", "New Mexico",
    "Nevada", "New York", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Puerto Rico", "Rhode Island",
    "South Carolina", "South Dakota", "Tennessee", "Texas",
    "Utah", "Virginia", "Virgin Islands", "Vermont",
    "Washington", "Wisconsin", "West Virginia", "Wyoming",
    ]
"""
after discussion w/ @Curtis - we want the match types preserved
in the keyword string 

apparently the `+` prefix indicates a different match type for 
tokens w/in a keyword - and should be separated out into
a different geo granular keyword

e.g.:
'marketplace new york'      =>      'marketplace <STATE>'
'+marketplace +new +york'   =>      '+marketplace +<STATE>'

NOTE: in general 2 word states will have a `+` in the middle 
    if and only if they are preceded by a `+` 

NOTE: kw query
```
# adgp = df.loc[geoI,"adgroup"].unique()[0]
# adgpI = df["adgroup"] == adgp
camp = df.loc[geoI, "campaign_id"].unique()[0]
campI = df["campaign_id"] == camp
nyI = df['keyword'].str.lower().str.contains("york")
dashI = df['keyword'].str.lower().str.contains("-")
df.loc[geoI & campI & nyI,"keyword"] \
    .drop_duplicates() \
    .values
```

"""
STATE_TAG = "<STATE>"
def get_state_regex(state):
    # optionally match intravening but not preceding `+`
    state_regex = "[ -]\+?".join(state.split())
    # state_regex = f"\\b{state_regex}\\b"
    return state_regex
import re
state_regexes = [get_state_regex(state) for state in STATES]
unified_state_regex = re.compile(f"({'|'.join(state_regexes)})",re.IGNORECASE)

reporting_df[["adgroup_norm", "keyword_norm"]] = reporting_df[["adgroup", "keyword"]]
reporting_df.loc[geoI,"keyword_norm"] = reporting_df \
    .loc[geoI,"keyword"] \
    .str.replace(unified_state_regex,STATE_TAG)
reporting_df.loc[geoI,"adgroup_norm"] = reporting_df \
    .loc[geoI,"adgroup"] \
    .str.replace(unified_state_regex,STATE_TAG)

I = reporting_df[["keyword","adgroup"]].values == reporting_df[["keyword_norm","adgroup_norm"]]
I = (I.prod(axis=1) == 1) | reporting_df["keyword"].isna()
I.mean(),I[~geoI].mean(),I[geoI].mean() 
assert I[~geoI].mean() == 1
assert I[geoI].mean() == 0
assert all(reporting_df.loc[geoI,"keyword_norm"].str.contains(STATE_TAG))
assert all(reporting_df.loc[geoI,"adgroup_norm"].str.contains(STATE_TAG))

# NOTE: grouping by campaign/adgroup/keyword strs creates more rows
#       than grouping by their ids
#   - maybe this is b/c the strs may be changed over time?
#   - means that grouping on str may be artificially restricting data
# TODO: assign most recent campaign/adgroup/kw str to each respective id
#   - write check to determine if that then makes grouping by str less
#       granular than grouping by id
# TODO: why are some keyword_ids both geo-gran and not?
#   - guessing this has something to do w/ changes to adgroup & campaign names
camp_idx_C = ["account_id", "geoI", "campaign_id"]
adgp_idx_C = ["account_id", "geoI", "campaign_id", "adgroup_norm"]
kw_gp_idx_C = ["account_id", "geoI", "campaign_id", "adgroup_norm", "keyword_norm"]
kw_idx_C = ["account_id", "geoI", "campaign_id", "adgroup_id", "keyword_id"]
reporting_df["cnt"] = 1
geo_gran_kw_cnts = reporting_df \
    .drop_duplicates(kw_idx_C) \
    .groupby(kw_gp_idx_C)[["cnt"]] .count()
geo_gran_kw_cnts = geo_gran_kw_cnts.sort_values(by="cnt")
# TODO: eventuall want `all(geo_gran_kw_cnts["cnt"] == 51)`
# make sure any geo gran kw grouping w/ more than 51 kws
#   has the no longer used double state pattern
# TODO: should be > 51 - but breaking out bids by name 
#       creates some extra bid entries per bid
bad_kws = geo_gran_kw_cnts \
    [geo_gran_kw_cnts["cnt"] > 60] \
    .reset_index() ["keyword_norm"] 
assert all(bad_kws.str.count(STATE_TAG) > 1)

# reporting_df["bid_key"] = reporting_df[kw_idx_C].apply(tuple,axis=1).apply(str)
reporting_df["bid_key"] = reporting_df["keyword_id"]
#%%
DATE_WINDOW = 7
# get rev,click,cost sums for past week
performance_C = ["clicks",'rev','cost']
DAY = datetime.timedelta(days=1)
def n_day_performance(df,performance_C,n=7):
    return df \
        [df["date"].dt.date >= TODAY - n*DAY] \
        .groupby("bid_key") [performance_C] \
        .sum() \
        .rename(columns={c: f"{c}_sum_{n}day" for c in performance_C})
df_bid_perf = pd.concat((
        n_day_performance(reporting_df,performance_C,n=n)
        for n in [1,3,7,14,30]
    ),
    axis=1)
df_bid_perf[performance_C] = \
    df_bid_perf[[f"{c}_sum_{DATE_WINDOW}day" for c in performance_C]]
df_bid_perf.index.name = "bid_key"
ipydisp(df_bid_perf.sum())

df_bid = reporting_df[["bid_key",
                        "account_id", "account_num", "match",
                        "campaign_id", 'adgroup_id', "keyword_id",
                        "campaign", "adgroup", "keyword",
                        "adgroup_norm", "keyword_norm", 'geoI',
                        "max_cpc", "latest_max_cpc", ]] \
    .drop_duplicates(subset=["bid_key"],keep="last")
print("|df_bid|", df_bid.shape)

df_bid = pd.merge(
    df_bid,
    accnt_df[[
    "account_id",
    "click_threshold","roi_target","max_push","max_cut","cpc_min",
    "keyword_bidder_enabled","keyword_geo_bidder_enabled"]],
    how="left",
    on="account_id"
)
df_bid = df_bid[df_bid["keyword_bidder_enabled"] == "Y"]
print("|df_bid|", df_bid.shape)

#jon kw attributes
df_bid = pd.merge(df_bid, df_bid_perf, how="left", on=["bid_key"],suffixes=("","_"))
df_bid[[f"{c}_raw" for c in df_bid_perf.columns]] = df_bid[df_bid_perf.columns]
# agg-transform performance data accross geo gran keyword groups
df_bid .loc[df_bid["geoI"],df_bid_perf.columns] = \
    df_bid.loc[df_bid["geoI"]] \
        .groupby(kw_gp_idx_C)[df_bid_perf.columns] .transform(sum)
df_bid["latest_max_cpc"] = df_bid["latest_max_cpc"].fillna(df_bid["latest_max_cpc"].mean())
df_bid = df_bid.fillna(0)
print("|df_bid|",df_bid.shape)

df_bid["clicks_in_window"] = df_bid["clicks"] > 0
#keep only kws with clicks in aggregation window
df_bid = df_bid[df_bid["clicks"] > 0]
print("|df_bid|", df_bid.shape)

print("portion of reporting rows that are geo-granular:",reporting_df["geoI"].mean())
print("portion of keywords that are geo-granular:",
      reporting_df[["bid_key", "geoI"]].drop_duplicates()["geoI"].mean())
I = reporting_df.groupby("campaign")["geoI"].sum() > 0
print("portion of campaigns that are geo-granular:",I.mean())
print("portion of bid updates that are geo_granular:",df_bid["geoI"].mean())

assert all(df_bid.isna().sum() == 0), df_bid.isna().sum()

upsampled_rev_agg = df_bid[["geoI",*performance_C]].groupby("geoI").sum()
rev_agg = reporting_df \
    .loc[reporting_df["date"].dt.date >= TODAY-DATE_WINDOW*DAY, 
         ["geoI", *performance_C]] \
    .groupby("geoI").sum()
upsample_proportions = (upsampled_rev_agg - rev_agg).abs() / rev_agg
print("revenue upsampling broken out by geolocation bool:")
from IPython.display import display as ipydisp
ipydisp(upsample_proportions)

assert all(upsample_proportions.loc[False] < 0.5)
assert all(upsample_proportions.loc[True] > 40)
assert all(upsample_proportions.loc[True] < 60)
#%%
#### CREATE AGGREGATIONS FOR USE IN RPC ###
# simplify df
df_rpc = reporting_df \
    .groupby(["date","match",*kw_gp_idx_C]) \
    [["clicks","rev","cost"]] \
    .sum()
#find adg, campaign level data for rpc estimation
df_rpc[["clicks_adg", "rev_adg"]] = df_rpc \
    .groupby(["date","match",*adgp_idx_C]) \
    [["clicks","rev"]] .transform(sum)
df_rpc[['clicks_cmp', 'rev_cmp']] = df_rpc \
    .groupby(["date","match",*camp_idx_C]) \
    [["clicks", "rev"]] .transform(sum)
df_rpc[["clicks_act","rev_act"]] = df_rpc \
    .groupby(["date","match","account_id"]) \
    [["clicks", "rev"]] .transform(sum)

#### FIND DECAY MULTIPLIER ####
decay_factor = 0.03 
days_back = (TODAY - df_rpc.reset_index()["date"].dt.date).dt.days
df_rpc["decay_multiplier"] = np.exp(-decay_factor * days_back.values)

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
df_bid["rpc_est"] = df_bid[kw_gp_idx_C] \
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
- NOTE: actually no - over time we will eventuall use the adgroup,campaign,account_id
        level stats instead of the kw level stats - so assuming at least one of those
        levels makes some revenue we will eventually start bidding >0 for this kw again

"""
#### FINALIZE BIDS ####
df_bid["cpc_observed"] = df_bid["cost"]/df_bid["clicks"]
#find cpc target
df_bid["cpc_target"] = df_bid["rpc_est"]/ROI_TARGET
#apply bids change rules
df_bid = df_bid.rename(columns={'latest_max_cpc': 'max_cpc_old'})

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
df_out = df_bid
df_out = df_out.rename(
    columns={
        "clicks": "clicks_kw_group",
        "rev": "rev_kw_group",
        "cost": "cost_kw_group",
    })

df_out = df_out.sort_values("clicks_kw_group",ascending = False)

df_out["rev_kw_group"] = df_out["rev_kw_group"].round(2)  
df_out["cost_kw_group"] = df_out["cost_kw_group"].round(2)

def write_kw_bids_to_s3(df,accnt):
    accnt_dir = f"{OUTPUT_DIR}/{accnt}"
    os.makedirs(accnt_dir, exist_ok=True)
    bids_fnm = f"BIDS_{TODAY}.csv"
    bids_fpth = f"{OUTPUT_DIR}/{accnt}/{bids_fnm}"
    df.to_csv(bids_fpth, index=False, encoding='utf-8')

    #### WRITE OUTPUT TO S3 ####    
    s3_client = boto3.client('s3')
    response = s3_client.upload_file(
        bids_fpth,
        S3_OUTPUT_BUCKET,
        f"{S3_OUTPUT_PREFIX}/{accnt}/{bids_fnm}")
    return response

#write csv to local/s3
write_kw_bids_to_s3(df_out, "ALL_ACCOUNTS")
for accnt in df_out["account_id"].unique():
    write_kw_bids_to_s3(df_out[df_out["account_id"] == accnt],accnt)
#%%