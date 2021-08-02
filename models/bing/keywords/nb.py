#%%
import models.taboola.utils
import joblib
from models.common import *
from models.utils import wavg, get_wavg_by, wstd
from models.data.queries.session import *

from models.bing.keywords.common import *

start_date = TODAY - 90*DAY
end_date = TODAY

split_cols = ["product","campaign_id","adgroup_id","keyword"]
rps_df = agg_rps(start_date, end_date, None, traffic_source=BING,
                 agg_columns=tuple([*split_cols, "utc_dt"]))
rps_df = rps_df.reset_index()
rps_df
#%%
import models.taboola.utils
import importlib
importlib.reload(models.taboola.utils)
TaboolaRPSEst = models.taboola.utils.TaboolaRPSEst
# rps_model: TaboolaRPSEst = joblib.load(MODEL_PTH)
rps_model = TaboolaRPSEst(clusts=None,enc_min_cnt=10).fit(
    rps_df.set_index([*split_cols, "utc_dt"]), None)
rps_df["clust"] = rps_model.transform(rps_df.set_index([*split_cols, "utc_dt"]))
#%%
import pandas as pd
camp = "361640619"
adgp = "1283130453341861"
kwd = "medicare part b"
I = (rps_df["campaign_id"] == camp) & (rps_df["adgroup_id"] == adgp) & (rps_df["keyword"] == kwd)
df = rps_df[I].set_index("utc_dt")
(df['revenue'] / df['sessions']).rolling(7).mean().plot()
#%%
df['sessions'].rolling(7).mean().plot()
#%%
df['revenue'].rolling(7).mean().plot()
#%%
"""
we want rps estimation to do 2 contradictory things:
1. respond to short term changes in (rps,lps,rpl,etc....) 
2. use long range historical data to "revive" keywords

I think the process we use for rps estimation needs to be separate from the process
we use to "revive" low volume keywords
"""
self = rps_model
X = rps_df.set_index([*split_cols, "utc_dt"]).copy()
y = X["revenue"]
w = X["sessions"]
sample_thresh=100

X["clust"] = self.transform(X)
#%%
Xdf = X
X = X .reset_index()[X.index.names].iloc[:, :-1]
X = self.enc_1hot.transform(X)
print("|X|", X.shape)
P = self.clf.decision_path(X)
P
#%%

#%%
Pdf = pd.DataFrame(P.todense(),index=Xdf.index)
y_Pdf = Pdf * y.values.reshape(-1,1)
w_Pdf = Pdf * w.values.reshape(-1,1)
y_agg_Pdf = Pdf * y_Pdf.groupby("utc_dt").transform(sum)
w_agg_Pdf = Pdf * w_Pdf.groupby("utc_dt").transform(sum)

# SAMPLE_THRESH = 100
# I = (~(session_agg_Pdf < SAMPLE_THRESH)).iloc[:,::-1].idxmax(axis=1)
# I = np.eye(self.clf.tree_.node_count).astype(bool)[I]

# rev_rollup = (revenue_agg_Pdf * I).sum(axis=1)
# sess_rollup = (session_agg_Pdf * I).sum(axis=1)
# rps_rollup = rev_rollup / sess_rollup
# rps_rollup

def running_suffix_max(df):
    df_running_max = df.copy()
    H, W = df_running_max.shape
    for ci in reversed(range(W-1)):
        df_running_max.iloc[:, ci] = np.maximum(
            df_running_max.iloc[:, ci], df_running_max.iloc[:, ci+1])
    return df_running_max

y_contrib_Pdf = y_agg_Pdf - running_suffix_max(y_agg_Pdf).shift(-1,axis=1).fillna(0)
w_contrib_Pdf = w_agg_Pdf - running_suffix_max(w_agg_Pdf).shift(-1,axis=1).fillna(0)
y_contrib_Pdf = np.maximum(0,y_contrib_Pdf)
w_contrib_Pdf = np.maximum(0,w_contrib_Pdf)

"""
total_sess = 0
total_rev = 0
while total_sess < THRESH - scan up through decision tree path:
    rollup_factor = min(n.sessions,THRESH - total_sess) / n.sessions
    total_sess += n.sessions * rollup_factor
    total_rev += n.rev * rollup_factor
ROAS = total_rev / total_sess
"""
H,W = w_contrib_Pdf.shape
total_w = w_contrib_Pdf.iloc[:,-1]
total_y  = y_contrib_Pdf.iloc[:,-1]
import tqdm
for ni in tqdm.tqdm(reversed(w_contrib_Pdf.columns[:-1])):
    wni = w_contrib_Pdf.iloc[:, ni]
    yni = y_contrib_Pdf.iloc[:,ni]
    rollup_factor = np.clip((sample_thresh - total_w) / wni, 0, 1).fillna(0)
    total_w += wni * rollup_factor
    total_y += yni * rollup_factor 
#%%
rps_df["rps_est"] = rps_model.predict(
    rps_df.set_index([*split_cols, "utc_dt"]))

rps_df_campaign = rps_df[rps_df["utc_dt"].dt.date > TODAY - 7*DAY] \
    .groupby(["campaign_id"])[["rps_est"]] \
    .agg(get_wavg_by(rps_df, "sessions"))
rps_df_keyword = rps_df[rps_df["utc_dt"].dt.date > TODAY - 7*DAY] \
    .groupby(["campaign_id", "keyword"])[["rps_est"]] \
    .agg(get_wavg_by(rps_df, "sessions")) \
    .unstack()

# rps_df["clust"] = rps_model.transform(rps_df.set_index([*split_cols,"utc_dt"]))
# rps_df["clust_sessions"] = rps_df.groupby(["utc_dt","clust"])["sessions"].transform(sum)
# rps_df["clust_leads"] = rps_df.groupby(["utc_dt","clust"])["leads"].transform(sum)
# rps_df["clust_revenue"] = rps_df.groupby(["utc_dt","clust"])["revenue"].transform(sum)
#%%
rps_df["rps_est"]
#%%
df = rps_df.groupby(["product","campaign_id","adgroup_id","keyword"]).agg({
    "revenue": sum,
    "sessions": sum,
    "rps_est": get_wavg_by(rps_df,"sessions"),
}) .sort_values(by="revenue",ascending=False)
df
#%%
DAY = datetime.timedelta(1)
from matplotlib import pyplot as plt
kwd = "+molina +insurance"
# kwd = "+obamacare +cost"
I = reporting_df["keyword"] == kwd
reporting_df[I][["rev","cost","clicks"]].sum()
df = reporting_df[I].set_index("date").sort_index()[["rev","cost","clicks","max_cpc"]].rolling(7).sum()
df["ROAS"] = df["rev"] / df["cost"]
df.plot()
plt.xlim([TODAY - 90*DAY,TODAY])
plt.show()
(df/df.mean()).plot()
plt.xlim([TODAY - 90*DAY,TODAY])
plt.show()
#%%
from models.utils import get_wavg_by
kwdf = reporting_df \
        .groupby("keyword_id") \
        .agg({
            "keyword": "last",
            "rev": sum,
            "cost": sum,
            "clicks": sum,
            "latest_max_cpc": "last",
            "max_cpc": get_wavg_by(reporting_df,"cost"),
        })
kwdf = kwdf.sort_values(by="cost",ascending=False)

def get_kwnd(reporting_df,n):
    kwnd = reporting_df \
        [reporting_df["date"].dt.date > TODAY - n*DAY] \
        .groupby("keyword_id") \
        [["rev","cost","clicks"]].sum() \
        .reindex(reporting_df['keyword_id'].unique())
    kwnd.columns = [f"{c}{n}d" for c in kwnd.columns]
    return kwnd
kwdf = pd.concat((
            kwdf, get_kwnd(reporting_df, 7), get_kwnd(reporting_df, 30),
            get_kwnd(reporting_df,60), get_kwnd(reporting_df,90)),
        axis=1)

profitableI = (kwdf["rev"] / kwdf["cost"]) > 0.95
deadI = kwdf['clicks7d'] < (kwdf['clicks60d'] * 7 / 60 * 0.5)
#%%
kwdf[profitableI & deadI].sum()
#%%
kwdf[profitableI & deadI]
#%%
kwdf
#%%
kwdf = kwdf .join(
    df_bid.set_index("keyword_id") \
        [["adgroup","keyword","rpc_est","cpc_observed","cpc_target","max_cpc_old","max_cpc_new"]],
    how='left',rsuffix="_") \
    .sort_values(by="cost",ascending=False)
kwdf[profitableI & deadI].head(20)
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
                        "latest_max_cpc", ]] \
    .drop_duplicates(subset=["bid_key"],keep="last")
print("|df_bid|", df_bid.shape)
# remove entries w/ NULL account metadata entries
df_bid = df_bid[~df_bid[kw_gp_idx_C].isna().any(axis=1)]
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

assert all(upsample_proportions.loc[False] < 0.1)
assert all(upsample_proportions.loc[True] > 35)
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
df_rpc = df_rpc.sort_index(level="date",ascending=False)

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
# TODO: write check validating `rpc_est`
# df = df_bid[df_bid["clicks"] > 120][["rpc_est","clicks","rev","cost"]]
# df["rpc_est_"] = df["rev"] / df["clicks"]
# df
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
#find cpc target
df_bid["cpc_observed"] = df_bid["cost"] / df_bid["clicks"]
df_bid["cpc_target"] = df_bid["rpc_est"] / ROI_TARGET
#apply bids change rules
df_bid = df_bid.rename(columns={'latest_max_cpc': 'max_cpc_old'})
df_bid["max_cpc_new"] = np.clip(
    df_bid["cpc_target"],
    df_bid["max_cpc_old"] * (1 + MAX_CUT),
    df_bid["max_cpc_old"] * (1 + MAX_PUSH))
df_bid.loc[df_bid["max_cpc_new"] < CPC_MIN, "max_cpc_new"] = CPC_MIN
#round bids
df_bid["max_cpc_new"] = df_bid["max_cpc_new"].round(2) 
#record change
df_bid["bid_change"] = df_bid["max_cpc_new"] - df_bid["max_cpc_old"]
#%%
DAY = datetime.timedelta(1)
from matplotlib import pyplot as plt
kwd = "+molina +insurance"
# kwd = "+obamacare +cost"
I = reporting_df["keyword"] == kwd
reporting_df[I][["rev","cost","clicks"]].sum()
df = reporting_df[I].set_index("date").sort_index()[["rev","cost","clicks","max_cpc"]].rolling(7).sum()
df["ROAS"] = df["rev"] / df["cost"]
df.plot()
plt.xlim([TODAY - 90*DAY,TODAY])
plt.show()
(df/df.mean()).plot()
plt.xlim([TODAY - 90*DAY,TODAY])
plt.show()
#%%
from models.utils import get_wavg_by
kwdf = reporting_df \
        .groupby("keyword_id") \
        .agg({
            "keyword": "last",
            "rev": sum,
            "cost": sum,
            "clicks": sum,
            "latest_max_cpc": "last",
            "max_cpc": get_wavg_by(reporting_df,"cost"),
        })
kwdf = kwdf.sort_values(by="cost",ascending=False)

def get_kwnd(reporting_df,n):
    kwnd = reporting_df \
        [reporting_df["date"].dt.date > TODAY - n*DAY] \
        .groupby("keyword_id") \
        [["rev","cost","clicks"]].sum() \
        .reindex(reporting_df['keyword_id'].unique())
    kwnd.columns = [f"{c}{n}d" for c in kwnd.columns]
    return kwnd
kwdf = pd.concat((
            kwdf, get_kwnd(reporting_df, 7), get_kwnd(reporting_df, 30),
            get_kwnd(reporting_df,60), get_kwnd(reporting_df,90)),
        axis=1)

profitableI = (kwdf["rev"] / kwdf["cost"]) > 0.95
deadI = kwdf['clicks7d'] < (kwdf['clicks60d'] * 7 / 60 * 0.5)
#%%
kwdf[profitableI & deadI].sum()
#%%
kwdf[profitableI & deadI]
#%%
kwdf
#%%
kwdf = kwdf .join(
    df_bid.set_index("keyword_id") \
        [["adgroup","keyword","rpc_est","cpc_observed","cpc_target","max_cpc_old","max_cpc_new"]],
    how='left',rsuffix="_") \
    .sort_values(by="cost",ascending=False)
kwdf[profitableI & deadI].head(20)
#%%

"""
rps = rpl * lpc
    = rpl_short * rpl_mod_long * lpc_short * lpc_mod_long
tree rollup
rollup through time
"""
#%%
# 361640621
# 1282030941525812
# #%%
# df_out[df_out["campaign_id"] == "361640621"]
# #%%
# I = df_out["campaign_id"] == "361640621"
# r = df_out[I].sort_values(by="cost_raw").iloc[-1][kw_gp_idx_C]
# kw_slc = (slice(None), slice(None), *r)
# df_kw = df_rpc.loc[kw_slc]
# df_kw
# #%%
# I = reporting_df["campaign_id"] == "361640621"
# df = reporting_df[I].groupby("date")[["cost","rev"]].sum()
# df["roas"] = df["rev"] / df["cost"]
# # df["roas"].plot()
# df[["cost","rev"]].plot()
# #%%
# I = reporting_df["adgroup_id"] == "1282030941525812"
# df = reporting_df[I].groupby("date")[["cost","rev"]].sum()
# df["roas"] = df["rev"] / df["cost"]
# # df["roas"].plot()
# df[["cost","rev"]].plot()
# #%%
# df = reporting_df.groupby(["campaign_id","date"]) \
#     [["cost","rev"]] .sum()
# df = df.groupby("campaign_id") \
#     .apply(lambda df:
#                 df.reset_index("campaign_id",drop=True) \
#                     .reindex(pd.date_range(TODAY-90*DAY,TODAY)) \
#                         .rolling(7).sum()
#     )
# df["roas"] = df["rev"] / df["cost"]
# for cid in df.index.unique("campaign_id"):
#     df.loc[cid]["roas"].plot()
# #%%

#%%
from api.bingads.bingapi.client import BingClient, LocalAuthorizationData
import datetime,pytz
NOW = datetime.datetime.now(pytz.timezone('EST'))
NOW_ = datetime.datetime.now(pytz.timezone('US/Eastern'))
print(NOW,NOW_)
#%%
ADTECH_ENV_SECRET = "SM_ENV_BASE"

import boto3
import json

secretsmanager = boto3.client('secretsmanager')
secret = secretsmanager.get_secret_value(SecretId='data_science_mysql_analytics_login')

import json
import base64
environ_bytes = \
    base64.b64encode(
        json.dumps(
            {**os.environ}).encode('utf-8'))
try:
    resp = secretsmanager.create_secret(
        Name=ADTECH_ENV_SECRET,
        SecretBinary=environ_bytes,
    )
except Exception as ex:
    print(f"Creating secret failed w/ {type(ex)}: {ex} - attempting update")
    resp = secretsmanager.put_secret_value(
        SecretId=ADTECH_ENV_SECRET,
        SecretBinary=environ_bytes,
    )
    print("...Successs!!")

import boto3
import json
import os

secretsmanager = boto3.client('secretsmanager')
sm_env_base_secret = secretsmanager.get_secret_value(SecretId=ADTECH_ENV_SECRET)
sm_env_base = json.loads(base64.b64decode(sm_env_base_secret["SecretBinary"]))
# already set os env vars take precendence over aws vals
os.environ.update({**sm_env_base, **os.environ})
#%%
sm_env_base_secret
#%%
query = f"""
SELECT 
    count(*)
FROM tron.intraday_profitability
"""
from ds_utils.db.connectors import HealthcareDW
# TODO: https://healthcareinc.slack.com/archives/C01VBRTJ4R5/p1619728061020600
with HealthcareDW() as db:
    print(db.to_df(query))
# %%
env = "REDSHIFT_PH_USER"
old_val = os.getenv(env)
new_val = "Asdfafdasfd"
os.environ.update({env: new_val,**os.environ})
os.getenv(env)
# %%
os.environ.update({env: old_val})

# %%
import datetime
now = datetime.datetime.now()
now.__str__()
# %%
now.date().__str__()
# %%
from utils.env import load_env_from_aws
load_env_from_aws()
import json,logging
bing_creds = json.loads(os.getenv("BING_CREDS"))
LOGLEVEL = logging.WARN
from api.bingads.bingapi.client import BingClient

accnt_id = "3196099"

accnt_client = BingClient(
        account_id=accnt_id,
        customer_id=bing_creds['BING_CUSTOMER_ID'],
        dev_token=bing_creds['BING_DEVELOPER_TOKEN'],
        client_id=bing_creds['BING_CLIENT_ID'],
        # refresh_token=None,
        refresh_token=bing_creds['BING_REFRESH_TOKEN'],
        loglevel=LOGLEVEL,
    )

from api.bingads.bingapi.client import BingClient,LocalAuthorizationData
from models.bing.keywords.common import *

# Debug requests and responses
LOGLEVEL = logging.WARN
logging.basicConfig(level=LOGLEVEL)
logging.getLogger('suds.client').setLevel(logging.DEBUG)
logging.getLogger('suds.transport.http').setLevel(logging.DEBUG)

camps = accnt_client.get_campaigns()
camps
# %%
accnt_client.authentication._oauth_tokens.__dict__
#%%
accnt_client.authentication._oauth_scope
#%%
accnt_client.authentication._require_live_connect
# %%
accnt_client = BingClient(
    account_id=accnt_id,
    customer_id=bing_creds['BING_CUSTOMER_ID'],
    dev_token=bing_creds['BING_DEVELOPER_TOKEN'],
    client_id=bing_creds['BING_CLIENT_ID'],
    # refresh_token=None,
    # refresh_token=bing_creds['BING_REFRESH_TOKEN'],
    refresh_token=accnt_client.authentication._oauth_tokens._refresh_token,
    loglevel=LOGLEVEL,
)

# Debug requests and responses
LOGLEVEL = logging.WARN
logging.basicConfig(level=LOGLEVEL)
logging.getLogger('suds.client').setLevel(logging.DEBUG)
logging.getLogger('suds.transport.http').setLevel(logging.DEBUG)

camps = accnt_client.get_campaigns()
camps

# %%
accnt_client.authentication._oauth_tokens.__dict__
#%%
bing_creds["BING_REFRESH_TOKEN"] = accnt_client.authentication._oauth_tokens._refresh_token
os.environ["BING_CREDS"] = json.dumps(bing_creds)
from utils.env import dump_env_to_aws
dump_env_to_aws()
# %%
bing_creds
# %%
import bingads
bingads.__version__
# %%
