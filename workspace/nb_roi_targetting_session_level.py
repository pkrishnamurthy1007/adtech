#%%
import numpy as np
import jmespath
from pytaboola.services.report import \
    CampaignSummaryReport, RecirculationSummaryReport, \
    TopCampaignContentReport, RevenueSummaryReport, VisitValueReport
import json
from pkg_resources import resource_filename as rscfn
import os
import tqdm
import itertools
from pytaboola.services import CampaignService
from pytaboola.services import AccountService
from pytaboola import TaboolaClient
from typing import Iterable
from ds_utils.db.connectors import HealthcareDW

import datetime
NOW = datetime.datetime.now()
DAY = datetime.timedelta(days=1)

start_date = NOW - 14*DAY
end_date = NOW - 0*DAY

start_date_ymd = start_date.strftime("%Y%m%d")
end_date_ymd = end_date.strftime("%Y%m%d")
product = None
traffic_source = None
start_date_ymd, end_date_ymd

TABOOLA = "TABOOLA"
O65 = 'MEDICARE'

start_date = start_date_ymd
end_date = end_date_ymd
product = O65
traffic_source = TABOOLA
product_filter = "" if product is None else \
                f"AND UPPER(s.product) = UPPER('{product}')"
traffic_filter = "" if traffic_source is None else \
                f"AND UPPER(s.traffic_source) = UPPER('{traffic_source}')"
timeofday_query = f"""
    with
        rps as (
            SELECT
                session_id,
                sum(revenue) AS revenue
            FROM tron.session_revenue s
            WHERE session_creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
            {traffic_filter}
            GROUP BY 1
        ),
        ip_locs as (
            SELECT 
                l.country_iso_code,
                l.time_zone,
                l.subdivision_1_iso_code,
                b.netowrk_index,
                b.start_int,
                b.end_int
            FROM 
                data_science.maxmind_ipv4_geo_blocks AS b
                JOIN data_science.maxmind_geo_locations AS l
                    ON b.maxmind_id = l.maxmind_id
        ),
        rps_tz_adj as (
            SELECT
                s.*,
                s.creation_date AS utc_ts,
                extract(HOUR FROM convert_timezone('UTC', l.time_zone, s.creation_date) - s.creation_date)::INT AS utc_offset,
                l.time_zone,
                convert_timezone('UTC', l.time_zone, s.creation_date) AS user_ts,
                l.subdivision_1_iso_code AS state,
                r.revenue
            FROM 
                tracking.session_detail AS s
                JOIN ip_locs as l
                    ON ip_index(s.ip_address) = l.netowrk_index
                    AND inet_aton(s.ip_address) BETWEEN l.start_int AND l.end_int
                    AND l.country_iso_code = 'US'
                JOIN rps as r
                    ON s.session_id = r.session_id
            WHERE nullif(s.ip_address, '') IS NOT null
            AND s.creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
            {product_filter}
            {traffic_filter}
        )
    SELECT
        1 as cnt,
        *
    FROM rps_tz_adj
"""
with HealthcareDW() as db:
    df = db.to_df(timeofday_query)

df['int_ix'] = range(len(df))
revdf = df

import pandas as pd
import urllib.parse
def parse_url_series(urlS: pd.Series) -> pd.DataFrame:
    parsedS = urlS \
        .fillna("_") \
        .apply(urllib.parse.urlparse)
    parseddf = parsedS \
        .apply(lambda url: pd.Series(url._asdict()))
    hostS = parsedS.apply(lambda url: url.netloc)
    pathS = parsedS.apply(lambda url: url.path)
    querydf = parsedS \
        .apply(lambda url: urllib.parse.parse_qs(url.query)) \
        .apply(lambda d: {k:v[0] for k,v in d.items()}) \
        .apply(pd.Series)
    # urldf = pd.concat((hostS,pathS,querydf),axis=1)
    urldf = pd.concat((parseddf,querydf),axis=1)
    return urldf
source_url_df = parse_url_series(revdf["source"])
redir_url_df = parse_url_series(source_url_df["redir"])
landing_page_url_df = parse_url_series(revdf["landing_page"])

# eqdf = (redir_url_df == landing_page_url_df)
# I = (1-eqdf).sum(axis=1) > 0 
# redir_url_df[I]
# landing_page_url_df[I]

revdf = pd.concat((revdf,source_url_df,redir_url_df,landing_page_url_df),axis=1)
revdf.to_csv("_.csv")
# #%%
# with HealthcareDW() as db:
#     print(db.to_df(f"""
#         select count(distinct session_id) from tron.session_revenue
#         WHERE session_creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
#         """))
# #%%
# with HealthcareDW() as db:
#     print(db.to_df(f"""
#         select count(distinct session_id) from tracking.session_detail
#         WHERE creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
#         """))
#%%
"""
\sum_{r^i} = s = 1 + r + r^2 + ....
s*r = r + r^2 + r^3 + ... = s-1
s * r = s - 1 ===> s = 1 / (1-r)
s - 1 = 1 / (1-r) - 1 = r / (1-r)
r \approx (window-1)/window

ema(X,t) = (1-r)*X[t] + r*ema(X,t-1)
"""
S = []
window = 5
L = S.__len__()
r = (window-1)/window
EMAker = (1-r) * 1/(1-r**window) * np.array([r**i for i in range(window)])
assert abs(EMAker.sum() - 1) < 1e-10
T = np.convolve([*S, *S, *S], EMAker, mode="same")[L:-L]

#%%
"""
- ROI = RPC / CPC
- bid mods = [TOD mods|loc mods|publisher mods|platform mods|os mods] 
- dROI/dbid = dRPC/dbid 

- d(f(x)g(x)) = df(x)g(x) + f(x)dg(x)
- d(1/x) = -1/x^2
- d(1/f(x)) = 1/f(x)^2 * df(x)/dx
- d(f(x)/g(x)) = df(x)*1/g(x) - f(x)dg(x)/g(x)^2
    = (df(x)g(x) - f(x)dg(x))/g(x)^2

f = RPC
g = CPC

dROI/dbid = (dRPC*CPC - RPC*dCPC)/CPC^2
CPC = bid, bid = bid_base + bid_mod => 
dCPC/dbid = dCPC/dbase = dCPC/dbid = 1

dROI/dbid = d(RPC/CPC)
    = (dRPC*CPC - RPC)/CPC^2
    = dRPC/CPC - RPC/CPC^2 = dRPC/CPC - ROI/CPC 
    = (dRPC - ROI)/CPC

current mtd - track ROI target via online step:
    - if ROI > target => decrease ROI => increase bid
    - elif ROI < target => increase ROI => decrease bid
    - assume RPC_target = RPC, CPC == bid
    ROI_target = RPC / CPC_target => CPC_target = RPC/ROI_target
    ROI[t+1] = ROI[t] - \alpha * (ROI - ROI_target)
    bid[t+1] = bid[t] - \alpha * (CPC - CPC_target)
        = bid[t] - \alpha * -(CPC - RPL/ROI_target) 

basically eq to online GD:
    eps = ROI - ROI_target
    argmin_{bid} eps**2 <- convex err fn
    deps^2/dROI = ROI - ROI_target
    ROI[t+1] = ROI[t] - \alpha*deps^2 = ROI[t] - \alpha * (ROI[t] - ROI_target)
    deps^2/dbid = deps^2/dROI * dROI/dbid = (ROI - ROI_target)(dRPC - ROI)/CPC
        ***
        assume dRPC = 1, ROI = RPL_0/CPC, ROI_target = RPL_0/CPC_target
        deps^2/dbid = -ROI/CPC * (RPL_0/CPC - RPL_0/CPC_target)
            = - RPL_0/CPC * RPL_0 (1/CPC - 1/CPC_target)
            \prop - 1/CPC^2 * (CPC_target - CPC)/(CPC * CPC_target)
            = - 1/CPC^3 * 1/CPC_target * (CPC_target - CPC)
            = CPC_target^-1 * CPC^-3 * (CPC - CPC_target)
        assume CPC relatively stable
            ===>
            \approx \prop CPC - CPC_target
        GD step
        bid[t+1] = bid[t] - \alpha deps^2 = bid[t] - \alpha (CPC - CPC_target)
        roughly what we have now assuming CPC relatively stable
        ***
    bid[t+1] = bid[t] - \alpha deps^2
        = bid[t] - \alpha * (ROI - ROI_target)(dRPC - ROI)/CPC

***NOTE: would be using some kindof linear regressor here - might want L2 regularization but dont 
        want to encourage sparsity w/ L1 or L0 regularization
can also approximate ROI[bid] as w@<x=[bid|1/bid|various 1-hot params]> + b
x[t+1] = x[t] - \alpha * w s.t. campaign constraints
"""
#%%
# hc creds
hccreds = {
    "client_id": "3cc90f549fd44a9fbe36f70aa6e7cfde",
    "client_secret": "eaa74a94108648579e65167bf276c3a1",
}
# pivot creds
pivotcreds = {
    "client_id": "31d7f068eb3d43acb85b5ad64eef2ca9",
    "client_secret": "c5010044df194010817e76edac0fd426",
}


client = TaboolaClient(**hccreds)
client.token_details


acct_service = AccountService(client)
accnts = acct_service.list()["results"]
NETWORK_ACCNT_ID = 1248460
O65_ACCNT_ID = 1150131
id2accnt = {a["id"]: a for a in accnts}

def accnt_camps(accnt):
    camp_service = CampaignService(client, accnt["account_id"])
    return camp_service.list()

camps = accnt_camps(id2accnt[O65_ACCNT_ID])
id2camp = {c["id"]: c for c in camps}
camp_ids = [*id2camp.keys()]

json.dump(id2camp, open("camps.json", "w"))

# with open("camps.json", "w") as f:
#     for _,camp in id2camp.items():
#         f.write(json.dumps(camp))
#         f.write("\n")


"""
Dimension 10 is not allowed for CampaignSummaryReport. Must be one of 
('day', 'week', 'month', 'content_provider_breakdown', 
'campaign_breakdown', 'site_breakdown', 'country_breakdown', 
'platform_breakdown', 'campaign_day_breakdown', 'campaign_site_day_breakdown')
"""
# for accnt in accnts:
#     d = CampaignSummaryReport(client, accnt["account_id"]) \
#         .fetch("site_breakdown", NOW-DAY, NOW)
#     print(accnt["name"], len(d["results"]))

# #%%
# # cmpaigns always on
# import jmespath
# get = jmespath.search
# # evercamps = get("[?activity_schedule.mode=='ALWAYS' && is_active].[activity_schedule.mode,is_active]", camps)
# evercamps = get(
#     "[?activity_schedule.mode=='ALWAYS' && is_active]", camps)
# c = evercamps[0]
# c
# [*yield_time_rules(camps[0])]
# # [*yield_time_rules(evercamps[0])]

cross = itertools.product
get = jmespath.search


def yield_from_type_pivot(camp, k):
    for t, v in cross(
            get(f"[{k}.type]", camp),
            get(f"{k}.value", camp) or [""]):
        yield [(k, t, v), 1]

def yield_time_rules(camp):
    k = "activity_schedule"
    time_mode = get(f"[{k}.mode]",camp)
    included_hrs = [
        [(rule, day, hr), int(rule == "INCLUDE")]
        for rule,day,st,end in
        get(f"{k}.rules[*].[type,day,from_hour,until_hour]", camp)
        for hr in range(st,end)
    ] or [([],1)]
    for m, [hr, v] in cross(time_mode, included_hrs):
        yield [(k, m, *hr),v]


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
            yield [(k, day, hr), v]


def yield_bid_strategy_rules(camp):
    k = "publisher_bid_strategy_modifiers"
    for r in get(f"{k}.values[*].*", camp):
        yield [(k, *r), 1]


def yield_bid_modifiers(camp):
    k = "publisher_bid_modifier"
    for site, mod in get(f"{k}.values[*].*", camp):
        yield [(k, site), mod]


flat_K = [
    "id",
    "cpc",
    # "safety_rating",
    # "daily_cap",
    # "daily_ad_delivery_model",
    # "bid_type",
    # "bid_strategy",
    # "traffic_allocation_mode",
    # "marketing_objective",
]
type_pivoted_K = [
    # "country_targeting",
    # "sub_country_targeting",
    # "dma_country_targeting",
    # "region_country_targeting",
    # "city_targeting",
    # "postal_code_targeting",
    # "contextual_targeting",
    # "platform_targeting",
    # "publisher_targeting",
    # "auto_publisher_targeting",
    # # "os_targeting",
    # "connection_type_targeting",
    # "browser_targeting",
]


def flatten_camp(camp):
    campd = {k: camp[k] for k in flat_K}
    for k in type_pivoted_K:
        campd.update(dict(yield_from_type_pivot(camp, k)))
    # campd.update(dict(yield_time_rules(camp)))
    # campd.update(dict(yield_bid_strategy_rules(camp)))
    campd.update(dict(yield_bid_modifiers(camp)))
    return campd


import pandas as pd
campdf = pd.DataFrame([flatten_camp(camp) for camp in camps])
bid_mod_C = [c for c in campdf.columns if "publisher_bid_modifier" == c[0]]
campdf[bid_mod_C] = campdf[bid_mod_C].fillna(1)
campdf = campdf.fillna(0)
print("|campdf|", campdf.shape)

print("campaign df sparsity:", (campdf == 0).sum().sum() / np.prod(campdf.shape))
strC = campdf.dtypes[campdf.dtypes == object].index
floatC = campdf.dtypes[campdf.dtypes == np.float64].index
#%%
# I = revdf["utm_term"].iloc[:,1] == revdf["utm_term"].iloc[:,2]
# J = revdf["utm_term"].iloc[:, 1].isna()
# (J | I).mean()
# #%%
# I = revdf["keyword"] == revdf["utm_term"].iloc[:, 2]
# J = revdf["keyword"].isna()
# K = J | I
# print(K.mean())
# revdf[["keyword", "utm_term"]][~K]
# #%%
#%%
_,publishers = zip(*bid_mod_C)
# s1 = {*publishers}
# s2 = {*revdf["keyword"]}
# # s2 = {*redir_url_df["utm_term"]}
# # s2 = {*source_url_df["utm_term"]}
# # s2 = {*landing_page_url_df["utm_term"]}
# len(s1),len(s2),len(s1 - s2),len(s2 - s1),len(s1 & s2)

_, publishers = zip(*bid_mod_C)
new_publishers = {*revdf["keyword"]} - {*publishers,None}
new_bid_mod_C = [("publisher_bid_modifier",p) for p in new_publishers]
print("|campdf|", campdf.shape)
campdf[new_bid_mod_C] = 1
print("|campdf|", campdf.shape)

print(len(bid_mod_C))
bid_mod_C = [*{*bid_mod_C + new_bid_mod_C}]
print(len(bid_mod_C))
#%%
def get_bid_mod(r):
    try:
        bidmods = campdf[("publisher_bid_modifier", r["keyword"])]
        return bidmods.loc[r["campaign_id"]]
    except KeyError:
        return 1.
# revdf["publisher_bid_mod"] = revdf.apply(
#     lambda r: campdf[("publisher_bid_modifier", r["keyword"])].loc[r["campaign_id"]], axis=1)
revdf["publisher_bid_mod"] = revdf.apply(get_bid_mod, axis=1)
if "cpc" not in revdf.columns:
    revdf = revdf.join(campdf.set_index("id")["cpc"],on="campaign_id")
revdf["effective_cpc"] = revdf["cpc"] * revdf["publisher_bid_mod"]
revdf["roi"] = revdf["revenue"] / revdf["effective_cpc"]

from statsmodels.stats.power import TTestIndPower
EFFECT_COL = "roi"
WEIGHT_COL = "cnt"
TARGET_ROI = 0.5
MAX_PUSH = 0.1
#%%
def to_nearest(d,inc=0.05):
    iinc = np.round(1/inc,0)
    return (d * iinc + 0.5) // 1 / iinc

c2roi = revdf[["campaign_id", "revenue", "cpc", "roi", "cnt"]] \
    .rename(columns={"revenue": "rpc"}) \
    .groupby("campaign_id") \
    .agg(["mean", "sum", "std"])
c2roi[("cpc","target")] = c2roi[("rpc","mean")] / TARGET_ROI
c2roi[("cpc","delta")] = c2roi[("cpc","target")] - c2roi[("cpc","mean")]
c2roi[("cpc","ub")] = c2roi[("cpc", "mean")] * (1+MAX_PUSH)
c2roi[("cpc","lb")] = c2roi[("cpc", "mean")] * (1-MAX_PUSH)
c2roi[("cpc","t+1")] = c2roi[("cpc","target")]W
c2roi[("cpc","t+1")] = c2roi[[("cpc","t+1"), ("cpc","ub")]].min(axis=1)
c2roi[("cpc","t+1")] = c2roi[[("cpc","t+1"), ("cpc","lb")]].max(axis=1)
c2roi[("cpc","t+1")] = to_nearest(c2roi[("cpc","t+1")],0.05)
c2roi[[("cpc","target"),("cpc","ub"),("cpc","lb"),("cpc","t+1"),]]
c2roi[("roi","lift")] = TARGET_ROI - c2roi[("roi","mean")]
c2roi[("roi","effect")] = (c2roi[("roi","lift")] - c2roi[("roi","mean")]) / c2roi[("roi","std")]

# TODO: think about ratio here
c2roi[(EFFECT_COL,"power")] =  c2roi.apply(lambda r: TTestIndPower().solve_power(
    r[EFFECT_COL]["effect"], power=None, nobs1=r[WEIGHT_COL]["sum"], ratio=1, alpha=0.05),axis=1)
I = c2roi[EFFECT_COL]["power"] > 0.9
c2roi[["cpc", EFFECT_COL]][I]
#%%
c2p2roi = revdf \
    .rename(columns={"revenue":"rpc"}) \
    [["campaign_id","keyword","rpc","cpc","roi","cnt"]] \
    .groupby(by=["campaign_id","keyword"]) \
    .agg(["mean","sum","std"])
c2p2roi[("cpc","target")] = c2p2roi[("rpc","mean")] / TARGET_ROI
c2p2roi[("cpc","delta")] = c2p2roi[("cpc","target")] - c2p2roi[("cpc","mean")]
c2p2roi[("cpc","ub")] = c2p2roi[("cpc", "mean")] * (1+MAX_PUSH)
c2p2roi[("cpc","lb")] = c2p2roi[("cpc", "mean")] * (1-MAX_PUSH)
c2p2roi[("cpc","t+1")] = c2p2roi[("cpc","target")]
c2p2roi[("cpc","t+1")] = c2p2roi[[("cpc","t+1"), ("cpc","ub")]].min(axis=1)
c2p2roi[("cpc","t+1")] = c2p2roi[[("cpc","t+1"), ("cpc","lb")]].max(axis=1)
c2p2roi[("cpc","t+1")] = to_nearest(c2p2roi[("cpc","t+1")],0.05)
c2p2roi[[("cpc","target"),("cpc","ub"),("cpc","lb"),("cpc","t+1"),]]
c2p2roi[("roi","lift")] = TARGET_ROI - c2p2roi[("roi","mean")]
c2p2roi[("roi","effect")] = (c2p2roi[("roi","lift")] - c2p2roi[("roi","mean")]) / c2p2roi[("roi","std")]

# TODO: think about ratio here
c2p2roi[(EFFECT_COL,"power")] =  c2p2roi.apply(lambda r: TTestIndPower().solve_power(
    r[EFFECT_COL]["effect"], power=None, nobs1=r[WEIGHT_COL]["sum"], ratio=1, alpha=0.05),axis=1)
I = c2p2roi[EFFECT_COL]["power"] > 0.9
c2p2roi[["cpc", EFFECT_COL]][I]
#%%
from pkg_resources import resource_filename as rscfn
import os
TABLE_DUMPS = rscfn(__name__,"TABLE_DUMPS")
os.makedirs(TABLE_DUMPS,exist_ok=True)
def fetch_head(rsc,limit=1000,src=HealthcareDW):
    with src() as db:
        db \
            .to_df(f"select * from {rsc} limit {limit}") \
            .to_csv(f"{TABLE_DUMPS}/{src.__name__}.{rsc}.csv")
# fetch_head("tracking.session_detail")
# fetch_head("tron.session_revenue")
fetch_head("tron.intraday_profitability")

#%%
import datetime
NOW = datetime.datetime.now()
DAY = datetime.timedelta(days=1)

start_date = NOW - 90*DAY
end_date = NOW - 0*DAY

start_date_ymd = start_date.strftime("%Y%m%d")
end_date_ymd = end_date.strftime("%Y%m%d")
product = None
traffic_source = None
start_date_ymd,end_date_ymd

TABOOLA = "TABOOLA"
O65     = 'MEDICARE'

df_tz_adj = hc_quarter_hour_tz_adjusted(
    start_date=start_date_ymd, end_date=end_date_ymd, product=O65, traffic_source=TABOOLA)
df_tz_adj["rps_raw"] = df_tz_adj["rps"]
#%%
import typing
import numpy as np

def spread_outliers(S,percentile=97.5) -> typing.Iterable:
    OUTTHRESH = np.percentile(S,percentile)
    OUTI = S > OUTTHRESH
    print("outlier thresh:", OUTTHRESH)
    T = OUTI * OUTTHRESH + (1-OUTI) * S
    T = (S.sum() / T.sum()) * T
    assert abs(T.sum() - S.sum()) < 1e-10
    return T
def cma(S,window) -> typing.Iterable:
    L = S.__len__()
    CMAker = [1/window] * window
    return np.convolve([*S,*S,*S], CMAker, mode="same")[L:-L]
def ema(S, window) -> typing.Iterable:
    """
    \sum_{r^i} = s = 1 + r + r^2 + ....
    s*r = r + r^2 + r^3 + ... = s-1
    s * r = s - 1 ===> s = 1 / (1-r)
    s - 1 = 1 / (1-r) - 1 = r / (1-r)
    r \approx (window-1)/window

    ema(X,t) = (1-r)*X[t] + r*ema(X,t-1)
    """
    L = S.__len__()
    r = (window-1)/window
    EMAker = (1-r) * 1/(1-r**window) * np.array([r**i for i in range(window)])
    assert abs(EMAker.sum() - 1) < 1e-10
    return np.convolve([*S,*S,*S], EMAker, mode="same")[L:-L]


rps = df_tz_adj["rps_raw"]
rps = spread_outliers(rps)

window = 16
df_tz_adj["rps_cma"] = cma(rps,window)
df_tz_adj["rps_ema"] = ema(rps,window)
df_tz_adj["rps_cema"] = cma(ema(rps, window), window)
df_tz_adj["sessions_cema"] = cma(ema(df_tz_adj["sessions"], window), window)

df_tz_adj[["rps_cma", "rps_ema", "rps_cema"]].plot(figsize=(15, 5))
# s1,s2 = df_tz_adj[["rps_raw","rps"]].sum()
# assert abs(s1 - s2) < 1e-10
#%%
import pandas as pd
rpsdf = pd.DataFrame.multiply(
    df_tz_adj[["rps", "rps_raw", "rps_cma","rps_ema"]], df_tz_adj["sessions"],
    axis=0)
rpsdf.sum()
#%%
rev_cema_sum = (df_tz_adj["rps_cema"] * df_tz_adj["sessions_cema"]).sum()
rev_sum = (df_tz_adj["rps"] * df_tz_adj["sessions"]).sum()
rev_cema_sum,rev_sum
# %%
RPS_SPLINE_K = 3
RPS_SPLINE_S = 45

df_tz_adj["rps"] = spread_outliers(df_tz_adj["rps_raw"],97.5)
df_tz_adj = add_spline(df_tz_adj, index_col='int_ix',
                       smooth_col='rps', spline_k=RPS_SPLINE_K, spline_s=RPS_SPLINE_S)

ax = df_tz_adj[['rps']].reset_index().plot.scatter(x='int_ix', y='rps')
df_tz_adj[['rps_spline',"rps_cema"]].plot(ax=ax, figsize=(15, 5), colormap='Dark2')
# %%
SESSIONS_SPLINE_K = 3
SESSIONS_SPLINE_S = 10 * 1000

df_tz_adj = add_spline(df_tz_adj, index_col='int_ix', smooth_col='sessions',
                       spline_k=SESSIONS_SPLINE_K, spline_s=SESSIONS_SPLINE_S)

ax = df_tz_adj[['sessions']].reset_index().plot.scatter(x='int_ix', y='sessions')
df_tz_adj[['sessions_spline',"sessions_cema"]].plot(ax=ax, figsize=(15, 5), colormap='spring')
# %%
rps_mean = (df_tz_adj['sessions'] * df_tz_adj['rps']).sum() / df_tz_adj['sessions'].sum()
df_tz_adj["rps_mean_adjusted"] = df_tz_adj["rps"] / rps_mean
df_tz_adj['baseline'] = 1
rps_spline_mean = (df_tz_adj['sessions_spline'] * df_tz_adj['rps_spline']).sum() \
                                / df_tz_adj['sessions_spline'].sum()
df_tz_adj["rps_spline_mean"] = rps_spline_mean
df_tz_adj['spline_modifier'] = df_tz_adj['rps_spline'] / rps_spline_mean
df_tz_adj['spline_modifier'] = df_tz_adj['spline_modifier'] * 20 // 1 / 20 # set to incs of 0.05
rps_cema_mean = (df_tz_adj['sessions_spline'] * df_tz_adj['rps_spline']).sum() \
    / df_tz_adj['sessions_spline'].sum()
df_tz_adj["rps_cema_mean"] = rps_cema_mean
df_tz_adj['cema_modifier'] = df_tz_adj['rps_cema'] / rps_cema_mean
df_tz_adj['cema_modifier'] = df_tz_adj['cema_modifier'] * 20 // 1 / 20  # set to incs of 0.05
ax = df_tz_adj.reset_index().plot.scatter(x='int_ix', y='rps_mean_adjusted')
df_tz_adj[["baseline", "spline_modifier","cema_modifier"]].plot(ax=ax,figsize=(15, 5))
#%%
df_tz_adj.shape
#%%
1/0
df_tz_adj_global = hc_quarter_hour_tz_adjusted(
    start_date=start_date_ymd, end_date=end_date_ymd, product=O65)
df_tz_adj_global["rps_raw"] = df_tz_adj_global["rps"]
#%%
rps = df_tz_adj_global["rps_raw"]
rps = spread_outliers(rps)

window = 16
df_tz_adj_global["rps_cma"] = cma(rps, window)
df_tz_adj_global["rps_ema"] = ema(rps, window)
df_tz_adj_global["rps_cema"] = cma(ema(rps, window), window)
df_tz_adj_global[["rps_cma", "rps_ema", "rps_cema"]].plot(figsize=(15, 5))
#%%
import pandas as pd
rpsdf = pd.DataFrame.multiply(
    df_tz_adj_global[["rps", "rps_raw", "rps_cma",
                      "rps_ema"]], df_tz_adj_global["sessions"],
    axis=0)
rpsdf.sum()
