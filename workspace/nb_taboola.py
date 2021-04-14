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


from pytaboola import TaboolaClient
client = TaboolaClient(**hccreds)
client.token_details

from pytaboola.services import AccountService
from pytaboola.services import CampaignService

import itertools
import tqdm
import pandas as pd
import os
from pkg_resources import resource_filename as rscfn


acct_service = AccountService(client)
accnts = acct_service.list()["results"]
NETWORK_ACCNT_ID = 1248460
id2accnt = {a["id"]: a for a in accnts}

def accnt_camps(accnt):
    camp_service = CampaignService(client, accnt["account_id"])
    return camp_service.list()

id2camp = {c["id"]: c for a in tqdm.tqdm(accnts) for c in accnt_camps(a)}
camp_ids = [*id2camp.keys()]
camps = [*id2camp.values()]

import json
json.dump(id2camp, open("camps.json", "w"))

# with open("camps.json", "w") as f:
#     for _,camp in id2camp.items():
#         f.write(json.dumps(camp))
#         f.write("\n")

from pytaboola.services.report import \
    CampaignSummaryReport, RecirculationSummaryReport, \
    TopCampaignContentReport, RevenueSummaryReport, VisitValueReport
import datetime

DAY = datetime.timedelta(days=1)
MONTH = 30 * DAY
YEAR = 365 * DAY
NOW = datetime.datetime.now()

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

import itertools
cross = itertools.product
import jmespath
get = jmespath.search

def yield_from_type_pivot(camp,k):
    for t,v in cross(
            get(f"[{k}.type]", camp),
            get(f"{k}.value", camp) or [""]):
        yield [(k,t,v),1]

# def yield_time_rules(camp):
#     k = "activity_schedule"
#     time_mode = get(f"[{k}.mode]",camp)
#     included_hrs = [
#         [(rule, day, hr), int(rule == "INCLUDE")]
#         for rule,day,st,end in 
#         get(f"{k}.rules[*].[type,day,from_hour,until_hour]", camp)
#         for hr in range(st,end)
#     ] or [([],1)]
#     for m, [hr, v] in cross(time_mode, included_hrs):
#         yield [(k, m, *hr),v]


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
    for r in get(f"{k}.values[*].*",camp):
        yield [(k,*r),1]


def yield_bid_modifiers(camp):
    k = "publisher_bid_modifier"
    for site,mod in get(f"{k}.values[*].*", camp):
        yield [(k, site), mod]

flat_K = [
    "id",
    "safety_rating",
    "daily_cap",
    "daily_ad_delivery_model",
    "bid_type",
    "bid_strategy",
    "traffic_allocation_mode",
    "marketing_objective",
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
    campd = {k: camp[k] for k in flat_K}
    for k in type_pivoted_K:
        campd.update(dict(yield_from_type_pivot(camp,k)))
    campd.update(dict(yield_time_rules(camp)))
    campd.update(dict(yield_bid_strategy_rules(camp)))
    campd.update(dict(yield_bid_modifiers(camp)))
    return campd

import pandas as pd
campdf = pd.DataFrame([flatten_camp(camp) for camp in camps])
bid_mod_C = [c for c in campdf.columns if "publisher_bid_modifier" == c[0]]
campdf[bid_mod_C].fillna(1)
campdf = campdf.fillna(0)
print("|campdf|", campdf.shape)

import numpy as np
print("campaign df sparsity:",(campdf == 0).sum().sum() / np.prod(campdf.shape))
strC = campdf.dtypes[campdf.dtypes == object].index
floatC = campdf.dtypes[campdf.dtypes == np.float64].index

import typing
from ds_utils.db.connectors import \
    HealthcareDW, PivotDW, AnalyticsDB, \
    MySqlContext, RedshiftContext


"""
- user monetizes 3 months ago
- but they have a session in the past 14 days
- TODO: 
    - treat 1 session == 1 click
        - dont agg by user
        - dont worry about click id
        - sessoin in UTC, revenue in EST
    - focus on RPC not ROI(CPC)
    - compare macro spline to taboola apline
    - segment by product type: session detail & tron revenue
"""
#%%
DAYS_FROM = 30
DAYS_TILL = 0
with HealthcareDW() as db:
    df = db.to_df(
        f"""
        select 
            *
        from
            tracking.session_detail as s
            JOIN
            tron.session_revenue r
            ON s.session_id = r.session_id
        where
            s.creation_date >= CURRENT_DATE - '{DAYS_FROM} days'::interval AND 
            s.creation_date < CURRENT_DATE - '{DAYS_TILL} days'::interval AND 
            UPPER(s.traffic_source) = 'TABOOLA'
        limit 100
        """
    )
#%%
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
source_url_df = parse_url_series(df["source"])
redir_url_df = parse_url_series(source_url_df["redir"])
landing_page_url_df = parse_url_series(df["landing_page"].iloc[:,0])
#%%

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
def get_schema(rsc,src=HealthcareDW):
    with src() as db:
        df = db \
            .to_df(f"select * from {rsc} where false")
        return df.columns
sC = get_schema("tracking.session_detail")
rC = get_schema("tron.session_revenue")
sC - rC
#%%
s1 = {*sC}
s2 = {*rC}
len(s1 - s2),len(s2-s1),len(s1 & s2)
#%%
with HealthcareDW() as db:
    df = db.to_df(
        f"""
        with user_revenue as (
            select 
                s.campaign_id,
                s.user_id,
                SUM(COALESCE(r.revenue,0))  as user_revenue,
                MAX(s.creation_date)        as date
            from
                tracking.session_detail as s
                RIGHT JOIN
                tron.session_revenue r
                ON s.session_id = r.session_id
            where
                s.creation_date >= CURRENT_DATE - '{DAYS_FROM} days'::interval AND 
                s.creation_date < CURRENT_DATE - '{DAYS_TILL} days'::interval AND 
                UPPER(s.traffic_source) = 'TABOOLA'
            group by 
                s.user_id,s.campaign_id 
        )
        select 
            r.campaign_id,
            AVG(COALESCE(user_revenue,0))       as avg_click_revenue,
            SUM(COALESCE(user_revenue,0))       as sum_click_revenue,
            STDDEV(COALESCE(user_revenue,0))    as std_click_revenue,
            COUNT(*)                            as cnt_click
        from
            user_revenue r
        group by
            r.campaign_id
        """
    )
campaign_click_revenue_df = df

d = CampaignSummaryReport(client, id2accnt[NETWORK_ACCNT_ID]["account_id"]) \
    .fetch("campaign_breakdown", NOW-14*DAY, NOW)
camp_perf_df = pd.DataFrame(d["results"])

campdf = campdf \
    .set_index("id") \
    .join(campaign_click_revenue_df.set_index("campaign_id"),how="inner") \
    .reset_index() \
    .rename(columns={"index": "id"})
print("|campdf|",campdf.shape)
campdf = campdf \
    .set_index("id") \
    .join(camp_perf_df.set_index("campaign"),how="inner",rsuffix="_perf") \
    .reset_index() \
    .rename(columns={"index": "id"})
print("|campdf|", campdf.shape)

campdf["roi"] = campdf["avg_click_revenue"] / campdf["cpc"]
campdf["profit"] = campdf["sum_click_revenue"] - campdf["cpc"] * campdf["cnt_click"]
campdf[["roi","profit"]]
print("|campdf|", campdf.shape)

EFFECT_COLS = ["profit", "roi", "cnt_click","avg_click_revenue"]
EFFECT_COL = "profit"
WEIGHT_COL = "cnt_click"
#%%
k2C = {}
for c in campdf.columns:
    if isinstance(c, tuple):
        k2C.setdefault(c[0],[]).append(c)
k2C.keys()
k2C["activity_schedule"]
campdf[k2C["activity_schedule"]]
#%%
"""
TODO:
- look into no TOD targetting in past 2 weeks
"""

C = k2C["activity_schedule"]
# C = k2C["connection_type_targeting"]
# C = k2C["browser_targeting"]
# C = k2C["platform_targeting"]
# effect_col = "roi"
# effect_col = "profit"
effect_col = "avg_click_revenue"
weight_col = WEIGHT_COL

from statsmodels.stats.power import TTestIndPower
def nanavg(V,weights,**kwargs):
    try:
        nidx = np.isnan(V) | np.isnan(weights) | \
            ~np.isfinite(V) | ~np.isfinite(weights)
        return np.average(V[~nidx],weights=weights[~nidx]+1,**kwargs)
    except ZeroDivisionError:
        return np.NaN
def coalesce(v,w):
    return w if not v or np.isnan(v) else v 

nidx = campdf[effect_col].isna()
D = []
for c in tqdm.tqdm(C):
    E = campdf[effect_col][~nidx]
    W = campdf[weight_col][~nidx]
    v1 = 1
    v1idx = campdf[c] == 1
    E1 = E[v1idx]
    E2 = E[~v1idx]
    W1 = W[v1idx]
    W2 = W[~v1idx]
    mu = nanavg(E, weights=W)
    mu1 = coalesce(nanavg(E1,weights=W1),mu)
    mu2 = coalesce(nanavg(E2,weights=W2),mu)
    stddev = nanavg((campdf[effect_col]-mu)**2,
                    weights=campdf[weight_col]) ** 0.5
    lift = mu1 - mu2
    effect = abs(lift) / stddev
    cnt1 = W1.sum()
    cnt2 = W2.sum()
    analysis = TTestIndPower()
    power = analysis.solve_power(
        effect, power=None, nobs1=cnt1, ratio=cnt2/cnt1, alpha=0.05)
    D.append(dict(
        effect_col="effect_col",
        c=c,
        v1=v1,
        mu1=mu1,
        mu2=mu2,
        cnt1=cnt1,
        cnt2=cnt2,
        lift=lift,
        mu=mu,
        stddev=stddev,
        effect=effect,
        power=power,
    ))
powerdf = pd.DataFrame(D)
powerdf.head(20)
#%%
powerdf[powerdf["power"] > 0.9]
#%%
not np.NaN
#%%
campdf[k2C["activity_schedule"]]
#%%


def nanavg(V, weights, **kwargs):
    try:
        nidx = np.isnan(V) | np.isnan(weights)
        return (V[~nidx] * weights[~nidx]).sum() / weights[~nidx].sum()
    except ZeroDivisionError:
        return np.NaN

nanavg(campdf[effect_col],campdf[weight_col])
#%%
campdf[effect_col] * campdf[weight_col]
#%%
(campdf[effect_col] * campdf[weight_col]).sum()
#%%
np.average()
#%%
nidx = powerdf["power"].isna()
powerdf[~nidx]
#%%
import tqdm
import collections
import scipy
import scipy.sparse
import typing

from IPython.display import display as ipydisp
from IPython.display import display, Markdown, Latex, FileLink, FileLinks

import logging
LOGLEVEL = logging.DEBUG
# LOGFMT = "%(levelname)s %(asctime)s %(filename)s %(lineno)s: %(message)s"
LOGFMT = "%(filename)s %(lineno)s: %(message)s"
logger = logging.getLogger(__name__)
logger.handlers = []
logger.handlers.append(logging.StreamHandler(sys.stdout))
logger.propagate = False
logger.setLevel(logging.DEBUG)
for h in logger.handlers:
    h.setFormatter(logging.Formatter(LOGFMT))

DEFAULT_VAL = "EMPTY"
MINCNT = 30

encC = []
for c in strC:
    v2c = campdf[[c,"cnt_click"]].groupby(c).sum()
    encC += [(c,v) for v in v2c.index]
    for i,v in enumerate(v2c.index):
        campdf[(c,v)] = (campdf[c] == v).astype(int)
campdf[encC]
#%%
from statsmodels.stats.power import TTestIndPower
def nanavg(V,weights,**kwargs):
    nidx = np.isnan(V)
    return np.average(V[~nidx],weights=weights[~nidx]+1,**kwargs)

POWER_DIR = rscfn(__name__, "POWER_TEST")
os.makedirs(POWER_DIR, exist_ok=True)
def power_test(campdf,effect_col,weight_col):
    D = []
    for c in tqdm.tqdm([*encC,*floatC]):
        vset = {*campdf[c]}
        if len(vset) <= 1: continue
        # # # get "oddest 1 out" value to run power test on
        # # v1,_ = max((abs(v-np.mean(vset)),v) for v in vset)
        # # pick most common value to run power test on
        # _,v1 = max(((campdf[c] == v).sum(),v) for v in vset)
        # run power test on every val:
        for v1 in vset:
            v1idx = campdf[c] == v1
            mu1 = nanavg(campdf[v1idx][effect_col],weights=campdf[v1idx][weight_col])
            mu2 = nanavg(campdf[~v1idx][effect_col],weights=campdf[~v1idx][weight_col])
            mu = nanavg(campdf[effect_col],weights=campdf[weight_col])
            stddev = nanavg((campdf[effect_col]-mu)**2, weights=campdf[weight_col]) ** 0.5
            lift = mu1 - mu2
            effect = abs(lift) / stddev
            cnt1 = campdf[v1idx][weight_col].sum()
            cnt2 = campdf[~v1idx][weight_col].sum()
            analysis = TTestIndPower()
            power = analysis.solve_power(effect, power=None, nobs1=cnt1, ratio=cnt2/cnt1, alpha=0.05)
            D.append(dict(
                c=c,
                v1=v1,
                mu1=mu1,
                mu2=mu2,
                lift=lift,
                stddev=stddev,
                effect=effect,
                power=power,
            ))
    powerdf = pd.DataFrame(D).sort_values(by="power",ascending=False)
    powerdf.to_csv(f"{POWER_DIR}/{effect_col}_powerdf.csv")
    return powerdf

for effect_col in EFFECT_COLS:
    power_test(campdf,effect_col,WEIGHT_COL)
#%%
import sklearn.linear_model
import sklearn.neural_network

def train_eval_reg(Xdf,regcls,target_reg_col,weight_col):
    nidx = Xdf[target_reg_col].isna()
    
    logger.warning(
        f"Training {regcls.__module__}.{regcls.__name__} against `{target_reg_col}`")
    mlC = [*encC, *floatC]
    X = Xdf[mlC][~nidx]
    y = Xdf[target_reg_col][~nidx]
    weights = Xdf[weight_col][~nidx]
    Xmu = np.average(X,axis=0,weights=weights)
    Xvar = np.average((X - Xmu)**2,axis=0,weights=weights) 
    # X = (X - Xmu) / (Xvar ** 0.5 + 1e-10)

    reg: sklearn.linear_model.Lasso = regcls()
    reg = reg.fit(X,y,sample_weight=weights)
    yhat = reg.predict(X)
    MSE = ((yhat - y)**2).mean()
    MAE = ((yhat - y).abs()).mean()
    MRE = ((yhat - y).abs() / (y.abs() + 1e-10)).mean()
    ipydisp(pd.DataFrame(
        data=[
            [MSE, MAE, MRE],
            # [MSEt, MAEt, MREt],
        ],
        columns=["MSE", "MAE", "MRE"],
        # index=["Train", "Test"]
        index=["Train", ]
    ))
    # logger.info(f"train     MSE: {MSEf}     MAE: {MAEf}     MRE: {MREf}")
    # logger.info(f"test      MSE: {MSEt}     MAE: {MAEt}     MRE: {MREt}")

    Xdf["reg_hat"] = np.NaN
    Xdf["reg_hat"][~nidx] = yhat
    reg.HCmlC = mlC
    return reg

# regcls = sklearn.linear_model.Lasso
for regcls in (
    sklearn.linear_model.LinearRegression,
    sklearn.linear_model.Ridge,
    sklearn.linear_model.Lasso,
    sklearn.linear_model.ElasticNet,
    sklearn.linear_model.SGDRegressor,
    # slow
    # sklearn.tree.DecisionTreeRegressor,
    # doenst work w/ sparse
    # sklearn.gaussian_process.GaussianProcessRegressor,
    # slow
    # sklearn.ensemble.RandomForestRegressor,
    # sklearn.neural_network.MLPRegressor, # doesnt aaccept weights in fit
    # redundant - default MLP is already activated by relu
    # lambda: sklearn.neural_network.MLPRegressor(activation="relu"),
):
    continue
    reg = train_eval_reg(campdf, regcls, "avg_click_revenue", WEIGHT_COL)

REG_DIR = rscfn(__name__,"REG_ANA")
os.makedirs(REG_DIR,exist_ok=True)

def reg_ana(campdf,effect_col,weight_col,regcls):
    reg: sklearn.linear_model.Ridge = \
        train_eval_reg(campdf, regcls, effect_col, weight_col)
    featdf = pd.DataFrame(zip(reg.HCmlC,reg.coef_),columns=['feat','imp'])
    featdf["abs_imp"] = featdf["imp"].abs()
    featdf.sort_values(by="abs_imp",ascending=False)\
        .to_csv(f"{REG_DIR}/{effect_col}_reg_feat_imp.csv")
    return featdf

for effect_col in EFFECT_COLS:
    reg_ana(campdf, effect_col, WEIGHT_COL, sklearn.linear_model.Ridge)
#%%
from pkg_resources import resource_filename as rscfn
import os
TABLE_DUMPS = rscfn(__name__,"TABLE_DUMPS")
os.makedirs(TABLE_DUMPS,exist_ok=True)
def fetch_head(rsc,limit=1000,src=PivotDW):
    with src() as db:
        db \
            .to_df(f"select * from {rsc} limit {limit}") \
            .to_csv(f"{TABLE_DUMPS}/{src.__name__}.{rsc}.csv")


fetch_head("tracking.session_detail", src=HealthcareDW)
fetch_head("tron.session_revenue", src=HealthcareDW)
#%%
