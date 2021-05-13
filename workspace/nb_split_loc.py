#%%
from pkg_resources import resource_filename as rscfn

import jmespath
import json
import datetime
import typing
import os
import tqdm
import itertools
import pandas as pd
import numpy as np

from pytaboola import TaboolaClient
from pytaboola.services import AccountService
from pytaboola.services import CampaignService
from pytaboola.services.report import \
    CampaignSummaryReport, RecirculationSummaryReport, \
    TopCampaignContentReport, RevenueSummaryReport, VisitValueReport

from ds_utils.db.connectors import HealthcareDW

from pkg_resources import resource_filename as rscfn
import os
TABLE_DUMPS = rscfn(__name__, "TABLE_DUMPS")
os.makedirs(TABLE_DUMPS, exist_ok=True)


def fetch_head(rsc, limit=1000, src=HealthcareDW):
    with src() as db:
        db \
            .to_df(f"select * from {rsc} limit {limit}") \
            .to_csv(f"{TABLE_DUMPS}/{src.__name__}.{rsc}.csv")


fetch_head("data_science.maxmind_ipv4_geo_blocks")
fetch_head("data_science.maxmind_geo_locations")

with HealthcareDW() as db:
    df = db.to_df(
        """
        select distinct 
            metro_code,subdivision_1_iso_code 
        from data_science.maxmind_geo_locations
        where country_iso_code = 'US' AND 
                metro_code IS NOT NULL AND 
                metro_code != ''
        """)
df
#%%
NOW = datetime.datetime.now().date()
DAY = datetime.timedelta(days=1)

start_date = NOW - 30*DAY
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
                l.*,
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
                s.creation_date                                         AS utc_ts,
                extract(
                    HOUR FROM
                    convert_timezone('UTC', l.time_zone, s.creation_date) 
                        - s.creation_date
                )::INT                                                  AS utc_offset,
                l.time_zone,
                convert_timezone('UTC', l.time_zone, s.creation_date)   AS user_ts,
                l.subdivision_1_iso_code                                AS state,
                l.metro_code                                            AS dma,
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
                AND nullif(dma,'') IS NOT NULL 
                AND s.creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
            {product_filter}
            {traffic_filter}
        )
    SELECT
        state,
        dma,
        user_ts::date                           as date,
        COUNT(*)                                as clicks_num,
        SUM((revenue>0)::int)                   as leads_num,
        SUM(revenue)                            as rev_sum,
        AVG(revenue)                            as rev_avg,
        STDDEV(revenue)                         as rev_std
    FROM rps_tz_adj
    GROUP BY 
        state,dma,date
"""
"""
ctr = clicks/impressions
lead/click = (rev > 0) / clicks
rev/lead = rev / (rev > 0)
rev/click = rev / clicks
"""
#%%
with HealthcareDW() as db:
    df = db.to_df(timeofday_query)

df["lpc"] = df["leads_num"] / df["clicks_num"]
df["rpl"] = df["rev_sum"] / df["leads_num"]
df["rpc"] = df["rev_avg"]
df_bkp = df
df = df.set_index(["state","dma","date"]) \
    .sort_index()

WEIGHT_COL = "clicks_num"
kpis = ["lpc", "rpl", "rpc"]
wkpis = [f"weighted_{kpi}" for kpi in kpis]
df[wkpis] = df[kpis] * df[WEIGHT_COL].values.reshape(-1, 1)

date_range = pd.date_range(start_date,end_date)
# %%
state_dmas = df \
    .sum(level=[0, 1]) \
    .sort_values(by=WEIGHT_COL, ascending=False) \
    .index
state_dma_locs = [([state],[dma],slice(None)) for state,dma in state_dmas]
states = df \
    .sum(level=[0,]) \
    .sort_values(by=WEIGHT_COL, ascending=False) \
    .index
state_locs = [([l],slice(None),slice(None)) for l in states]
dmas = df \
    .sum(level=[1,]) \
    .sort_values(by=WEIGHT_COL, ascending=False) \
    .index
dma_locs = [(slice(None),[l],slice(None)) for l in dmas]

for locs in state_dma_locs,state_locs,dma_locs:
    for l in locs:
        print(l,df.loc[l].shape)
#%%
def identity(X): return X
import scipy.ndimage
def gaussian(X,sigma):
    Y = X.copy()
    Y.loc[:,:] = scipy.ndimage.gaussian_filter1d(X, sigma=sigma, axis=0)
    return Y

def dfmult(df1,df2):
    intersection = {*df1.columns} & {*df2.index}
    return df1.loc[:, intersection].fillna(0) @ df2.loc[intersection, :].fillna(0)
def matnorm(X):
    return (X - X.mean())/X.std()
def kpicorr(l1,l2, filter=identity):
    df1 = filter(df.loc[l1].sum(level="date").reindex(date_range, fill_value=0))
    df2 = filter(df.loc[l2].sum(level="date").reindex(date_range, fill_value=0))
    X1 = df1[kpis]
    X2 = df2[kpis]
    kpicorr = dfmult(matnorm(X1).T, matnorm(X2))
    return np.diag(kpicorr).sum()
def weighted_kpicorr(l1,l2, filter=identity):
    df1 = filter(df.loc[l1].sum(level="date").reindex(date_range, fill_value=0))
    df2 = filter(df.loc[l2].sum(level="date").reindex(date_range, fill_value=0))
    W1 = df1[WEIGHT_COL].values.reshape(-1, 1)
    W2 = df2[WEIGHT_COL].values.reshape(-1, 1)
    X1 = df1[kpis]
    X2 = df2[kpis]
    kpicorr = dfmult(matnorm(X1).T, matnorm(X2))
    return W1.sum() * W2.sum() * np.diag(kpicorr).sum()
def wkpicorr(l1, l2, filter=identity):
    df1 = filter(df.loc[l1].sum(level="date").reindex(date_range, fill_value=0))
    df2 = filter(df.loc[l2].sum(level="date").reindex(date_range, fill_value=0))
    W1 = df1[WEIGHT_COL].values.reshape(-1, 1)
    W2 = df2[WEIGHT_COL].values.reshape(-1, 1)
    X1 = df1[kpis] * W1
    X2 = df2[kpis] * W2
    kpicorr = dfmult(matnorm(X1).T, matnorm(X2))
    return np.diag(kpicorr).sum()
def weighted_wkpicorr(l1, l2, filter=identity):
    df1 = filter(df.loc[l1].sum(level="date").reindex(date_range, fill_value=0))
    df2 = filter(df.loc[l2].sum(level="date").reindex(date_range, fill_value=0))
    W1 = df1[WEIGHT_COL].values.reshape(-1, 1)
    W2 = df2[WEIGHT_COL].values.reshape(-1, 1)
    X1 = df1[kpis] * W1
    X2 = df2[kpis] * W2
    kpicorr = dfmult(matnorm(X1).T, matnorm(X2))
    return W1.sum() * W2.sum() * np.diag(kpicorr).sum()
def sim_sort_loc_split(locs,K,simf):
    lh,*lT = locs
    gps = [[lh],*[[] for _ in range(1,K)]]
    locsims = [simf(lh,lt) for lt in lT]
    for i,(_,l) in enumerate(sorted(zip(locsims,lT),reverse=True)):
        gps[(i+1)%K].append(l)
    return gps

from matplotlib import pyplot as plt
def gpsum(df, gp):
    return pd \
        .concat((df.loc[l] for l in gp)) \
        .sum(level=["date"])

def gpcorr(df,gpi,gpj):
    X = gpsum(df,gpi)[wkpis]
    Y = gpsum(df,gpj)[wkpis]
    corrM = dfmult(matnorm(X).T,matnorm(Y)) / max(len(X),len(Y))
    return pd.Series(np.diag(corrM),index=wkpis)

def score_split(df,gps):
    pair_scores = []
    for i in range(len(gps)):
        for j in range(i+1,len(gps)):
            pair_scores.append(gpcorr(df,gps[i],gps[j]).mean())
    return np.array(pair_scores).mean()

def plt_split_kpis(df, gps):
    for c in wkpis:
        ax = plt.gca()
        for i, gp in enumerate(gps):
            # gpsum(df, gp)[c].plot(figsize=(20, 10), label=f"gp {i}")
            matnorm(gpsum(df, gp))[c].plot(figsize=(20, 10), label=f"gp {i}")
        plt.title(f"Daily normalized {c} by group")
        plt.legend()
        plt.show()

def gaussian1(X):
    return gaussian(X,sigma=1)
def f(*args, **kwargs):
    return weighted_wkpicorr(*args, **kwargs, filter=gaussian1)
score_split(df, sim_sort_loc_split(states, 2, f))
#%%
gps = sim_sort_loc_split(state_locs, 2, f)
gps
#%%
df1,df2 = gpsum(df,gps[0]),gpsum(df,gps[1])
df1.shape
#%%
def matnorm_fit(X):
    H,W = X.shape
    H = int(H/2)
    return (X-X.iloc[:H].mean()) / X.iloc[:H].std()
import scipy.stats
ind_test = scipy.stats.ttest_ind(matnorm_fit(df1),matnorm_fit(df2))
pd.DataFrame(ind_test,index=["t","p"],columns=df.columns)
#%%
plt_split_kpis(df,gps)
#%%
"""
AA test
- null hypo true
- p-val = prob(observe result >= observed result | no effect)
    => AA test - so want small observed result
    => want p-val to be large > 0.9
- power = prob(test rejects null hypo correctly | null hypothesis false)
- want negative power: prob(test rejects null)
    = prob(test configms null hypo | null hypothesis true)
"""
#%%
locs = pd.Series(state_dma_locs)
K = 4

locdf = df.groupby(["state", "dma", "date"]) \
    .sum() \
    .unstack(level="date")
kpi_tensor = np.stack((
    locdf[kpi] .reindex(date_range, axis=1) .fillna(0)
    for kpi in kpis
))
D, H, W = kpi_tensor.shape
mu = kpi_tensor.mean(axis=2).reshape(D, H, 1)
std = kpi_tensor.std(axis=2).reshape(D, H, 1)
kpi_tensor_norm = (kpi_tensor - mu) / std
kpi_tensor_norm[np.isnan(kpi_tensor_norm)] = 0
kpi_corr = (kpi_tensor_norm @ kpi_tensor_norm.transpose(0, 2, 1)) / W
loc_corr_df = pd.DataFrame(kpi_corr.mean(axis=0), columns=locs, index=locs)
# assert np.abs(np.diag(loc_corr_df) - 1).max() < 1e-10

import sklearn.cluster
clust = sklearn.cluster \
    .AgglomerativeClustering(affinity="precomputed",n_clusters=K,linkage='complete') \
    .fit_predict((1-loc_corr_df.values) / 2)

gps = [[] for _ in range(K)]
clusts = [locs[clust == i] for i in range(clust.max()+1)]
for i,loc in enumerate(np.concatenate(clusts)):
    gps[i%K].append(loc)

score_split(df,gps)
#%%
locs
#%%
np.abs(np.diag(loc_corr_df) - 1).max()
#%%

np.diag(loc_corr_df)
#%%
import scipy.cluster.hierarchy
scipy.cluster.hierarchy.fcluster()

#%%
T = np.ones((5,4,3))
M = np.ones((5 ,3, 4))
M[0,:,:] = 0
S = T @ M
S.shape
S
#%%
df.shape
#%%

#%%
D = []
# simple check validating similarity metrics & group indexing
for simf in [kpicorr, weighted_kpicorr, wkpicorr, weighted_wkpicorr]:
    s1 = score_split(df, sim_sort_loc_split(states, 4, simf))
    s2 = score_split(df, sim_sort_loc_split(state_locs, 4, simf))
    assert s1 == s2, (simf.__name__, f"{s1} != {s2}")

for k in tqdm.tqdm(range(2, 7)):
    for simf in [kpicorr,weighted_kpicorr,wkpicorr,weighted_wkpicorr]:
        for locs_nm, locs in {
                    "states": state_locs,
                    "dmas": dma_locs,
                    "state_dmas": state_dma_locs,
                }.items():
            score = score_split(df, sim_sort_loc_split(locs, k, simf))
            D.append(dict(
                k=k,
                locs_nm=locs_nm,
                simf=simf.__name__,
                score=score,
            ))
#%%
scoredf = pd.DataFrame(D) \
    .set_index(["locs_nm","simf","k"])
scoredf
#%%
scoredf
#%%
ax = plt.gca()
labels = scoredf.index.unique(level="locs_nm")
for locs_nm in labels:
    scoredf \
        .loc[locs_nm] \
        .max(level="k") \
        .plot(ax=ax,label=locs_nm)
plt.legend(labels)
plt.show()
#%%
ax = plt.gca()
for simf in scoredf.index.unique(level="simf"):
    scoredf \
        .loc[(slice(None),simf),] \
        .max(level="k") \
        .plot(ax=ax, label=simf)
plt.legend(scoredf.index.unique(level="simf"))
plt.show()

#%%
simf = kpicorr
K = 2
gps = [[] for _ in range(K)]
locsims = [simf(states[0], l) for l in states]
locsims_ = [simf(state_locs[0], l) for l in state_locs]
[*zip(states,locsims,locsims_)]
#%%
"""
- [ ] run A-A sig test
    - 0.05, .9 power , 
- [x] compare STATExDMA w/ DMA and STATE splits
- [x] exclude low volume DMAs
- [ ] exclude outliers - try applying filters
"""
#%%
locs = locs[:10]
loc_sim_M = pd.DataFrame(
    data = [[locsim(l1,l2) for l1 in locs] for l2 in tqdm.tqdm(locs)],
    index=locs,
    columns=locs,
)
#%%
np.fill_diagonal(loc_sim_M.values, -np.inf)
loc_sim_M
# %%
r = loc_sim_M.max(axis=1).idxmax()
c = loc_sim_M.loc[:,[r]].iloc[:,0].idxmax()
#%%
r,c
#%%
loc_sim_M.loc[[r],[c]]
#%%
loc_sim_M.max(axis=0)
# %%
