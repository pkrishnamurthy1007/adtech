#%%
from models.utils import wavg, get_wavg_by, wstd
from IPython.display import display as ipydisp
from models.utils.rpc_est import get_split_factor
import models.utils.rpc_est
import importlib
from notebooks.aduriseti_shared.utils import *
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime
import itertools
import collections
import pprint
from utils.env import load_env_from_aws
import sys
import re
import os
# detect if we are running from a `notebooks/*_shared` folder
# re.match("not.*shared",sys.path[0])
if sys.path[0].endswith("_shared"):
    sys.path[0] = "/".join(sys.path[0].split("/")[:-2])
assert sys.path[0].endswith("adtech")

load_env_from_aws()


NOW = datetime.datetime.now()
TODAY = NOW.date()
DAY = datetime.timedelta(days=1)

start_date = TODAY - 90*DAY
eval_date = TODAY - 30*DAY
end_date = TODAY

split_cols = ["state", "device", "keyword"]
rps_df = agg_rps(start_date, end_date, None, traffic_source=TABOOLA,
                 agg_columns=tuple(["campaign_id", *split_cols, "utc_dt"]))
rps_df = translate_taboola_vals(rps_df)
#%%
importlib.reload(models.utils.rpc_est)
AggRPSClust = models.utils.rpc_est.AggRPSClust
TreeRPSClust = models.utils.rpc_est.TreeRPSClust
KpiSimClust = models.utils.rpc_est.KpiSimClust
HybridCorrTreeClust = models.utils.rpc_est.HybridCorrTreeClust

# rps_df = rps_df \
#     .reset_index() \
#     .set_index(["state","device","keyword","utc_dt"]) \
#     .sort_index()
rps_df = rps_df.reset_index()
rps_df["leads"] = rps_df["num_leads"].fillna(0)
rps_df["lps"] = rps_df["leads"] / rps_df["sessions"]
rps_df["rpl"] = rps_df["revenue"] / rps_df["leads"]
rps_df["score"] = rps_df[["score_null_avg",
                          "score_adv_avg", "score_supp_avg"]].sum(axis=1)
rps_df["rps"] = rps_df["rps_avg"]
fitI = rps_df['utc_dt'].dt.date < eval_date
fitI.index = rps_df.index
rps_df["rps_"] = rps_df["revenue"] / rps_df["sessions"]
delta = rps_df["rps"] - rps_df["rps_"]
assert delta.abs().max() < 1e-10
assert abs(rps_df["revenue"].sum() / rps_df["sessions"].sum() -
           wavg(rps_df["rps"], rps_df["sessions"])) < 1e-10

clusterer = TreeRPSClust(clusts=32).fit(
    rps_df[fitI].set_index([*split_cols, "utc_dt"]), None)
rps_df.loc[fitI, "clust"] = clusterer.transform(
    rps_df[fitI].set_index([*split_cols, "utc_dt"]))
rps_df.loc[~fitI, "clust"] = clusterer.transform(
    rps_df[~fitI].set_index([*split_cols, "utc_dt"]))
rps_df["clust"] = rps_df["clust"].fillna(-1)
rps_df["rps_clust"] = rps_df \
    .groupby(["clust", "utc_dt"])["rps"].transform(get_wavg_by(rps_df, "sessions"))
daily_rps_mae = (rps_df["rps"] - rps_df["rps_clust"]).abs()
assert abs(
    wavg(rps_df["rps_clust"], rps_df["sessions"]) -
    wavg(rps_df["rps"], rps_df["sessions"])) < 1e-10
#%%
rps_df[~fitI].groupby("clust") \
    .apply(lambda df: wavg(df[kpis_lead].fillna(0).values, df['leads']))
#%%
kpis_agg = ["revenue", "sessions", "leads"]
kpis_session = ["rps", "lps"]
kpis_lead = ["rpl"]
clust_rps_df = rps_df[~fitI].groupby("clust")[kpis_agg].sum()
clust_rps_df[kpis_session] = rps_df[~fitI].groupby("clust") \
    .apply(lambda df: wavg(df[kpis_session], df['sessions']))
clust_rps_df[kpis_lead] = rps_df[~fitI].groupby("clust") \
    .apply(lambda df: wavg(df[kpis_lead].fillna(0).values, df['leads']))
clust_rps_df["rps_"] = clust_rps_df["revenue"] / clust_rps_df["sessions"]
clust_rps_df["rpl_"] = clust_rps_df["revenue"] / clust_rps_df['leads']
# agg_rps_df = rps_df[~fitI].groupby(rps_df.index.names[:-1]).agg({
#         "sessions": sum,
#         "rps": get_wavg_by(rps_df[~fitI],"sessions")
#     })
ipydisp(clust_rps_df)
# assert clust_rps_df["rps"].max() <= agg_rps_df["rps"].max()
# rps_wavg = wavg(agg_rps_df[["rps"]], agg_rps_df["sessions"])
rps_wavg = wavg(rps_df[~fitI]["rps"], rps_df[~fitI]["sessions"])
rps_clust_wavg = wavg(clust_rps_df[["rps"]], clust_rps_df["sessions"])
assert all((rps_wavg - rps_clust_wavg).abs()
           < 1e-3), (rps_wavg, rps_clust_wavg)
rps_wavg, rps_clust_wavg

perfd = {
    "clusterer": clusterer,
    # "fit_shape": agg_rps_df.shape,
    "clust_shape": clust_rps_df.shape,
    # "split_variance": wstd(agg_rps_df["rps"], agg_rps_df["sessions"]),
    "cluster_variance": wstd(clust_rps_df["rps"], clust_rps_df["sessions"]),
    # wstd(rps_df["rps_avg"],rps_df["sessions"])
    # "clustered_split_factor": get_split_factor(rps_df),
    "rps_mae": wavg(daily_rps_mae, rps_df["sessions"]),
}
pprint.pprint(perfd)
#%%
# clust_rps_df = rps_df[~fitI].groupby("clust")[kpis_agg].sum()
# clust_rps_df[kpis_session] = rps_df[~fitI].groupby("clust") \
#                                 .apply(lambda df: wavg(df[kpis_session],df['sessions']))
# clust_rps_df[kpis_lead[0]] = rps_df[~fitI].groupby("clust") \
#                             .apply(lambda df: wavg(df[kpis_lead[0]], df['leads']))

clust_dt_rps_df = rps_df.groupby(["clust", "utc_dt"])[kpis_agg].sum()
clust_dt_rps_df[kpis_session] = rps_df.groupby(["clust", "utc_dt"]) \
    .apply(lambda df: wavg(df[kpis_session], df['sessions']))
clust_dt_rps_df[kpis_lead] = rps_df[~fitI].groupby(["clust", "utc_dt"]) \
    .apply(lambda df: wavg(df[kpis_lead].fillna(0).values, df['leads']))
clust_dt_rps_df = clust_dt_rps_df.groupby("clust") \
    .apply(lambda df:
           df
           .reset_index("clust", drop=True)
           .reindex(pd.date_range(start_date, end_date)).fillna(method="ffill"))
kpis_session_7dcma = [f"{kpi}_7dcma" for kpi in kpis_session]
clust_dt_rps_df[kpis_session_7dcma] = clust_dt_rps_df.groupby("clust") \
    .apply(lambda df:
           df
           .reset_index("clust", drop=True)
           [kpis_session].rolling(7).mean())
# for ci in clust_dt_rps_df.index.unique("clust"):
#     clust_dt_rps_df.loc[ci, "rps"].plot()
# %%
clust_dt_rps_df.loc[(slice(None), TODAY.__str__()), :].round(3)
# %%
rps_df.groupby(["campaign_id", "utc_dt"]) \
    .apply(lambda df: wavg(df[["rps", "rps_clust"]], df["sessions"]))
# %%
