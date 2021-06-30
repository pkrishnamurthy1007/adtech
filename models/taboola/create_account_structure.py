# %%
import sys
import re
import os
# detect if we are running from a `notebooks/*_shared` folder
# re.match("not.*shared",sys.path[0])
if sys.path[0].endswith("_shared"):
    sys.path[0] = "/".join(sys.path[0].split("/")[:-2])
assert sys.path[0].endswith("adtech")

from utils.env import load_env_from_aws
load_env_from_aws()

import pprint
from IPython.display import display as ipydisp    
import collections
import itertools
import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from models.utils import wavg,get_wavg_by
from notebooks.aduriseti_shared.utils import *

NOW = datetime.datetime.now()
DAY = datetime.timedelta(days=1)

campaign_split_fields = dict(
    # traffic_source = ["traffic_source"],
    browser = ["browser"],
    operating_system = ["operating_system"],
    device = ["device"],
    # channel = ["channel"],
    # domain = ["domain"],
    product = ["product"],
    # keyword = ["keyword"],
    # campaign_id = ["campaign_id"],
    # landing_page = ["landing_page"],
    TOD = ["dayofweek","hourofday"],
    dma = ["dma"],
    state =["state",],
    location = ["state","dma"],
    
    dma_os=["dma", "operating_system"],
    dma_device=["dma", "device", ],
    dma_os_device=["dma", "operating_system", "device"],

    state_os=["state", "operating_system"],
    state_device=["state", "device", ],
    state_os_device=["state", "operating_system", "device"],

    location_os = ["state", "dma", "operating_system"],
    location_device=["state", "dma", "device", ],
    location_os_device = ["state", "dma", "operating_system","device"],
)

def get_wthresh(W,p):
    W = rps_df["sessions"].sort_values(ascending=False)
    Wsum = W.sum()
    cumsum = 0
    for wthresh in W:
        if cumsum > Wsum * p:
            break
        cumsum += wthresh
    return wthresh

start_date = NOW - 90*DAY
eval_date = NOW - 30*DAY
end_date = NOW

split2aggrps = {}
for split,split_cols in campaign_split_fields.items():
    print(split,split_cols)
    rps_df = agg_rps(start_date,end_date,None,traffic_source=TABOOLA,agg_columns=tuple(split_cols+["utc_dt"]))
    rps_df = translate_taboola_vals(rps_df)
    rps_df["split_on"] = split
    split2aggrps[split] = rps_df
    print(split,rps_df.shape)


# %%
import sklearn.preprocessing
import sklearn.cluster
import sklearn.tree

class AggRPSClust:
    def __init__(self,minclust=8):
        self.minclust = minclust
        self.split_size = None
    
    def fit(self,X,_):
        globals()["X"] = X
        assert X.index.names[-1] == "utc_dt"
        split_idx = X.reset_index()[X.index.names[:-1]] .apply(tuple,axis=1).apply(str)
        self.split_size = split_idx.drop_duplicates().__len__()
        if self.split_size <= self.minclust:
            self.ord_enc = sklearn.preprocessing.OrdinalEncoder(
                    handle_unknown="use_encoded_value",unknown_value=-1) \
                .fit(split_idx.drop_duplicates().values.reshape(-1,1))
            self.ord_enc.transform(split_idx.values.reshape(-1,1))
        else:
            self.aggX = X                 .groupby(X.index.names[:-1])                 .agg({
                    "sessions": sum,
                    "rps": get_wavg_by(X,"sessions")
                })

            # nclust = MINCLUST
            nclust = max(self.minclust, int(np.log(self.split_size)))
            # nclust = max(MINCLUST,int(split_size ** 0.5))
            # print("nclust", nclust, split_size, np.log(split_size))
            
            self.aggX['clust'] = sklearn.cluster.KMeans(n_clusters=nclust)                             .fit_predict(self.aggX[["rps"]],sample_weight=self.aggX["sessions"])

        return self

    def transform(self,X):
        globals()["X"] = X
        assert X.index.names[-1] == "utc_dt"
        split_idx = X.reset_index()[X.index.names[:-1]] .apply(tuple,axis=1).apply(str)
        if self.split_size <= self.minclust:
            return self.ord_enc.transform(split_idx.values.reshape(-1,1))
        else:
            X["clust"] = 1
            X["clust"] *= self.aggX["clust"]
            return X["clust"]
        
    def fit_transform(self,X,_):
        raise NotImplementedError
        
class TreeRPSClust:
    def __init__(self):
        pass
    
    def fit(self,X,_):

        globals()["X"] = X
        
        Xdf = X .reset_index()[X.index.names].copy()
        ydf = X["rps"]
        wdf = X["sessions"]

        ipydisp(Xdf.isna().sum())

        mincnt = 100
        for c in Xdf.columns:
            too_few_I = Xdf.groupby(c).transform("count").iloc[:,0] < mincnt
            Xdf.loc[too_few_I,c] = np.NaN

        ipydisp(Xdf.isna().sum())
        Xdf = Xdf.fillna("")
        Xdf = Xdf.iloc[:,:-1]

        self.enc_1hot = sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore") .fit(Xdf)
        self.enc_features = [*self.enc_1hot.get_feature_names()]
        X = self.enc_1hot.transform(Xdf)
        print("|X|",X.shape)
        y = ydf.fillna(0)
        w = wdf

        self.clf = sklearn.tree.DecisionTreeRegressor(min_samples_leaf=30)                         .fit(X,y,sample_weight=wdf)
        print(sklearn.tree.export_text(self.clf,feature_names=self.enc_features))
        
        
        yhat = self.clf.predict(X)
        print("Tree RPS MAE:",(y - yhat).abs().mean())

        return self
        
    def transform(self,X):
        globals()["X"] = X
        Xdf = X .reset_index()[X.index.names]
        X = self.enc_1hot.transform(Xdf.fillna("").values[:,:-1])
        print("|X|",X.shape)
        return self.clf.apply(X)

from models.utils import wstd
def get_split_factor(rps_df):
    split_attr2unique_vals = {c: rps_df.index.unique(c) for c in rps_df.index.names[:-1]}
    _,new_index_order = zip(*sorted((V.__len__(),c) for c,V in split_attr2unique_vals.items()))
    return rps_df.reset_index()[[*new_index_order[:-1],"clust"]].drop_duplicates().__len__()
# %%
perfD = []
for split,rps_df in split2aggrps.items():
    
    rps_df["lps"] = rps_df["num_leads"] / rps_df["sessions"]
    rps_df["rpl"] = rps_df["revenue"] / rps_df["num_leads"]
    rps_df["score"] = rps_df[["score_null_avg","score_adv_avg","score_supp_avg"]].sum(axis=1)
    rps_df["rps"] = rps_df["rps_avg"]
    fitI = rps_df.reset_index()['utc_dt'] < eval_date
    fitI.index = rps_df.index

    for clust_mtd in [AggRPSClust,TreeRPSClust]:
        clusterer = clust_mtd().fit(rps_df[fitI],None)
        rps_df["clust"] = -1
        rps_df.loc[fitI,"clust"] = clusterer.transform(rps_df[fitI])
        rps_df.loc[~fitI,"clust"] = clusterer.transform(rps_df[~fitI])

        agg_rps_df = rps_df             [~fitI]             .groupby(rps_df.index.names[:-1])             .agg({
                "sessions": sum,
                "rps": get_wavg_by(rps_df[~fitI],"sessions")
            })
        clust_rps_df = rps_df             [~fitI]             .groupby("clust")             .agg({
                "sessions": sum,
                "rps": get_wavg_by(rps_df[~fitI],"sessions")
            })

        assert clust_rps_df["rps"].max() <= agg_rps_df["rps"].max()
        rps_wavg = wavg(agg_rps_df[["rps"]], agg_rps_df["sessions"])
        rps_clust_wavg = wavg(clust_rps_df[["rps"]], clust_rps_df["sessions"])
        assert all((rps_wavg - rps_clust_wavg).abs() < 1e-2), (rps_wavg, rps_clust_wavg)


        perfd = {
            "clust_mtd": clusterer.__class__.__name__,
            "clusterer": clusterer,
            "split": split,
            "fit_shape": agg_rps_df.shape,
            "clust_shape": clust_rps_df.shape,
            "split_variance": wstd(agg_rps_df["rps"], agg_rps_df["sessions"]),
            "cluster_variance": wstd(clust_rps_df["rps"], clust_rps_df["sessions"]),
            # wstd(rps_df["rps_avg"],rps_df["sessions"])
            "clustered_split_factor": get_split_factor(rps_df),
        }
        perfD.append(perfd)
        pprint.pprint(perfd)
        ipydisp(clust_rps_df)

perfdf = pd.DataFrame(perfD)
ipydisp(perfdf)


# %%
split,clusterer = perfdf.loc[27,["split","clusterer"]]
rps_df = split2aggrps[split]
rps_df["lps"] = rps_df["num_leads"] / rps_df["sessions"]
rps_df["rpl"] = rps_df["revenue"] / rps_df["num_leads"]
rps_df["score"] = rps_df[["score_null_avg","score_adv_avg","score_supp_avg"]].sum(axis=1)
rps_df["rps"] = rps_df["rps_avg"]
fitI = rps_df.reset_index()['utc_dt'] < eval_date
fitI.index = rps_df.index

rps_df["clust"] = -1
rps_df.loc[fitI,"clust"] = clusterer.transform(rps_df[fitI])
rps_df.loc[~fitI,"clust"] = clusterer.transform(rps_df[~fitI])

split_attr2unique_vals = {index_col: rps_df.index.unique(index_col) 
                          for index_col in rps_df.index.names[:-1]}
_,new_index_order = zip(*sorted((V.__len__(),c) for c,V in split_attr2unique_vals.items()))
rps_df = rps_df.reset_index()
campaign_df = rps_df     .groupby([*new_index_order[:-1], "clust"])     .agg({
        "sessions": sum,
        "rps_avg": get_wavg_by(rps_df,"sessions"),
        new_index_order[-1]: lambda seq: tuple(set(seq))
    }) \
    .sort_index()

assert campaign_df["sessions"].sum() == rps_df["sessions"].sum()
camp_rps_wavg = wavg(campaign_df["rps_avg"],campaign_df["sessions"])
fit_rps_wavg = wavg(rps_df["rps_avg"], rps_df["sessions"])
assert abs(camp_rps_wavg - fit_rps_wavg) < 1e-5
# %%
camps = []
for idx,r in campaign_df.iterrows():
    camp = {
        "sessions_60d": r["sessions"], 
        "rps_avg_60d": r["rps_avg"]
    }
    for field,val in zip(new_index_order[:-1],idx):
        camp[field] = {"INCLUDE": val}
    last_field = new_index_order[-1]
    camp[last_field] = {
        "INCLUDE": r[last_field]
    }
    camps.append(camp)

excl_campaign_df = campaign_df.groupby([*new_index_order[:-1]])     .agg({
        new_index_order[-1]: tuple
    })
def flatten(M):
    return tuple(el for r in M for el in r)
excl_campaign_df[new_index_order[-1]] = excl_campaign_df[new_index_order[-1]]     .apply(flatten)
for idx, r in excl_campaign_df.iterrows():
    camp = {}
    for field, val in zip(new_index_order[:-1], idx):
        camp[field] = {"INCLUDE": val}
    last_field = new_index_order[-1]
    camp[last_field] = {
        "EXCLUDE": r[last_field]
    }
    camps.append(camp)

camp_attr_df = pd.DataFrame(camps)
camp_attr_df.to_csv("campaign_dump.csv")
camp_attr_df
# %%
import json

TABOOLA_HC_CREDS = json.loads(os.getenv("TABOOLA_HC_CREDS"))
TABOOLA_PIVOT_CREDS = json.loads(os.getenv("TABOOLA_PIVOT_CREDS"))

from pytaboola import TaboolaClient
client = TaboolaClient(**TABOOLA_HC_CREDS)
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
NETWORK_ACCNT_ID = "healthcareinc-network"
TEST_ACCNT_ID = "healthcareinc-sc2"
O65_ACCNT_ID = "taboolaaccount-rangaritahealthcarecom"
id2accnt = {a["account_id"]: a for a in accnts}

def accnt_camps(accnt):
    camp_service = CampaignService(client, accnt["account_id"])
    return camp_service.list()
aid2cid2camp = {}
for aid,a in tqdm.tqdm(id2accnt.items()):
    cid2camp = {c["id"]: c for c in accnt_camps(a)}
    aid2cid2camp[aid] = cid2camp

import json
json.dump(aid2cid2camp, open("camps.json", "w"))
O65_accnt_camps = accnt_camps(id2accnt[O65_ACCNT_ID])
print("|065_accnt_camps|:",len(O65_accnt_camps))

test_accnt_camps = accnt_camps(id2accnt[TEST_ACCNT_ID])
print("|test_accnt_camps|",len(test_accnt_camps))

import itertools
cross = itertools.product
import jmespath
get = jmespath.search

active_camps = get(
  "*.*[] | [?is_active]",
  aid2cid2camp,
)
print("|active campaigns|:",len(active_camps))
# %%
def taboola_targetting_fields(r):
    def pivot(o):
        (t,v),*_ = o.items()
        return {
            "type": t,
            "value": [*v] if isinstance(v,(list,tuple,set)) else [v]
        }
    targetting_fields = {}
    if "device" in r:
        targetting_fields["platform_targeting"] = pivot(r["device"])
    if "state" in r:
        targetting_fields["sub_country_targeting"] = pivot(r["state"])
    if "dma" in r:
        targetting_fields["dma_country_targeting"] = pivot(r["dma"])
    if "operating_system" in r:
        (t,v),*_ = r["operating_system"].items()
        targetting_fields["os_targeting"] = {
            "type": "INCLUDE",
            "value": [
                {
                    "os_family": v
                },
            ],
        }
    return targetting_fields
ipydisp(camp_attr_df.apply(taboola_targetting_fields,axis=1).head())
taboola_targetting_fields(camp_attr_df.iloc[0,:])


# %%
import requests
TODAY = NOW.date()
TABOOLA_BASE = "https://backstage.taboola.com/backstage/api/1.0"
for src_camp in active_camps:
    src_accnt = id2accnt[O65_ACCNT_ID]
    dest_accnt = id2accnt[TEST_ACCNT_ID]

    client = TaboolaClient(**TABOOLA_HC_CREDS)
    client.authorization_header,client.token_details

    for _,camp_attrs in camp_attr_df.iterrows():
        resp = requests.post(
            f"{TABOOLA_BASE}/{src_accnt['account_id']}/campaigns/{src_camp['id']}/duplicate",
            json={
                "name": src_camp["id"],
                "start_date": (TODAY + 2*DAY).__str__(),
                **taboola_targetting_fields(camp_attrs),
            },
            params= {
                "destination_account": dest_accnt["account_id"],
            },
            headers=client.authorization_header,)
        resp.raise_for_status()

        new_camp = resp.json()
# %%
