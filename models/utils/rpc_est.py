import datetime
DAY = datetime.timedelta(days=1)
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from IPython.display import display as ipydisp

import sklearn.preprocessing
import sklearn.cluster
import sklearn.tree

from models.utils import wavg, wstd


CLUSTS = 8

def get_split_factor(rps_df):
    split_attr2unique_vals = {c: rps_df.index.unique(
        c) for c in rps_df.index.names[:-1]}
    _, new_index_order = zip(*sorted((V.__len__(), c)
                                     for c, V in split_attr2unique_vals.items()))
    return rps_df.reset_index()[[*new_index_order[:-1], "clust"]].drop_duplicates().__len__()


class AggRPSClust:
    def __init__(self, clusts=CLUSTS, kpis=["rps", "score", "lps", "rpl"]):
        self.clusts = clusts
        self.kpis = kpis

    def fit(self, X, _):
        #         globals()["X"] = X
        assert X.index.names[-1] == "utc_dt"
        aggX = X.groupby(X.index.names[:-1])[["sessions", "revenue"]].sum()
        aggX[self.kpis] = X.groupby(X.index.names[:-1]) \
            .apply(lambda df: wavg(df[self.kpis], df["sessions"]))
        if len(aggX) > self.clusts:
            aggX["clust"] = sklearn.cluster \
                .KMeans(n_clusters=self.clusts) \
                .fit_predict(aggX[self.kpis], sample_weight=aggX["sessions"])
        else:
            aggX["clust"] = np.arange(len(aggX))
        self.aggX = aggX
        return self

    def transform(self, X):
        #         globals()["X"] = X
        assert X.index.names[-1] == "utc_dt"
        X["clust"] = 1
        return X["clust"] * self.aggX["clust"]

    def fit_transform(self, X, _):
        return self.fit(X, _).transform(X)


class TreeRPSClust:
    def __init__(self, clusts=None, cma=7, enc_min_cnt=100, plot=True):
        self.clusts = clusts
        self.cma = cma
        self.enc_min_cnt = enc_min_cnt
        self.plot = plot

    def fit(self, X, y=None, w=None):
        #         globals()["X"] = X
        assert X.index.names[-1] == "utc_dt"

        split_idx = X.index.names[:-1]

        def translate(df, td):
            tdf = df.copy().reset_index()
            tdf['utc_dt'] = tdf['utc_dt'] + td
            return tdf.set_index(df.index.names)
        X = pd.concat((translate(X[["sessions", "revenue"]], n*DAY) / self.cma
                       for n in range(self.cma))) \
            .sum(level=[X.index.names])
        X["rps"] = X["revenue"] / X["sessions"]

        Xdf = X .reset_index()
        ydf = X["rps"]
        wdf = X["sessions"]

        if self.plot:
            ipydisp(Xdf[split_idx].isna().sum())
        for c in split_idx:
            too_few_I = Xdf.groupby(c)["sessions"].transform(sum) < self.enc_min_cnt
            Xdf.loc[too_few_I, c] = np.NaN
        if self.plot:
            ipydisp(Xdf[split_idx].isna().sum())

        self.enc_1hot = sklearn.preprocessing.OneHotEncoder(
            handle_unknown="ignore") .fit(Xdf[split_idx])
        self.enc_features = [*self.enc_1hot.get_feature_names()]
        X = self.enc_1hot.transform(Xdf[split_idx])
        y = ydf
        w = wdf

        if self.plot:
            print("|X|",X.shape,"|y|",y.shape,"|w|",w.shape)

        if self.clusts:
            self.clf = sklearn.tree.DecisionTreeRegressor(
                min_weight_fraction_leaf=0.5/self.clusts) \
                .fit(X, y, sample_weight=wdf)
        else:
            self.clf = sklearn.tree.DecisionTreeRegressor() \
                .fit(X, y, sample_weight=wdf)
        
        if self.plot:
            print(sklearn.tree.export_text(self.clf, feature_names=self.enc_features))
            yhat = self.clf.predict(X)
            print("Tree RPS MAE:", (y - yhat).abs().mean())

        return self

    def transform(self, X):
        assert X.index.names[-1] == "utc_dt"
        Xdf = X .reset_index()[X.index.names].iloc[:, :-1]
        X = self.enc_1hot.transform(Xdf)
        print("|X|", X.shape)
        return self.clf.apply(X)

    def fit_transform(self, X, _):
        return self.fit(X, _).transform(X)

    def rollup(self,X,y,w,sample_thresh=100):
        Xdf = X
        X = X .reset_index()[X.index.names].iloc[:, :-1]
        X = self.enc_1hot.transform(X)
        print("|X|", X.shape)
        P = self.clf.decision_path(X)
        
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

        return total_y


COKPI = "COKPI"
COOR = "COOR"
MEAN = "MEAN"
STACK = "STACK"
class KpiSimClust:
    def __init__(self, clusts=CLUSTS, kpis=["rps", "score", "lps", "rpl"], sim=COKPI, mtd=MEAN, plot=True):
        self.clusts = clusts
        self.kpis = kpis
        self.plot = plot
        self.sim = sim
        self.mtd = mtd

    def fit(self, X, _):
        #         globals()["X"] = X
        assert X.index.names[-1] == "utc_dt"

        aggX = X.groupby(X.index.names[:-1])[["sessions", "revenue"]].sum()
        aggX[self.kpis] = X.groupby(X.index.names[:-1]) \
            .apply(lambda df: wavg(df[self.kpis], df["sessions"]))
        buckets = aggX.index.values

        min_date = X.index.unique("utc_dt").min()
        max_date = X.index.unique("utc_dt").max()
        date_range = pd.date_range(min_date, max_date)
        kpi_tensor = np.stack(X.loc[bucket, self.kpis]
                              .reindex(date_range).fillna(0)
                              .rolling(7).mean().fillna(0)
                              for bucket in buckets)
        kpi_tensor = kpi_tensor.transpose(2, 0, 1)
        D, H, W = kpi_tensor.shape
        kpi_tensor.shape

        if self.plot:
            # i = (kpi_tensor > 0).sum(axis=1)[:,0].argmax()
            # i = kpi_tensor.sum(axis=2)[0,:].argmax()
            i = (kpi_tensor > 1e-3).sum(axis=2)[0].argmax()
            plt.plot(kpi_tensor[0, i, :])
            plt.show()

        if self.sim == COOR:
            mu = kpi_tensor.mean(axis=2).reshape(D, H, 1)
            std = kpi_tensor.std(axis=2).reshape(D, H, 1)
            kpi_tensor_norm = (kpi_tensor - mu) / std
            kpi_tensor_norm[np.isnan(kpi_tensor_norm)] = 0
            kpi_corr = (kpi_tensor_norm @
                        kpi_tensor_norm.transpose(0, 2, 1)) / W
#             assert np.abs(np.diag(loc_corr_df) - 1).max() < 1e-10
            assert (np.einsum('dii->di', kpi_corr) - 1).abs().max() < 1e-10
            kpi_sim = kpi_corr
        elif self.sim == COKPI:
            kpi_sqrt_tensor = kpi_tensor ** 0.5
            cokpi = (kpi_sqrt_tensor @ kpi_sqrt_tensor.transpose(0, 2, 1))
#             cokpi = np.nan_to_num(np.log(cokpi))
            ma = cokpi.max(axis=2).max(axis=1).reshape(D, 1, 1)
#             cokpi = cokpi / (ma + 1e-10)
            cokpi = cokpi / ma
            kpi_sim = cokpi
        else:
            raise

        if self.mtd == MEAN:
            kpi_sim = kpi_sim.mean(axis=0)
        elif self.mtd == STACK:
            kpi_sim = kpi_sim.transpose(1, 0, 2).reshape(H, D*W)
        else:
            raise

        kpi_sim_df = pd.DataFrame(kpi_sim, index=buckets)

        if len(aggX) > self.clusts:
            aggX["clust"] = sklearn.cluster \
                .KMeans(n_clusters=self.clusts) \
                .fit_predict(kpi_sim_df.values, sample_weight=aggX["sessions"])
        else:
            aggX["clust"] = np.arange(len(aggX))
        self.aggX = aggX

#         if plot:
#             for ci in range(CLUSTS):
#                 clust_kpi_df = pd.concat(kpi_df.loc[tuple(uval)] for uval in col_uvals[clust==ci]) \
#                     .reset_index()
#                 print(ci,"rps:",wavg(clust_kpi_df["rps"],clust_kpi_df["sessions"]))
#                 ipydisp(clust_kpi_df[["sessions",'revenue']].sum())
#                 clust_kpi_df \
#                     .groupby("utc_dt")["rps"] \
#                     .agg(get_wavg_by(clust_kpi_df,"sessions")) \
#                     .reindex(pd.date_range(eval_date-7*DAY,end_date)) \
#                     .fillna(0).rolling(7).mean() \
#                     .plot(label=ci,figsize=(15,5))
#             plt.legend()
#             plt.show()

        return self

    def transform(self, X):
        #         globals()["X"] = X
        assert X.index.names[-1] == "utc_dt"
        X["clust"] = 1
        return X["clust"] * self.aggX["clust"]

    def fit_transform(self, X, _):
        return self.fit(X, _).transform(X)


class HybridCorrTreeClust:
    def __init__(
            self,
            clusts=CLUSTS, enc_min_cnt=100,
            kpis=["rps", "score", "lps", "rpl"], plot=True):
        self.clusts = clusts
        self.enc_min_cnt = enc_min_cnt
        self.kpis = kpis
        self.plot = plot

    def fit(self, X, _):
        assert X.index.names[-1] == "utc_dt"
        split_idx = X.index.names[:-1]
        self.splitcol2clusterer = {}
        for c in split_idx:
            aggX = X.groupby([c, "utc_dt"])[["sessions", "revenue"]].sum()
            aggX[self.kpis] = X.groupby([c, "utc_dt"]) \
                .apply(lambda df: wavg(df[self.kpis], df["sessions"]))
            clusterer = KpiSimClust(
                clusts=self.clusts,
                kpis=self.kpis,
                plot=False) \
                .fit(aggX, None)
            clust = clusterer \
                .transform(X.reset_index().set_index([c, "utc_dt"]))
            self.splitcol2clusterer[c] = clusterer
            X[f"{c}_clust"] = clust.values

        clust_idx = [f"{c}_clust" for c in split_idx]
#         Xclust = X.groupby([*clust_idx,"utc_dt"]) \
#             [["sessions","revenue"]].sum()
#         Xclust[kpis] = X.groupby([*clust_idx,"utc_dt"]) \
#             .apply(lambda df: wavg(df[kpis],df["sessions"]))

        self.tree_clusterer = TreeRPSClust(
            clusts=self.clusts,
            enc_min_cnt=self.enc_min_cnt,
            plot=False) \
            .fit(X.reset_index().set_index([*clust_idx, "utc_dt"]), None)
        # clusterer = TreeRPSClust() \
        #     .fit(Xclust,None)

        return self

    def transform(self, X):
        assert X.index.names[-1] == "utc_dt"
        split_idx = X.index.names[:-1]
        clust_idx = [f"{c}_clust" for c in split_idx]
        for c, d in zip(split_idx, clust_idx):
            X[d] = self.splitcol2clusterer[c] \
                .transform(X.reset_index().set_index([c, "utc_dt"])) \
                .values
        return self.tree_clusterer \
            .transform(X.reset_index().set_index([*clust_idx, "utc_dt"]))

    def fit_transform(self, X, _):
        return self.fit(X, _).transform(X)
