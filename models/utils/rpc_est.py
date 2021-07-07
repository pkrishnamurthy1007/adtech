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
    def __init__(self, clusts=CLUSTS, cma=7, enc_min_cnt=100, plot=True):
        self.clusts = clusts
        self.cma = cma
        self.enc_min_cnt = enc_min_cnt
        self.plot = plot

    def fit(self, X, _):
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

        ipydisp(Xdf[split_idx].isna().sum())
        for c in split_idx:
            too_few_I = Xdf.groupby(c)["sessions"].transform(
                sum) < self.enc_min_cnt
            Xdf.loc[too_few_I, c] = np.NaN
        ipydisp(Xdf[split_idx].isna().sum())
        #         Xdf = Xdf.astype(str).fillna("")
        Xdf = Xdf[split_idx]

        self.enc_1hot = sklearn.preprocessing.OneHotEncoder(
            handle_unknown="ignore") .fit(Xdf)
        self.enc_features = [*self.enc_1hot.get_feature_names()]
        X = self.enc_1hot.transform(Xdf)
        print("|X|", X.shape)
        y = ydf.fillna(0)
        w = wdf

        self.clf = sklearn.tree.DecisionTreeRegressor(
            min_weight_fraction_leaf=0.5/self.clusts) \
            .fit(X, y, sample_weight=wdf)
        print(sklearn.tree.export_text(self.clf, feature_names=self.enc_features))

        yhat = self.clf.predict(X)
        print("Tree RPS MAE:", (y - yhat).abs().mean())

        return self

    def transform(self, X):
        #         globals()["X"] = X
        assert X.index.names[-1] == "utc_dt"
        Xdf = X .reset_index()[X.index.names].iloc[:, :-1]
#         X = self.enc_1hot.transform(Xdf.astype(str).fillna(""))
        X = self.enc_1hot.transform(Xdf)
        print("|X|", X.shape)
        return self.clf.apply(X)

    def fit_transform(self, X, _):
        return self.fit(X, _).transform(X)


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
