#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from models.taboola.common import *
#%%
from pytaboola import TaboolaClient
from pytaboola.services import AccountService, CampaignService, CampaignSummaryReport
# d = CampaignSummaryReport(client, O65_ACCNT_ID).fetch(
#     dimension="campaign_day_breakdown",start_date=TODAY-7*DAY, end_date=TODAY)
# import jmespath
# jmespath.search("results[?cpc > `0`].{cpc: cpc,campaign_id: campaign, utc_dt: date}",d)

client = TaboolaClient(**TABOOLA_HC_CREDS)

import jmespath
import requests
resp = requests.get(
    f"{TABOOLA_BASE}/{O65_ACCNT_ID}/allowed-publishers/",
    headers=client.authorization_header)
resp.raise_for_status()
TABOOLA_PUBLISHERS = jmespath.search('results[].account_id', resp.json())
#%%
from models.utils.rpc_est import TreeRPSClust

class TaboolaRPSEst(TreeRPSClust):
    def __init__(self,
                 clusts=None,
                 cma=7,
                 enc_min_cnt=100,
                 plot=True,
                 leads_threshold=15,
                 leads_lookback=7,
                 sessions_threshold=100,
                 sessions_lookback=30):
        self.leads_thresh = leads_threshold
        self.leads_lookback = leads_lookback
        self.sess_thresh = sessions_threshold
        self.sess_lookback = sessions_lookback
        super(TaboolaRPSEst, self).__init__(clusts, cma, enc_min_cnt, plot)


    def predict(self,X):
        X = X.copy()
        X["orig_order"] = range(len(X))
        X["clust"] = self.transform(X)

        X["r / rpl"] = self.rollup(X,y=X["leads"],w=X["leads"],sample_thresh=self.leads_thresh)
        X["rpl * l"] = self.rollup(X,y=X["revenue"],w=X["leads"],sample_thresh=self.leads_thresh)
        X["l / lps"] = self.rollup(X,y=X["sessions"],w=X["sessions"],sample_thresh=self.sess_thresh)
        X["lps * s"] = self.rollup(X,y=X["leads"],w=X["sessions"],sample_thresh=self.sess_thresh)
        kpis_agg = ["r / rpl","rpl * l","l / lps","lps * s"]
        # kpis_agg = ["revenue", "sessions", "leads"]
        clust_dt_rps_df = X.groupby(["clust", "utc_dt"])[kpis_agg].first()

        # 30 is a good breakpt for using bag mtd
        min_date = X.index.unique("utc_dt").min()
        max_date = X.index.unique("utc_dt").max()
        clust_dt_rps_df = clust_dt_rps_df.groupby("clust") \
            .apply(lambda df:
                df
                .reset_index("clust", drop=True)
                .reindex(pd.date_range(min_date, max_date)).fillna(method="ffill"))
        clust_dt_rps_df.index.names = ["clust", "utc_dt"]

        def get_nday_sum(c, n):
            def f(df):
                return df.groupby("clust") \
                    .apply(lambda df:
                        df
                        .reset_index("clust", drop=True)
                        [[c]].rolling(n).sum())[c]
            return f

        # TODO: make time windows configurable in hyperparamas
        rpl = get_nday_sum("rpl * l", self.leads_lookback)(clust_dt_rps_df) / \
            (get_nday_sum("r / rpl", self.leads_lookback)(clust_dt_rps_df) + 1e-10)
        lps = get_nday_sum("lps * s", self.sess_lookback)(clust_dt_rps_df) / \
            (get_nday_sum("l / lps", self.sess_lookback)(clust_dt_rps_df) + 1e-10)
        clust_dt_rps_df["rps_est"] = rpl * lps

        if self.plot:
            for ci in lps.index.unique("clust"):
                lps.loc[ci].plot()
            plt.title("LPS per cluster")
            plt.show()

            # plt.title("Overall RPL by date")
            # rpl[ci].plot()
            # plt.show()

            for ci in clust_dt_rps_df.index.unique("clust"):
                clust_dt_rps_df.loc[ci, "rps_est"].plot(label=ci)
            # plt.legend()
            plt.title("RPS = LPS*RPL by cluster")
            plt.show()

        X["rps_est"] = X \
            .reset_index() \
            .set_index(["clust","utc_dt"])[[]] \
            .join(clust_dt_rps_df["rps_est"]).values

        return X.sort_values(by="orig_order")["rps_est"].values
#%%
import itertools
cross = itertools.product
import jmespath
get = jmespath.search

def yield_from_type_pivot(camp,k):
    for t,v in cross(
            get(f"[{k}.type]", camp),
            get(f"{k}.value", camp) or [""]):
        yield [(k,(t,v)),1]

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
            yield [(k, (day, hr)), v]

def yield_bid_strategy_rules(camp):
    k = "publisher_bid_strategy_modifiers"
    for r in get(f"{k}.values[*].*",camp):
        yield [(k,tuple(r)),1]

def yield_bid_modifiers(camp):
    k = "publisher_bid_modifier"
    for site,mod in get(f"{k}.values[*].*", camp):
        yield [(k, site), mod]

flat_K = [
    "advertiser_id",
    "id",
    "cpc",
    "safety_rating",
    "daily_cap",
    "daily_ad_delivery_model",
    "bid_type",
    "bid_strategy",
    "traffic_allocation_mode",
    "marketing_objective",
    "is_active",
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
    campd = {("attrs",k): camp[k] for k in flat_K}
    for k in type_pivoted_K:
        campd.update(dict(yield_from_type_pivot(camp,k)))
    campd.update(dict(yield_time_rules(camp)))
    campd.update(dict(yield_bid_strategy_rules(camp)))
    campd.update(dict(yield_bid_modifiers(camp)))
    return campd
#%%
