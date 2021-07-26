#%%
from models.taboola.common import *
from notebooks.aduriseti_shared.utils import *
from models.utils import wavg, get_wavg_by, wstd

from models.taboola.utils import *

split_cols = ["state", "device", "keyword"]
rps_df = agg_rps(start_date, end_date, None, traffic_source=TABOOLA,
                 agg_columns=tuple(["campaign_id", *split_cols, "utc_dt"]))
rps_df = translate_taboola_vals(rps_df)
rps_df = rps_df_postprocess(rps_df)
rps_df = rps_df.reset_index()
#%%
import joblib
import models.taboola.utils
importlib.reload(models.taboola.utils)
TaboolaRPSEst = models.taboola.utils.TaboolaRPSEst
clusterer: TaboolaRPSEst = joblib.load(MODEL_PTH)
rps_est = clusterer.predict(rps_df.set_index([*split_cols,"utc_dt"]))
rps_df["rps_est"] = rps_est

rps_df_campaign = rps_df[rps_df["utc_dt"].dt.date > TODAY - 7*DAY] \
    .groupby(["campaign_id"])[["rps_est"]] \
    .agg(get_wavg_by(rps_df, "sessions"))
rps_df_publisher = rps_df \
        [rps_df["utc_dt"].dt.date > TODAY - 7*DAY] \
        .groupby(["campaign_id","keyword"])[["rps_est"]] \
        .agg(get_wavg_by(rps_df, "sessions")) \
        .unstack()
#%%
from pytaboola import TaboolaClient
from pytaboola.services import AccountService,CampaignService,CampaignSummaryReport
# d = CampaignSummaryReport(client, O65_ACCNT_ID).fetch(
#     dimension="campaign_day_breakdown",start_date=TODAY-7*DAY, end_date=TODAY)
# import jmespath
# jmespath.search("results[?cpc > `0`].{cpc: cpc,campaign_id: campaign, utc_dt: date}",d)

client = TaboolaClient(**TABOOLA_HC_CREDS)
acct_service = AccountService(client)
accnts = acct_service.list()["results"]
id2accnt = {a["account_id"]: a for a in accnts}

camps = []
for aid in [TEST_ACCNT_ID,O65_ACCNT_ID]:
    camp_service = CampaignService(client, aid)
    camps += camp_service.list()

campdf = pd.DataFrame([flatten_camp(camp) for camp in camps])
campdf = campdf.set_index(("attrs", "id"))
campdf.columns = pd.MultiIndex.from_tuples(campdf.columns)
print("|campdf|", campdf.shape)

print("campaign df sparsity:",((campdf == 0) | campdf.isna()).sum().sum() / np.prod(campdf.shape))
strC = campdf.dtypes[campdf.dtypes == object].index
floatC = campdf.dtypes[campdf.dtypes == np.float64].index
# %%
active_camps = {*active_camps} & {*campdf.index}
cpc_df_campaign_new = np.clip(
    rps_df_campaign["rps_est"].reindex(active_camps) / ROI_TARGET,
    (1-MAX_CUT)*campdf["attrs"].loc[active_camps,"cpc"],
    (1+MAX_PUSH)*campdf["attrs"].loc[active_camps,"cpc"])
cpc_df_campaign_new = cpc_df_campaign_new \
                        .combine_first(campdf["attrs"].loc[active_camps,"cpc"])

import requests
resp = requests.get(
    f"{TABOOLA_BASE}/{O65_ACCNT_ID}/allowed-publishers/",
    headers=client.authorization_header)
taboola_publishers = jmespath.search('results[].account_id', resp.json())

bid_mod_df = campdf["publisher_bid_modifier"] \
    .reindex(active_camps) \
    .T.reindex(taboola_publishers).T
cpc_df_publisher = bid_mod_df.fillna(1) * \
    campdf["attrs"].loc[active_camps, ["cpc"]].values
cpc_df_publisher_new = np.clip(
    rps_df_publisher["rps_est"] \
        .reindex(active_camps) \
        .T.reindex(taboola_publishers).T / ROI_TARGET,
    (1-MAX_CUT)*cpc_df_publisher,
    (1+MAX_PUSH)*cpc_df_publisher,)
bid_mod_df_new = cpc_df_publisher_new / cpc_df_campaign_new.values.reshape(-1,1)
approx1 = (bid_mod_df_new - 1).abs() < 1e-2
bid_mod_df_new = bid_mod_df_new.loc[:,~(bid_mod_df_new.isna() | approx1).all(axis=0)]
bid_mod_df_new = bid_mod_df_new \
    .combine_first(bid_mod_df.loc[:,~bid_mod_df.isna().any()])
campdf.loc[active_camps,("updates","cpc")] = cpc_df_campaign_new.round(2)
campdf.loc[active_camps,("updates","publisher_bid_modifier")] = \
    bid_mod_df_new.round(2).apply(
        lambda r: {
                "values": [{'target': c, "bid_modification": v} for c,v in r[~r.isna()].items()]
            },
        axis=1)

updatedf = pd.concat((
    campdf.loc[active_camps,"attrs"]["advertiser_id"],
    campdf.loc[active_camps,"updates"].apply(dict,axis=1).apply(json.dumps),
),axis=1) \
    .reset_index()
updatedf.columns = ["campaign_id","account_id","update"]
updatedf["date"] = TODAY
updatedf["datetime"] = NOW
updatedf
#%%
upload_taboola_updates_to_redshift(updatedf)
