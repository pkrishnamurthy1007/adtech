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
rps_model: TaboolaRPSEst = joblib.load(MODEL_PTH)
# rps_model = TaboolaRPSEst(clusts=None,enc_min_cnt=10).fit(
#     rps_df.set_index([*split_cols, "utc_dt"]), None)

rps_df["rps_est"] = rps_model.predict(rps_df.set_index([*split_cols,"utc_dt"]))

rps_df_campaign = rps_df[rps_df["utc_dt"].dt.date > TODAY - 7*DAY] \
    .groupby(["campaign_id"])[["rps_est"]] \
    .agg(get_wavg_by(rps_df, "sessions"))
rps_df_publisher = rps_df \
        [rps_df["utc_dt"].dt.date > TODAY - 7*DAY] \
        .groupby(["campaign_id","keyword"])[["rps_est"]] \
        .agg(get_wavg_by(rps_df, "sessions")) \
        .unstack()

# rps_df["clust"] = rps_model.transform(rps_df.set_index([*split_cols,"utc_dt"]))
# rps_df["clust_sessions"] = rps_df.groupby(["utc_dt","clust"])["sessions"].transform(sum)
# rps_df["clust_leads"] = rps_df.groupby(["utc_dt","clust"])["leads"].transform(sum)
# rps_df["clust_revenue"] = rps_df.groupby(["utc_dt","clust"])["revenue"].transform(sum)
#%%
with HealthcareDW() as db:
    campaign_data_df = db.to_df(MOST_RECENT_CAMPAIGN_DATA_SQL)
campdf = pd.DataFrame(campaign_data_df["body"].apply(json.loads).apply(flatten_camp).tolist())
campdf = campdf.set_index(("attrs", "id"))
campdf.columns = pd.MultiIndex.from_tuples(campdf.columns)
print("|campdf|", campdf.shape)

print("campaign df sparsity:",((campdf == 0) | campdf.isna()).sum().sum() / np.prod(campdf.shape))
strC = campdf.dtypes[campdf.dtypes == object].index
floatC = campdf.dtypes[campdf.dtypes == np.float64].index

with HealthcareDW() as db:
    campaign_config_df = db.to_df(f"""
        SELECT * FROM {DS_SCHEMA}.{TABOOLA_CAMPAIGN_MANAGEMENT_TABLE}
        WHERE automatically_managed = True 
    """)
campaign_config_df = campaign_config_df.set_index("id")
campaign_config_df.columns = pd.MultiIndex.from_product([["bidder_config"],campaign_config_df.columns])
campdf = campdf.join(campaign_config_df,how="left")
# %%
# active_camps = {*active_camps} & {*campdf.index}
active_camps = campdf.index
# cpc_df_campaign_new = np.clip(
#     rps_df_campaign["rps_est"].reindex(active_camps) / ROI_TARGET,
#     (1-MAX_CUT)*campdf["attrs"].loc[active_camps,"cpc"],
#     (1+MAX_PUSH)*campdf["attrs"].loc[active_camps,"cpc"])
cpc_df_campaign_new = rps_df_campaign["rps_est"].reindex(active_camps) / ROI_TARGET
cpc_df_campaign_new = cpc_df_campaign_new \
                        .combine_first(campdf["attrs"].loc[active_camps,"cpc"])

bid_mod_df = campdf["publisher_bid_modifier"] \
    .reindex(active_camps) \
    .T.reindex(TABOOLA_PUBLISHERS).T
# cpc_df_publisher = bid_mod_df.fillna(1) * \
#     campdf["attrs"].loc[active_camps, ["cpc"]].values
# cpc_df_publisher_new = np.clip(
#     rps_df_publisher["rps_est"] \
#         .reindex(active_camps) \
#         .T.reindex(TABOOLA_PUBLISHERS).T / ROI_TARGET,
#     (1-MAX_CUT)*cpc_df_publisher,
#     (1+MAX_PUSH)*cpc_df_publisher,)
cpc_df_publisher_new = rps_df_publisher["rps_est"] \
                            .reindex(active_camps) \
                            .T.reindex(TABOOLA_PUBLISHERS).T \
                            / ROI_TARGET
bid_mod_df_new = cpc_df_publisher_new / cpc_df_campaign_new.values.reshape(-1,1)
approx1 = (bid_mod_df_new - 1).abs() < 1e-2
bid_mod_df_new = bid_mod_df_new.loc[:,(~bid_mod_df_new.isna() & ~approx1).any()]
bid_mod_df_new = bid_mod_df_new \
    .combine_first(bid_mod_df.loc[:,(~bid_mod_df.isna()).any()])

campdf.loc[active_camps,("updates","cpc")] = cpc_df_campaign_new.round(2)
bid_mod_items = bid_mod_df_new \
    .round(2) \
    .apply(lambda r: r[~r.isna() & ~(r==0)] .sort_values(ascending=False).items(),axis=1) \
    .apply(list)
campdf.loc[active_camps,("updates","publisher_bid_modifier")] = \
    bid_mod_items.apply(
        lambda mods: {
            "values": [{'target': c, "bid_modification": v} 
                        for c,v in mods[:TABOOLA_MAX_PUBLISHER_MODS_PER_CAMPAIGN]]}
    )
campdf.loc[active_camps,("updates","publisher_targeting")] = \
    bid_mod_items.apply(
        lambda mods: {
            "type": "EXCLUDE",
            "values": [c for c,_ in mods[TABOOLA_MAX_PUBLISHER_MODS_PER_CAMPAIGN:-TABOOLA_MAX_PUBLISHER_EXCL_PER_CAMPAIGN]]},
    )

updatedf = pd.concat((
    campdf.loc[active_camps,"attrs"]["advertiser_id"],
    campdf.loc[active_camps,"updates"].apply(dict,axis=1).apply(json.dumps),
),axis=1) \
    .reset_index()
updatedf.columns = ["campaign_id","account_id","update"]
updatedf["date"] = TODAY
updatedf["datetime"] = NOW
#%%
upload_taboola_updates_to_redshift(updatedf)
# %%
