#%%
from models.taboola.common import *
from models.data.queries.time_of_day import hc_15m_user_tz
from models.utils.time_of_day import add_modifiers, cema_transform, spread_outliers, lowess_transform
from models.utils import *

MONTHS = 6
start_date = NOW - MONTHS*30*DAY
end_date = NOW - 0*DAY
traffic_source = TABOOLA

df_15m = hc_15m_user_tz(
    start_date=start_date, end_date=end_date,
    product=None, traffic_source=traffic_source).copy()
#%%
df_1hr = df_15m.copy().reset_index()
df_1hr["hourofday"] = (df_1hr['hourofday'] // 1).values
df_1hr["revenue"] = df_1hr["rps"] * df_1hr["sessions"]
df_1hr = df_1hr \
    .groupby(["dayofweek","hourofday"]) \
    [["days_samplesize","sessions","conversions","revenue"]] .sum()
df_1hr["int_ix"] = range(df_1hr.__len__())
df_1hr["baseline"] = 1
df_1hr["rps"] = df_1hr["revenue"] / df_1hr["sessions"]

def f(x,w,show_plots,window=4):
    # w = np.ones(len(x))
    return cema_transform(x,w,show_plots,window)
f.__name__ = "cema_transform"
df_1hr = add_modifiers(df_1hr,"rps","sessions",f)
df_1hr = add_modifiers(df_1hr,"rps","sessions",lowess_transform)
df_1hr[["rps","rps_lowess_transform","rps_cema_transform"]].plot()
df_1hr[["sessions", "sessions_lowess_transform", "sessions_cema_transform"]].plot()
df_1hr[["rps_lowess_transform_modifier", "rps_cema_transform_modifier"]].plot()
#%%
df_15m = add_modifiers(df_15m,"rps","sessions",cema_transform)
df_15m = add_modifiers(df_15m,"rps","sessions",lowess_transform)
df_15m[["rps", "rps_lowess_transform", "rps_cema_transform"]].plot()
df_15m[["sessions", "sessions_lowess_transform", "sessions_cema_transform"]].plot()
df_15m[["rps_lowess_transform_modifier", "rps_cema_transform_modifier"]].plot()
#%%
schedule = df_1hr \
    .reset_index() \
    [['dayofweek','hourofday',"rps_cema_transform_modifier"]] \
    .rename(columns={
        'dayofweek': "day",
        'hourofday': "hour",
        "rps_cema_transform_modifier": "bid_modifier",
    })
schedule = {c: V.tolist() for c,V in schedule.items()}
#%%
updatedf = pd.DataFrame(
    [[TODAY,NOW,json.dumps(schedule)]],
    columns=["date","datetime","schedule"])
upload_taboola_updates_to_redshift(updatedf)
# %%
