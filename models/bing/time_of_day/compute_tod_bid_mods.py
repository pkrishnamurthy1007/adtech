#%%
import sys
sys.path.insert(0,"../../../")
from scipy.interpolate import UnivariateSpline
import typing
import numpy as np
import pandas as pd
import datetime
from ds_utils.db.connectors import HealthcareDW
from models.data.queries.time_of_day import hc_quarter_hour_tz_adjusted
from models.utils import *

def add_spline(df, index_col, smooth_col, spline_k, spline_s, suffix='_spline'):
    df = df.copy().reset_index()
    spline = UnivariateSpline(
        x=df[index_col], y=df[smooth_col], k=spline_k, s=spline_s)
    df.set_index(index_col, inplace=True)
    df[smooth_col + suffix] = spline(df.index)
    return df
    
def compute_cema_tod_modifiers(df_tz_adj, window=16):
    rps = df_tz_adj["rps"]
    rps = spread_outliers(rps)

    df_tz_adj["rps_cema"] = cma(ema(rps, window), window)
    df_tz_adj["sessions_cema"] = cma(
        ema(df_tz_adj["sessions"], window), window)

    df_tz_adj['baseline'] = 1
    rps_cema_mean = (df_tz_adj["sessions_cema"] * df_tz_adj['rps_cema']).sum() \
        / df_tz_adj["sessions_cema"].sum()
    df_tz_adj["rps_cema_mean"] = rps_cema_mean
    df_tz_adj['cema_modifier'] = df_tz_adj['rps_cema'] / rps_cema_mean
    df_tz_adj['cema_modifier'] = df_tz_adj['cema_modifier'] * \
        20 // 1 / 20  # set to incs of 0.05

    return df_tz_adj

def get_interval_modifier_table(product):
    df_tz_adj = hc_quarter_hour_tz_adjusted(
        start_date=start_date_ymd, end_date=end_date_ymd, product=product, traffic_source=BING)
    df_tz_adj = compute_cema_tod_modifiers(df_tz_adj, window)
    df_tz_adj = df_tz_adj.reset_index().set_index("dayofweek")
    modifier_rows = []
    for day in range(7):
        X = df_tz_adj.loc[day, "cema_modifier"].values
        W = df_tz_adj.loc[day, "sessions_cema"].values

        _, interval_bounds = interval_fit(
            X, W, BING_DAILY_INTERVALS, wavgapprox)
        interval_bounds = [0, *interval_bounds]
        interval_hr_bounds = df_tz_adj.loc[day,
                                           "hourofday"].iloc[interval_bounds[:-1]]
        interval_hr_bounds = [*interval_hr_bounds, 24]
        intervals = [*zip(interval_hr_bounds[:-1], interval_hr_bounds[1:])]
        interval_modifiers = [wavgapprox(X, W, lb, ub) for lb, ub in zip(
            interval_bounds[:-1], interval_bounds[1:])]
        DAYS = ["SUN", "MON", "TUE", "WED", "THR", "FRI", "SAT"]
        modifier_rows += [{
            "product": product,
            "weekday_index": day,
            "weekday": DAYS[day],
            "hr_start_inclusive": start_hr,
            "hr_end_exclusive": end_hr,
            "modifier": mod
        } for (start_hr, end_hr), mod in zip(intervals, interval_modifiers)]
    return pd.DataFrame(modifier_rows)


#%%
TABOOLA = "TABOOLA"
MEDIA_ALPHA = "MEDIAALPHA"
BING = "BING"
U65 = "HEALTH"
O65 = 'MEDICARE'
BING_DAILY_INTERVALS = 7

NOW = datetime.datetime.now()
DAY = datetime.timedelta(days=1)

start_date = NOW - 90*DAY
end_date = NOW - 0*DAY
window = 16

start_date_ymd = start_date.strftime("%Y%m%d")
end_date_ymd = end_date.strftime("%Y%m%d")

bing_modifiers = pd.concat((
    get_interval_modifier_table(U65),
    get_interval_modifier_table(O65)
))

bing_modifiers["calculation_date"] = NOW

SCHEMA = "data_science"
BING_TOD_MODIFIER_TABLE = "bing_tod_modifiers"
table_creation_sql = f"""
    CREATE TABLE IF NOT EXISTS 
    {SCHEMA}.{BING_TOD_MODIFIER_TABLE}
    (
        "product" VARCHAR(50),
        "weekday_index" INT,
        "weekday" VARCHAR(50),
        "hr_start_inclusive" FLOAT,
        "hr_end_exclusive" FLOAT,
        "modifier" FLOAT,
        "calculation_date" DATETIME
    );
"""

with HealthcareDW() as db:
    db.exec(table_creation_sql)
    db.load_df(bing_modifiers, schema=SCHEMA,
                table=BING_TOD_MODIFIER_TABLE)

def fit_eval_modfifiers(product):
    df_tz_adj = hc_quarter_hour_tz_adjusted(
        start_date=start_date_ymd, end_date=end_date_ymd, product=product, traffic_source=BING)
    df_tz_adj = compute_cema_tod_modifiers(df_tz_adj, window)
    df_tz_adj = df_tz_adj.reset_index().set_index("dayofweek")
    df_tz_adj["cema_modifier_interval"] = np.NaN
    for day in range(7):
        X = df_tz_adj.loc[day, "cema_modifier"].values
        W = df_tz_adj.loc[day, "sessions_cema"].values
        Xapprox = interval_fit_transform(
            X, W, BING_DAILY_INTERVALS, wavgapprox)
        df_tz_adj.loc[day, "cema_modifier_interval"] = Xapprox

    df_tz_adj["sessions_cema_mean_adjusted"] = \
        df_tz_adj["sessions_cema"] / df_tz_adj["sessions_cema"].mean()
    rps_mean = (df_tz_adj["rps"] * df_tz_adj["sessions"]
                ).sum() / df_tz_adj["sessions"].sum()
    df_tz_adj["rps_mean_adjusted"] = spread_outliers(
        df_tz_adj["rps"]) / rps_mean
    ax = df_tz_adj.reset_index().plot.scatter(x='int_ix', y='rps_mean_adjusted')
    df_tz_adj.reset_index().set_index("int_ix")[["sessions_cema_mean_adjusted", "baseline", "cema_modifier", "cema_modifier_interval"]].plot(
        ax=ax, figsize=(15, 5))
    ax.set_title(f"Bid modifiers for product={product}")

    return df_tz_adj

fit_eval_modfifiers(U65)
fit_eval_modfifiers(O65)
#%%
# #%%
# interval_fit(X,W,DAILY_INTERVALS,wavgapprox)
# df = pd.DataFrame(data=mem.values(),index=pd.MultiIndex.from_tuples(mem.keys()))
# df.loc[(slice(None),1),:]
# #%%
# for day in range(7):
#     ax = plt.gca()
#     df_tz_adj.loc[day] \
#         .reset_index().set_index("hourofday") \
#         ["cema_modifier_interval"]\
#         .plot(ax=ax, figsize=(15, 5))
# plt.show()
# #%%
# day = 5
# df_tz_adj.loc[day] \
#         .reset_index().set_index("hourofday") \
#         [["cema_modifier_interval", "cema_modifier", "sessions_cema_mean_adjusted"]]\
#         .plot(figsize=(15, 5))


#%%
