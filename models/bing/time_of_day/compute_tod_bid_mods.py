#%%
from models.bing.time_of_day.common import *
from scipy.interpolate import UnivariateSpline
import typing
import numpy as np
import pandas as pd
import datetime
from ds_utils.db.connectors import HealthcareDW
from models.data.queries.time_of_day import hc_session_conversions,hc_15m_user_tz
from models.utils import *

def add_spline(df, index_col, smooth_col, spline_k, spline_s, suffix='_spline'):
    df = df.copy().reset_index()
    spline = UnivariateSpline(
        x=df[index_col], y=df[smooth_col], k=spline_k, s=spline_s)
    df.set_index(index_col, inplace=True)
    df[smooth_col + suffix] = spline(df.index)
    return df

def cema_transform(y,show_plots=False,window=16):
    if show_plots:raise NotImplementedError
    y = spread_outliers(y)
    y_cema = cma(ema(y, window), window)
    return y_cema

from models.utils.time_of_day import Lowess
def lowess_transform(y, show_plots=False, frac=0.03, max_std_dev=5):
    if show_plots: raise NotImplementedError
    y = spread_outliers(y)
    y_lowess = Lowess().fit_predict(np.arange(len(y)), y.values, frac=frac, max_std_dev=max_std_dev)
    return y_lowess

def add_modifiers(df, target_field, weight_field, transform_fn, fit_kwargs={}, show_plots=False):
    if show_plots: raise NotImplementedError
    weight = df[weight_field]
    target = df[target_field]
    weight_transform = transform_fn(weight,show_plots=False,**fit_kwargs)
    target_transform = transform_fn(target,show_plots=False,**fit_kwargs)
    
    # target_wavg = (target_transform * weight_transform).sum() / weight_transform.sum()
    target_wavg = target.mean() 
    target_mod = target_transform / target_wavg
    target_mod = target_mod * 20 // 1 / 20  # set to incs of 0.05
    df["baseline"] = 1
    df[f"{weight_field}_{transform_fn.__name__}"] = weight_transform
    df[f"{target_field}_{transform_fn.__name__}"] = target_transform
    df[f"{target_field}_{transform_fn.__name__}_mean"] = target_wavg
    df[f"{target_field}_{transform_fn.__name__}_modifier"] = target_mod
    return df

def get_interval_modifier_table(df,modifier_field,weight_field,show_plots=False):
    df = df.copy()
    df = df.reset_index().set_index("dayofweek")
    modifier_rows = []
    for day in range(7):
        X = df.loc[day, modifier_field].values
        W = df.loc[day, weight_field].values

        interval_bounds,eps = \
            interval_fit(X, W, nintervals=BING_DAILY_INTERVALS, xapprox=wavgapprox)
        Xapprox = interval_transform(
            X, W, BING_DAILY_INTERVALS, wavgapprox, interval_bounds)
        df.loc[day, f"{modifier_field}_interval"] = Xapprox

        interval_bounds = [0, *interval_bounds]
        interval_hr_bounds = df.loc[day,"hourofday"].iloc[interval_bounds[:-1]]
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
    
    if show_plots:
        df[f"{weight_field}_mean_adjusted"] = df[weight_field] / df[weight_field].mean()
        # TODO: allow selecting scatter field somehow
        rps = spread_outliers(df["rps"])
        # decided not to use a weighted average to adjust rps
        # rps_mean = (rps * df[weight_field]).sum() / df[weight_field].sum()
        rps_mean = rps.mean()
        df["rps_mean_adjusted"] = rps / rps_mean
        ax = df.reset_index().plot.scatter(x='int_ix', y='rps_mean_adjusted',label="rps_mean_adjusted")
        df \
            .reset_index().set_index("int_ix") \
            [[f"{weight_field}_mean_adjusted", "baseline", modifier_field, f"{modifier_field}_interval"]]\
            .plot(ax=ax, figsize=(15, 5))
        ax.legend()
        # ax.set_title(f"Bid modifiers for product={product}")
    
    return pd.DataFrame(modifier_rows)

def upload_interval_modifier_table_to_redshift(modifier_df,product):
    SCHEMA = "data_science"
    BING_TOD_MODIFIER_TABLE = "tod_modifiers"
    table_creation_sql = f"""
        CREATE TABLE IF NOT EXISTS 
        {SCHEMA}.{BING_TOD_MODIFIER_TABLE}
        (
            "product" VARCHAR(50),
            "traffic_source" VARCHAR(50),
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
        db.load_df(modifier_df, schema=SCHEMA,
                table=BING_TOD_MODIFIER_TABLE)

def upload_interval_modifier_table_to_s3(df,product):
    product_dir = f"{OUTPUT_DIR}/{product}"
    os.makedirs(product_dir, exist_ok=True)
    bids_fnm = f"BIDS_{TODAY}.csv"
    bids_fpth = f"{OUTPUT_DIR}/{product}/{bids_fnm}"
    df.to_csv(bids_fpth, index=False, encoding='utf-8')

    #### WRITE OUTPUT TO S3 ####
    s3_resource = boto3.resource('s3')

    s3_client = boto3.client('s3')
    response = s3_client.upload_file(
        bids_fpth,
        S3_OUTPUT_BUCKET,
        f"{S3_OUTPUT_PREFIX}/{product}/{bids_fnm}")
#%%
TABOOLA = "TABOOLA"
MEDIA_ALPHA = "MEDIAALPHA"
BING = "BING"
U65 = "HEALTH"
O65 = 'MEDICARE'
BING_DAILY_INTERVALS = 7

NOW = datetime.datetime.now()
TODAY = NOW.date()
DAY = datetime.timedelta(days=1)

MONTHS = 5
start_date = NOW - MONTHS*30*DAY
end_date = NOW - 0*DAY
window = 16

start_date_ymd = start_date.strftime("%Y%m%d")
end_date_ymd = end_date.strftime("%Y%m%d")

traffic_source = BING
for product in [U65,O65]:
    df_15m = hc_15m_user_tz(
        start_date=start_date_ymd, end_date=end_date_ymd,
        product=product, traffic_source=traffic_source)

    # # CEMA
    # df_15m = add_modifiers(df_15m,"rps","sessions",cema_transform)
    # modifiers_df = get_interval_modifier_table(df_15m, "rps_cema_transform_modifier", "sessions_cema_transform",show_plots=True)
    # LOWESS <- hits peaks a little better
    df_15m = add_modifiers(df_15m,"rps","sessions",lowess_transform)
    modifier_df = get_interval_modifier_table(df_15m, "rps_lowess_transform_modifier", "sessions_lowess_transform",show_plots=True)
    modifier_df["calculation_date"] = NOW
    modifier_df["product"] = product
    modifier_df["traffic_source"] = traffic_source

    upload_interval_modifier_table_to_s3(modifier_df, product)
    upload_interval_modifier_table_to_redshift(modifier_df,product)
#%%
# TODO: automate regression checks
# ax = df_15m.plot.scatter(x="int_ix",y="sessions")
# df_15m[["sessions_lowess_transform"]].plot(ax=ax,figsize=(15,5))
# %%
