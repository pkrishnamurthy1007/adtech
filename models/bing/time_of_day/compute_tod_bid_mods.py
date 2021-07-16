#%%
from models.bing.time_of_day.common import *
import typing
import numpy as np
import pandas as pd
import datetime
from ds_utils.db.connectors import HealthcareDW
from models.data.queries.time_of_day import hc_15m_user_tz
from models.utils.time_of_day import add_modifiers,cema_transform,spread_outliers,lowess_transform
from models.utils import wavg

def lapprox(X, W, l, r):
    return X[l]
def midapprox(X, W, l, r):
    return X[(l+r)//2]
def wavgapprox(X, W, l, r):
    return wavg(X[l:r], W[l:r])
def interval_fit(X, W, nintervals, xapprox) -> typing.Tuple[typing.List[int],float]:
    """
    PREMISE:
        define subset of X,W w/ leftmost bound of l
        we then say there must be a unique minimum interval split for k remaining intervals

        then we test the end pt for this interval for every remaining index from l to N
    """
    assert len(X) == len(W)
    N = len(X)
    # dp matrix of size (N+1),(nintervals+1) representing fit err and interval splits
    #   for subsets starting at time index `r` and w/ `c` intervels left to allocate
    dp = np.empty((N+1, nintervals+2, 2)).astype(object)
    # l >= len(X|W): all indices assigned to interval - terminate w / 0 MSE
    dp[N, :] = 0, []
    # k > nintervals: k represetns # of intervals allocated - so if k > nintervals
    #                 we have used too many intervals - terminate w / `inf` MSE
    dp[:, -1] = float('inf'), []
    for l in reversed(range(N)):
        for k in reversed(range(0, nintervals+1)):
            # probe remaining time slots for first interval break
            def yield_suffix_fits():
                for r in range(l+1, N+1):
                    # interval err over l:r
                    interval_eps = W[l:r] * (X[l:r] - xapprox(X, W, l, r))**2
                    eps_suffix, int_suffix = dp[r, k+1]
                    yield interval_eps.sum() + eps_suffix, [r] + int_suffix
            dp[l, k] = min(yield_suffix_fits())
    eps,interval_bounds = dp[0, 0]
    return interval_bounds,eps

def interval_transform(X,W,nintervals,xapprox,interval_bounds,*_):
    assert len({*interval_bounds}) == nintervals
    interval_bounds = [0, *interval_bounds]
    Xapprox = np.zeros(len(X))
    for lb, ub in zip(interval_bounds[:-1], interval_bounds[1:]):
        Xapprox[lb:ub] = xapprox(X, W, lb, ub)
    return Xapprox

def interval_fit_transform(X, W, nintervals, xapprox):
    return interval_transform(X,W,nintervals,xapprox,*interval_fit(X, W, nintervals, xapprox))

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

start_date_ymd = start_date.strftime("%Y%m%d")
end_date_ymd = end_date.strftime("%Y%m%d")

traffic_source = BING
for product in [U65,O65,None]:
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