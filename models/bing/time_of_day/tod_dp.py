#%%
import sys
from scipy.interpolate import UnivariateSpline
import typing
import numpy as np
import pandas as pd
import datetime
from ds_utils.db.connectors import HealthcareDW

def hc_quarter_hour_tz_adjusted(start_date, end_date, product=None, traffic_source=None):
    product_filter = "" if product is None else \
                    f"AND UPPER(s.product) = UPPER('{product}')"
    traffic_filter = "" if traffic_source is None else \
                    f"AND UPPER(traffic_source) = UPPER('{traffic_source}')"
    """
    TODO: why is conv_rate > 1?
    why not just count distinct session_id as sessions?
    """
    timeofday_query = f"""
        with
            rps as (
                SELECT
                    session_id,
                    sum(revenue) AS revenue
                FROM tron.session_revenue
                WHERE session_creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
                {traffic_filter}
                GROUP BY 1
            ),
            ip_locs as (
                SELECT 
                    l.country_iso_code,
                    l.time_zone,
                    l.subdivision_1_iso_code,
                    b.netowrk_index,
                    b.start_int,
                    b.end_int
                FROM 
                    data_science.maxmind_ipv4_geo_blocks AS b
                    JOIN data_science.maxmind_geo_locations AS l
                        ON b.maxmind_id = l.maxmind_id
            ),
            rps_tz_adj as (
                SELECT
                    s.creation_date AS utc_ts,
                    extract(HOUR FROM convert_timezone('UTC', l.time_zone, s.creation_date) - s.creation_date)::INT AS utc_offset,
                    l.time_zone,
                    convert_timezone('UTC', l.time_zone, s.creation_date) AS user_ts,
                    s.session_id,
                    l.subdivision_1_iso_code AS state,
                    r.revenue
                FROM 
                    tracking.session_detail AS s
                    JOIN ip_locs as l
                        ON ip_index(s.ip_address) = l.netowrk_index
                        AND inet_aton(s.ip_address) BETWEEN l.start_int AND l.end_int
                        AND l.country_iso_code = 'US'
                    JOIN rps as r
                        ON s.session_id = r.session_id
                WHERE nullif(s.ip_address, '') IS NOT null
                AND s.creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
                {product_filter}
            )
        SELECT
            date_part(DOW, user_ts)::INT AS dayofweek,
            date_part(HOUR, user_ts) +
            CASE 
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 0 AND 14 THEN 0.0
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 15 AND 29 THEN 0.25
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 30 AND 44 THEN 0.5
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 45 AND 59 THEN 0.75
            END AS hourofday,
            count(DISTINCT user_ts::DATE) AS days_samplesize,
            count(session_id) / count(DISTINCT user_ts::DATE) AS sessions,
            count(revenue) AS conversions,
            (count(revenue)::NUMERIC / count(session_id)::NUMERIC)::NUMERIC(5,4) AS conv_rate,
            avg(revenue)::NUMERIC(8,4) AS avg_conv_value,
            (sum(revenue) / count(DISTINCT session_id))::NUMERIC(8,4) AS rps
        FROM rps_tz_adj
        GROUP BY dayofweek,hourofday
    """
    with HealthcareDW() as db:
        df = db.to_df(timeofday_query)
    
    df = df \
        .sort_values(by=['dayofweek', 'hourofday'], ascending=True) \
        .set_index(['dayofweek', 'hourofday'])
    
    df['int_ix'] = range(len(df))

    return df

def add_spline(df, index_col, smooth_col, spline_k, spline_s, postfix='_spline'):

    df = df.copy().reset_index()
    spline = UnivariateSpline(x=df[index_col], y=df[smooth_col], k=spline_k, s=spline_s)
    df.set_index(index_col, inplace=True)
    df[smooth_col + postfix] = spline(df.index)

    return df

def spread_outliers(S,percentile=97.5) -> typing.Iterable:
    OUTTHRESH = np.percentile(S,percentile)
    OUTI = S > OUTTHRESH
    # print("outlier thresh:", OUTTHRESH)
    T = OUTI * OUTTHRESH + (1-OUTI) * S
    T = (S.sum() / T.sum()) * T
    assert abs(T.sum() - S.sum()) < 1e-10
    return T
def cma(S,window) -> typing.Iterable:
    L = S.__len__()
    CMAker = [1/window] * window
    return np.convolve([*S,*S,*S], CMAker, mode="same")[L:-L]
def ema(S, window) -> typing.Iterable:
    """
    \sum_{r^i} = s = 1 + r + r^2 + ....
    s*r = r + r^2 + r^3 + ... = s-1
    s * r = s - 1 ===> s = 1 / (1-r)
    s - 1 = 1 / (1-r) - 1 = r / (1-r)
    r \approx (window-1)/window

    ema(X,t) = (1-r)*X[t] + r*ema(X,t-1)
    """
    L = S.__len__()
    r = (window-1)/window
    EMAker = (1-r) * 1/(1-r**window) * np.array([r**i for i in range(window)])
    assert abs(EMAker.sum() - 1) < 1e-10
    return np.convolve([*S,*S,*S], EMAker, mode="same")[L:-L]
def compute_cema_tod_modifiers(df_tz_adj,window=16):
    rps = df_tz_adj["rps"]
    rps = spread_outliers(rps)

    df_tz_adj["rps_cema"] = cma(ema(rps, window), window)
    df_tz_adj["sessions_cema"] = cma(ema(df_tz_adj["sessions"], window), window)

    df_tz_adj['baseline'] = 1
    rps_cema_mean = (df_tz_adj["sessions_cema"] * df_tz_adj['rps_cema']).sum() \
        / df_tz_adj["sessions_cema"].sum()
    df_tz_adj["rps_cema_mean"] = rps_cema_mean
    df_tz_adj['cema_modifier'] = df_tz_adj['rps_cema'] / rps_cema_mean
    df_tz_adj['cema_modifier'] = df_tz_adj['cema_modifier'] * 20 // 1 / 20  # set to incs of 0.05
    
    return df_tz_adj

def wavg(V,W):
    if W.sum() == 0: return 0
    return (V*W).sum() / W.sum()
def lapprox(X,W,l,r):
    return X[l]
def midapprox(X,W,l,r):
    return X[(l+r)//2]
def wavgapprox(X,W,l,r):
    return wavg(X[l:r],W[l:r])
def interval_fit(X,W,nintervals,xapprox) -> typing.Tuple[float,typing.List[int]]:
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
        for k in reversed(range(0,nintervals+1)):
            # probe remaining time slots for first interval break
            def yield_suffix_fits():
                for r in range(l+1, N+1):
                    # interval err over l:r
                    interval_eps = W[l:r] * (X[l:r] - xapprox(X, W, l, r))**2
                    eps_suffix, int_suffix = dp[r, k+1]
                    yield interval_eps.sum() + eps_suffix, [r] + int_suffix
            dp[l, k] = min(yield_suffix_fits())
    return dp[0,0]

def interval_fit_transform(X, W, nintervals, xapprox):
    Xapprox = np.zeros(len(X))
    eps, interval_bounds,*_ = interval_fit(X,W,nintervals,xapprox)
    # assert len(interval_bounds) <= nintervals, (nintervals,interval_bounds)
    assert len({*interval_bounds}) == nintervals
    interval_bounds = [0, *interval_bounds]
    for lb, ub in zip(interval_bounds[:-1], interval_bounds[1:]):
        Xapprox[lb:ub] = xapprox(X,W,lb,ub)
    return Xapprox

def get_interval_modifier_table(product):
    df_tz_adj = hc_quarter_hour_tz_adjusted(
        start_date=start_date_ymd, end_date=end_date_ymd, product=product, traffic_source=BING)
    df_tz_adj = compute_cema_tod_modifiers(df_tz_adj,window)
    df_tz_adj = df_tz_adj.reset_index().set_index("dayofweek")
    modifier_rows = []
    for day in range(7):
        X = df_tz_adj.loc[day, "cema_modifier"].values
        W = df_tz_adj.loc[day, "sessions_cema"].values
        
        _, interval_bounds = interval_fit(X, W, BING_DAILY_INTERVALS, wavgapprox)
        interval_bounds = [0, *interval_bounds]
        interval_hr_bounds = df_tz_adj.loc[day,"hourofday"].iloc[interval_bounds[:-1]]
        interval_hr_bounds = [*interval_hr_bounds,24]
        intervals = [*zip(interval_hr_bounds[:-1],interval_hr_bounds[1:])]
        interval_modifiers = [wavgapprox(X,W,lb,ub) for lb,ub in zip(interval_bounds[:-1],interval_bounds[1:])]
        DAYS = ["SUN","MON","TUE","WED","THR","FRI","SAT"]
        modifier_rows += [{
            "product": product,
            "weekday_index": day,
            "weekday": DAYS[day], 
            "hr_start_inclusive": start_hr,
            "hr_end_exclusive": end_hr,
            "modifier": mod
            } for (start_hr,end_hr),mod in zip(intervals,interval_modifiers)]
    return pd.DataFrame(modifier_rows)
#%%
TABOOLA                 = "TABOOLA"
MEDIA_ALPHA             = "MEDIAALPHA"
BING                    = "BING"
U65                     = "HEALTH"
O65                     = 'MEDICARE'
BING_DAILY_INTERVALS    = 7

if __name__ == "__main__":
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
    #%%
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
        db.load_df(bing_modifiers,schema=SCHEMA,table=BING_TOD_MODIFIER_TABLE)

    def fit_eval_modfifiers(product):
        df_tz_adj = hc_quarter_hour_tz_adjusted(
            start_date=start_date_ymd, end_date=end_date_ymd, product=product, traffic_source=BING)
        df_tz_adj = compute_cema_tod_modifiers(df_tz_adj,window)
        df_tz_adj = df_tz_adj.reset_index().set_index("dayofweek")
        df_tz_adj["cema_modifier_interval"] = np.NaN
        for day in range(7):
            X = df_tz_adj.loc[day, "cema_modifier"].values
            W = df_tz_adj.loc[day, "sessions_cema"].values
            Xapprox = interval_fit_transform(X, W, BING_DAILY_INTERVALS, wavgapprox)
            df_tz_adj.loc[day, "cema_modifier_interval"] = Xapprox

        df_tz_adj["sessions_cema_mean_adjusted"] = \
            df_tz_adj["sessions_cema"] / df_tz_adj["sessions_cema"].mean()
        rps_mean = (df_tz_adj["rps"] * df_tz_adj["sessions"]).sum() / df_tz_adj["sessions"].sum()
        df_tz_adj["rps_mean_adjusted"] = spread_outliers(df_tz_adj["rps"]) / rps_mean
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
