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
    print("outlier thresh:", OUTTHRESH)
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
#%%

NOW = datetime.datetime.now()
DAY = datetime.timedelta(days=1)

start_date = NOW - 90*DAY
end_date = NOW - 0*DAY

start_date_ymd = start_date.strftime("%Y%m%d")
end_date_ymd = end_date.strftime("%Y%m%d")
product = None
traffic_source = None
start_date_ymd,end_date_ymd

TABOOLA     = "TABOOLA"
MEDIA_ALPHA = "MEDIAALPHA"
BING        = "BING"
U65         = "HEALTH"
O65         = 'MEDICARE'

df_tz_adj = hc_quarter_hour_tz_adjusted(
    start_date=start_date_ymd, end_date=end_date_ymd, product=O65, traffic_source=TABOOLA)
df_tz_adj["rps_raw"] = df_tz_adj["rps"]

rps = df_tz_adj["rps_raw"]
rps = spread_outliers(rps)

window = 16
df_tz_adj["rps_cma"] = cma(rps,window)
df_tz_adj["rps_ema"] = ema(rps,window)
df_tz_adj["rps_cema"] = cma(ema(rps, window), window)
df_tz_adj["sessions_cema"] = cma(ema(df_tz_adj["sessions"], window), window)

df_tz_adj[["rps_cma", "rps_ema", "rps_cema"]].plot(figsize=(15, 5))
# s1,s2 = df_tz_adj[["rps_raw","rps"]].sum()
# assert abs(s1 - s2) < 1e-10

# rpsdf = pd.DataFrame.multiply(
#     df_tz_adj[["rps", "rps_raw", "rps_cma","rps_ema"]], df_tz_adj["sessions"],
#     axis=0)
# rpsdf.sum()

# rev_cema_sum = (df_tz_adj["rps_cema"] * df_tz_adj["sessions_cema"]).sum()
# rev_sum = (df_tz_adj["rps"] * df_tz_adj["sessions"]).sum()
# rev_cema_sum,rev_sum

RPS_SPLINE_K = 3
RPS_SPLINE_S = 45

df_tz_adj["rps"] = spread_outliers(df_tz_adj["rps_raw"],97.5)
df_tz_adj = add_spline(df_tz_adj, index_col='int_ix',
                       smooth_col='rps', spline_k=RPS_SPLINE_K, spline_s=RPS_SPLINE_S)

ax = df_tz_adj[['rps']].reset_index().plot.scatter(x='int_ix', y='rps')
df_tz_adj[['rps_spline',"rps_cema"]].plot(ax=ax, figsize=(15, 5), colormap='Dark2')

SESSIONS_SPLINE_K = 3
SESSIONS_SPLINE_S = 10 * 1000

df_tz_adj = add_spline(df_tz_adj, index_col='int_ix', smooth_col='sessions',
                       spline_k=SESSIONS_SPLINE_K, spline_s=SESSIONS_SPLINE_S)

ax = df_tz_adj[['sessions']].reset_index().plot.scatter(x='int_ix', y='sessions')
df_tz_adj[['sessions_spline',"sessions_cema"]].plot(ax=ax, figsize=(15, 5), colormap='spring')

rps_mean = (df_tz_adj['sessions'] * df_tz_adj['rps']).sum() / df_tz_adj['sessions'].sum()
df_tz_adj["rps_mean_adjusted"] = df_tz_adj["rps"] / rps_mean
df_tz_adj['baseline'] = 1
rps_spline_mean = (df_tz_adj['sessions_spline'] * df_tz_adj['rps_spline']).sum() \
                                / df_tz_adj['sessions_spline'].sum()
df_tz_adj["rps_spline_mean"] = rps_spline_mean
df_tz_adj['spline_modifier'] = df_tz_adj['rps_spline'] / rps_spline_mean
df_tz_adj['spline_modifier'] = df_tz_adj['spline_modifier'] * 20 // 1 / 20 # set to incs of 0.05
rps_cema_mean = (df_tz_adj['sessions_spline'] * df_tz_adj['rps_spline']).sum() \
    / df_tz_adj['sessions_spline'].sum()
df_tz_adj["rps_cema_mean"] = rps_cema_mean
df_tz_adj['cema_modifier'] = df_tz_adj['rps_cema'] / rps_cema_mean
df_tz_adj['cema_modifier'] = df_tz_adj['cema_modifier'] * 20 // 1 / 20  # set to incs of 0.05
ax = df_tz_adj.reset_index().plot.scatter(x='int_ix', y='rps_mean_adjusted')
df_tz_adj[["baseline", "spline_modifier","cema_modifier"]].plot(ax=ax,figsize=(15, 5))
#%%
"""
NOTE: can help to fit to lps * rpl = rps
- convrate noisy
- demoniator of conversoin rate less noisy than quotient
- can estimate conversion rate at a time interval from session rate
- how diff then finding conversion rate independent of TOD? - then modeling sessoins per TOD
    - s s c . ss | s c . sss c . sss | ss c . ss sss c 


15          30      
"""

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

DAILY_INTERVALS = 7
df_tz_adj = df_tz_adj.reset_index().set_index("dayofweek")
df_tz_adj["cema_modifier_interval"] = np.NaN
for day in range(7):
    X = df_tz_adj.loc[day,"cema_modifier"].values
    W = df_tz_adj.loc[day,"sessions_cema"].values
    Xapprox = interval_fit_transform(X, W, DAILY_INTERVALS, wavgapprox)
    df_tz_adj.loc[day, "cema_modifier_interval"] = Xapprox

df_tz_adj["sessions_cema_mean_adjusted"] = \
    df_tz_adj["sessions_cema"] / df_tz_adj["sessions_cema"].mean()
ax = df_tz_adj.reset_index().plot.scatter(x='int_ix', y='rps_mean_adjusted')
df_tz_adj.reset_index().set_index("int_ix")\
    [["sessions_cema_mean_adjusted","baseline", "cema_modifier", "cema_modifier_interval"]].plot(
    ax=ax, figsize=(15, 5))

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
#%%
1/0
df_tz_adj_global = hc_quarter_hour_tz_adjusted(
    start_date=start_date_ymd, end_date=end_date_ymd, product=O65)
df_tz_adj_global["rps_raw"] = df_tz_adj_global["rps"]
#%%
rps = df_tz_adj_global["rps_raw"]
rps = spread_outliers(rps)

window = 16
df_tz_adj_global["rps_cma"] = cma(rps, window)
df_tz_adj_global["rps_ema"] = ema(rps, window)
df_tz_adj_global["rps_cema"] = cma(ema(rps, window), window)
df_tz_adj_global[["rps_cma", "rps_ema", "rps_cema"]].plot(figsize=(15, 5))
#%%
import pandas as pd
rpsdf = pd.DataFrame.multiply(
    df_tz_adj_global[["rps", "rps_raw", "rps_cma",
                      "rps_ema"]], df_tz_adj_global["sessions"],
    axis=0)
rpsdf.sum()

# %%
