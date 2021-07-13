#%%
from pkg_resources import resource_filename as rscfn
import os
from ds_utils.db.connectors import HealthcareDW,PivotDW
TABLE_DUMPS = rscfn(__name__,"TABLE_DUMPS")
os.makedirs(TABLE_DUMPS,exist_ok=True)
def fetch_head(rsc,limit=1000,src=HealthcareDW):
    with src() as db:
        db \
            .to_df(f"select * from {rsc} limit {limit}") \
            .to_csv(f"{TABLE_DUMPS}/{src.__name__}.{rsc}.csv")
    return db
#%%
import functools
import datetime
from models.data.queries.lead_score import \
    SCHEMA as DS_SCHEMA, \
    SCORE_TABLE
def get_unified_session_sql(start_date,end_date,product,traffic_source):
    start_date = start_date.date() if isinstance(start_date,datetime.datetime) else start_date
    end_date = end_date.date() if isinstance(end_date,datetime.datetime) else end_date
    product_filter = "" if product is None else \
        f"AND UPPER(s.product) = UPPER('{product}')"
    
    if traffic_source is None:
        traffic_filter = ""
    elif isinstance(traffic_source,(list,tuple,set)):
#         traffic_source_str = ",".join(f"'{src.upper()}'" for src in traffic_source)
        traffic_filter = f"AND UPPER(s.traffic_source) in {tuple(traffic_source)}"
    else:
        traffic_filter = \
            f"AND UPPER(s.traffic_source) = '{traffic_source.upper()}'"
        
    session_revenue_sql = f"""
    SELECT
        session_id,
        sum(revenue) AS revenue
    FROM tron.session_revenue s
    WHERE session_creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
        {product_filter}
        {traffic_filter}
    GROUP BY 1
    """
    geoip_sql = f"""
    SELECT 
        l.*,
        b.netowrk_index,
        b.start_int,
        b.end_int
    FROM 
        data_science.maxmind_ipv4_geo_blocks AS b
        JOIN data_science.maxmind_geo_locations AS l
            ON b.maxmind_id = l.maxmind_id
    """
    score_pivot_sql = f"""
        SELECT  
            session_id,
            computed_dt,
            MAX(score_null) AS score_null,
            MAX(score_adv) AS score_adv,
            MAX(score_supp) AS score_supp
        FROM {DS_SCHEMA}.{SCORE_TABLE} 
        WHERE 
            '{start_date}'::DATE <= computed_dt AND computed_dt < '{end_date}'::DATE 
        GROUP BY
            session_id,computed_dt
    """
    session_score_sql = f"""
        SELECT 
            *
        FROM (
            SELECT  
                *,
                ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY computed_dt DESC) as compute_order
            FROM ({score_pivot_sql})
        )
        WHERE compute_order = 1
    """
    unified_session_sql = f"""
    WITH
        rps AS ({session_revenue_sql}),
        ip_locs AS ({geoip_sql}),
        lead_scores AS ({session_score_sql})
    SELECT
        UPPER(s.traffic_source) as traffic_source,
        s.browser,
        s.operating_system,
        s.device,
        s.channel,
        s.domain,
        s.product,
        s.keyword,
        s.campaign_id,
        s.landing_page,
        s.session_id,
        s.creation_date::DATE                                   AS utc_dt,
        s.creation_date                                         AS utc_ts,
        extract(
            HOUR FROM
            convert_timezone('UTC', l.time_zone, s.creation_date) 
                - s.creation_date
        )::INT                                                  AS utc_offset,
        l.time_zone,
        convert_timezone('UTC', l.time_zone, s.creation_date)   AS user_ts,
        date_part(DOW, user_ts)::INT                            AS dayofweek,
        date_part(HOUR, user_ts) +
        CASE 
            WHEN date_part(MINUTE, user_ts)::INT BETWEEN 0 AND 14 THEN 0.0
            WHEN date_part(MINUTE, user_ts)::INT BETWEEN 15 AND 29 THEN 0.25
            WHEN date_part(MINUTE, user_ts)::INT BETWEEN 30 AND 44 THEN 0.5
            WHEN date_part(MINUTE, user_ts)::INT BETWEEN 45 AND 59 THEN 0.75
        END                                                     AS hourofday,
        l.subdivision_1_iso_code                                AS state,
        l.metro_code                                            AS dma,
        r.revenue,
        scores.computed_dt                                      AS score_compute_dt,
        scores.score_null,
        scores.score_adv,
        scores.score_supp,
        (random() * 4)::INT                                     AS random_split_4,
        (random() * 8)::INT                                     AS random_split_8,
        (random() * 16)::INT                                    AS random_split_16,
        (random() * 32)::INT                                    AS random_split_32,
        (random() * 64)::INT                                    AS random_split_64,
        (random() * 128)::INT                                   AS random_split_128,
        (4.0   / 16 * (
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()))::INT      AS random_normal_split_4,
        (8.0   / 16 * (
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()))::INT      AS random_normal_split_8,
        (16.0  / 16 * (
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()))::INT      AS random_normal_split_16,
        (32.0  / 16 * (
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()))::INT      AS random_normal_split_32,
        (64.0  / 16 * (
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()))::INT      AS random_normal_split_64,
        (128.0 / 16 * (
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()+
                random()+random()+random()+random()))::INT      AS random_normal_split_128
FROM 
        tracking.session_detail AS s
        JOIN ip_locs as l
            ON ip_index(s.ip_address) = l.netowrk_index
            AND inet_aton(s.ip_address) BETWEEN l.start_int AND l.end_int
            AND l.country_iso_code = 'US'
        INNER JOIN rps as r
            ON s.session_id = r.session_id
        LEFT JOIN lead_scores as scores
            ON s.session_id = scores.session_id
    WHERE nullif(s.ip_address, '') IS NOT null
        AND nullif(dma,'') IS NOT NULL 
        AND s.creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
        {product_filter}
        {traffic_filter}
    """
    return unified_session_sql

@functools.lru_cache()
def unified_session(start_date,end_date,product,traffic_source):
    unified_session_sql = get_unified_session_sql(start_date,end_date,product,traffic_source)
    
    with HealthcareDW() as db:
        session_df = db.to_df(unified_session_sql)
    session_df["sessions"] = 1
    session_df["num_leads"] = session_df["revenue"] > 0
    return session_df
    
@functools.lru_cache()
def agg_rps(start_date,end_date,product,traffic_source,agg_columns):

    # from IPython.display import display as ipydisp
    # score_computed_I = ~session_rps_df["score_compute_dt"].isna()
    # session_rps_df["score_computed"] = score_computed_I
    # session_rps_df["rev>0"] = session_rps_df["revenue"] > 0
    # session_rps_df["cnt"] = 1
    # session_rps_df.groupby(["score_computed","rev>0"])[["cnt"]].count()

    agg_columns = [*agg_columns]
    unified_session_sql = get_unified_session_sql(start_date,end_date,product,traffic_source)
    agg_rps_query = f"""
    SELECT
        {','.join(agg_columns)},
        COUNT(session_id)                                                       AS sessions,
        SUM(revenue)                                                            AS revenue,
        SUM((revenue>0)::INT::FLOAT)                                            AS num_leads,
        AVG((revenue>0)::INT::FLOAT)                                            AS lps_avg,
        SUM(revenue) / CASE
            WHEN num_leads = 0 THEN 1
            ELSE num_leads
        END                                                                     AS rpl_avg,
        (SUM(revenue) / COUNT(DISTINCT session_id))::NUMERIC(8,4)               AS rps_,
        AVG(revenue)                                                            AS rps_avg,
        STDDEV(revenue)                                                         AS rps_std,
        VARIANCE(revenue)                                                       AS rps_var,
        SUM((score_null>0)::INT)                                                AS score_null_cnt,
        AVG(score_null)                                                         AS score_null_avg,
        SUM((score_adv>0)::INT)                                                 AS score_adv_cnt,
        AVG(score_adv)                                                          AS score_adv_avg,
        SUM((score_supp>0)::INT)                                                AS score_supp_cnt,
        AVG(score_supp)                                                         AS score_supp_avg,
        SUM(((score_null>0) OR (score_adv>0) OR (score_supp>0))::INT)           AS score_cnt
    FROM ({unified_session_sql})
    GROUP BY {','.join(agg_columns)}
    """
    # print(agg_rps_query)
    # print(traffic_filter)
    from ds_utils.db.connectors import HealthcareDW
    with HealthcareDW() as db:
        df = db.to_df(agg_rps_query)
    globals()["df"] = df

    delt = df["rps_avg"] - df['rps_']
    if not all(delt.abs() < 1e-3):
        print("session uniqueness assummption not satisfied")
    df = df \
        .sort_values(by=agg_columns, ascending=True) \
        .set_index(agg_columns)

    df['int_ix'] = range(len(df))

    return df
#%%
from models.utils import wavg,get_wavg_by
TABOOLA_DESK    = 'DESK'
TABOOLA_PHON    = 'PHON'
TABOOLA_TBLT    = 'TBLT'
TABOOLA_LINUX   = "Linux"
TABOOLA_MACOS   = 'Mac OS X'
TABOOLA_WINOS   = "Windows"
taboola_val_map = {
    "device": {
        'DESKTOP':  TABOOLA_DESK,
        'MOBILE':   TABOOLA_PHON,
        'TABLET':   TABOOLA_TBLT,
        "D":        TABOOLA_DESK,
        "P":        TABOOLA_PHON,
        "T":        TABOOLA_TBLT,            
    },
    "operating_system": {
        "Linux": TABOOLA_LINUX,
        'Linux armv7l': TABOOLA_LINUX,
        'Linux armv8l': TABOOLA_LINUX,
        'Linux x86_64': TABOOLA_LINUX,
        "ARM": TABOOLA_LINUX,
        "FreeBSD amd64": TABOOLA_LINUX,
        'Linux aarch64': TABOOLA_LINUX,
        'Linux armv7': TABOOLA_LINUX,
        'Linux i686': TABOOLA_LINUX,
        
        'MacIntel': TABOOLA_MACOS,
        
        'Win32': TABOOLA_WINOS,
        'Win64': TABOOLA_WINOS,
        'Windows': TABOOLA_WINOS,

        'iPad': "iPadOS",
        'iPhone': "iOS",
        "iPod touch": "iOS",
        'Android': 'Android',

        '': None,
    }
}

def translate_taboola_vals(df):
    index_cols = df.index.names
    df = df.reset_index()
    for c in df.columns:
        if c in taboola_val_map:
            df[c] = df[c] \
                .map(taboola_val_map[c]) \
                .combine_first(df[c])
    df = df.set_index(index_cols)
    avgC = ["lps_avg","rpl_avg","rps_avg", 
            "score_null_avg", "score_adv_avg", "score_supp_avg",]
    df_translated = df \
        .groupby(index_cols) \
        [["revenue","sessions","num_leads"]].sum()
    df_translated[avgC] = \
        (df[avgC] * df[["sessions"]].values) .groupby(index_cols) .sum() \
            / df.groupby(index_cols)[["sessions"]].sum().values
    df_wavg = wavg(df[avgC],df["sessions"])
    df_translated_avg = wavg(df_translated[avgC],df_translated["sessions"])
    assert all((df_translated_avg - df_wavg).abs() < 1e-2), (df_translated_avg,df_wavg)
    return df_translated
#%%
GOOGLE = "GOOGLE"
BING = "BING"
PERFORMMEDIA = "PERFORMMEDIA"
MEDIAALPHA = "MEDIAALPHA"
FACEBOOK = "FACEBOOK"
SUREHITS = "SUREHITS"
TABOOLA = "TABOOLA"
YAHOO = "YAHOO"
MAJOR_TRAFFIC_SOURCES = [
    GOOGLE,BING,PERFORMMEDIA,MEDIAALPHA,FACEBOOK,
#     SUREHITS,
    TABOOLA,
#     YAHOO,
]

U65 = "HEALTH"
O65 = 'MEDICARE'
PRODUCTS = [U65,O65]
TABOOLA_OS_DOMAIN = ['Mac_OS_X', 'Linux', 'Windows', 'iOS', 'iPadOS', 'Android']
#%%
def rps_df_postprocess(rps_df):
    rps_df["leads"] = rps_df["num_leads"].fillna(0)
    rps_df["lps"] = rps_df["leads"] / rps_df["sessions"]
    rps_df["rpl"] = rps_df["revenue"] / rps_df["leads"]
    rps_df["score"] = rps_df[["score_null_avg",
                            "score_adv_avg", "score_supp_avg"]].sum(axis=1)
    rps_df["rps"] = rps_df["rps_avg"]
    rps_df["rps_"] = rps_df["revenue"] / rps_df["sessions"]
    delta = rps_df["rps"] - rps_df["rps_"]
    assert delta.abs().max() < 1e-10
    assert abs(rps_df["revenue"].sum() / rps_df["sessions"].sum() -
            wavg(rps_df["rps"], rps_df["sessions"])) < 1e-10

    return rps_df