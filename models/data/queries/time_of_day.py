from ds_utils.db.connectors import HealthcareDW
import functools

@functools.lru_cache()
def hc_session_conversions(start_date, end_date, product=None, traffic_source=None):

    product_filter = "" if product is None else f"AND s.product = '{product}'"
    traffic_filter = "" if traffic_source is None else f"AND traffic_source = '{traffic_source}'"

    timeofday_query = f"""
        SELECT
            session_id,
            utc_ts,
            utc_offset,
            user_ts,
            state,
            revenue,
            CASE 
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 0 AND 14 THEN 0.0
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 15 AND 29 THEN 0.25
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 30 AND 44 THEN 0.5
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 45 AND 59 THEN 0.75
            END AS hourfraction,
            date_part(HOUR, user_ts)::INT as hourofday,
            date_part(DOW, user_ts)::INT as weekday
        FROM (
            SELECT
                s.creation_date AS utc_ts,
                extract(HOUR FROM convert_timezone('UTC', l.time_zone, s.creation_date) - s.creation_date)::INT AS utc_offset,
                l.time_zone,
                convert_timezone('UTC', l.time_zone, s.creation_date) AS user_ts,
                s.session_id,
                l.subdivision_1_iso_code AS state,
                r.revenue
            FROM tracking.session_detail AS s
            INNER JOIN data_science.maxmind_ipv4_geo_blocks AS b
                ON ip_index(s.ip_address) = b.netowrk_index
                AND inet_aton(s.ip_address) BETWEEN b.start_int AND b.end_int
            INNER JOIN data_science.maxmind_geo_locations AS l
                ON b.maxmind_id = l.maxmind_id
            INNER JOIN(
                SELECT
                    session_id,
                    sum(revenue) AS revenue
                 FROM tron.session_revenue
                 WHERE session_creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
                 {traffic_filter}
                 GROUP BY 1
            ) AS r
                ON s.session_id = r.session_id
            WHERE nullif(s.ip_address, '') IS NOT null
            AND s.creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
            AND l.country_iso_code = 'US'
            {product_filter}
        ) AS sub
        ;
    """

    with HealthcareDW() as db_context:
        df = db_context.to_df(timeofday_query)

    return df

@functools.lru_cache()
def hc_15m_user_tz(start_date, end_date, product=None, traffic_source=None):
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
                    INNER JOIN rps as r
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
            count(session_id) as sessions,
            sum((revenue>0)::INT::FLOAT) as conversions,
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

"""
========================= BAG KPI QUERY =========================
"""
@functools.lru_cache()
def hc_quarter_hour_tz_adjusted_bag(start_date, end_date, product=None, traffic_source=None):
    product_filter = "" if product is None else \
        f"AND UPPER(s.product) = UPPER('{product}')"
    traffic_filter = "" if traffic_source is None else \
        f"AND UPPER(traffic_source) = UPPER('{traffic_source}')"
    bag_kpis_by_tod_sql = f"""
        with
            rps as (
                SELECT
                    session_id,
                    sum(revenue) AS revenue
                FROM tron.session_revenue s
                WHERE session_creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
                {traffic_filter}
                GROUP BY 1
            ),
            session_rps as (
                SELECT
                    s.*,
                    s.creation_date     AS creation_ts_utc,
                    r.revenue
                FROM
                    tracking.session_detail AS s
                    INNER JOIN rps as r
                        ON s.session_id = r.session_id
                WHERE s.creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
                {product_filter}
                {traffic_filter}
            ),
            session_cum_conv as (
                SELECT
                    *,
                    SUM((revenue>0)::INT) OVER (
                        ORDER BY creation_ts_utc
                        ROWS BETWEEN UNBOUNDED PRECEDING AND
                                        CURRENT ROW)    as conversions_cum
                FROM session_rps        
            ), 
            bag_rps as (
                SELECT
                    *,
                    conversions_cum                                                     as bag_id,
                    COUNT(*) OVER (PARTITION BY conversions_cum)                        as bag_len,        
                    SUM((revenue>0)::INT) OVER (PARTITION BY conversions_cum)           as bag_conv,
                    AVG((revenue>0)::INT::FLOAT) OVER (PARTITION BY conversions_cum)    as bag_lpc,
                    SUM(revenue) OVER (PARTITION BY conversions_cum)                    as bag_rpl,
                    AVG(revenue) OVER (PARTITION BY conversions_cum)                    as bag_rpc
                FROM session_cum_conv
            )
        SELECT
            creation_ts_utc::date                   as date,
            date_part(HOUR, creation_ts_utc) +
            CASE 
                WHEN date_part(MINUTE, creation_ts_utc)::INT BETWEEN 0 AND 14 THEN 0.0
                WHEN date_part(MINUTE, creation_ts_utc)::INT BETWEEN 15 AND 29 THEN 0.25
                WHEN date_part(MINUTE, creation_ts_utc)::INT BETWEEN 30 AND 44 THEN 0.5
                WHEN date_part(MINUTE, creation_ts_utc)::INT BETWEEN 45 AND 59 THEN 0.75
            END                                     as hour,
            COUNT(*)                                as clicks_num,
            SUM(bag_lpc)                            as leads_num,
            SUM(bag_rpc)                            as rev_sum,
            AVG(bag_rpc)                            as rev_avg        
        FROM bag_rps
        GROUP BY 
            date,hour
    """
    """ 
    NOTE:
        our version is PG 8 - range queries w/ definite bounds not supported 
        until PG 11
    SUM((r.revenue>0)::INT) OVER (
                ORDER BY r.creation_ts_utc 
                RANGE BETWEEN '10 days'::INTERVAL PRECEDING AND
                                CURRENT ROW)    as conversions_past_day
    """
    with HealthcareDW() as db:
        df = db.to_df(bag_kpis_by_tod_sql)

    df = df \
        .sort_values(by=['date', 'hour'], ascending=True) \
        .set_index(['date', 'hour'])

    df['int_ix'] = range(len(df))

    return df
