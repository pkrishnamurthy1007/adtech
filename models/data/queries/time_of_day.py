from ds_utils.db.connectors import HealthcareDW

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
                HAVING sum(revenue) > 0.0
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
                    LEFT JOIN rps as r
                        ON s.session_id = r.session_id
                WHERE nullif(s.ip_address, '') IS NOT null
                AND s.creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
                {product_filter}
            )
        SELECT
            user_ts,
            utc_ts,
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
