from ds_utils.db.connectors import HealthcareDW

SCHEMA = "data_science"
SCORE_TABLE = "materialized_session_scores"

def refresh_session_scores(start_date,end_date):
    score_sql = f"""
    SELECT
        r.topic,
        r.content_type,
        r.exchange,
        r.received,
        r.body.sessionid                        AS session_id,
        TO_DATE(r.body."on", 'YYYY-MM-DD')      AS computed_dt,
        r.body.jornayaid,
        r.body.score                            AS score,
        r.body.response.meta.model              AS model,
        CASE 
            WHEN model IS NULL THEN score 
            ELSE null
        END                                     AS score_null,
        CASE 
            WHEN model = 'med-adv' THEN score 
            ELSE null   
        END                                     AS score_adv,
        CASE 
            WHEN model = 'med-supp' THEN score 
            ELSE null   
        END                                     AS score_supp
    FROM dl_landing.internal_kraken_leadscore_scored AS r
    WHERE
        /* Data partitioned on date - these filters greatly speed query */
        (r.year > {start_date.year} OR 
            (r.year = {start_date.year} AND r.month >= {start_date.month})) 
        AND
        (r.year < {end_date.year} OR 
            (r.year = {end_date.year} AND r.month <= {end_date.month}))
    """
    with HealthcareDW() as db:
        db.exec(f"""
            DROP TABLE IF EXISTS {SCHEMA}.{SCORE_TABLE};
            CREATE TABLE {SCHEMA}.{SCORE_TABLE}
            AS ({score_sql})
        """)
        df = db.to_df(f"SELECT count(*) as num_records FROM {SCHEMA}.{SCORE_TABLE}")
    return df
