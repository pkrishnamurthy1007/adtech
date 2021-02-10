from ds_utils.db.connectors import PivotDW


schema = 'tracking'
table = 'app_premium'

"""
app_premium
app_members
app_ltv
"""

def view_create():

    create = f"""
        CREATE MATERIALIZED VIEW {schema}.{table} AS
        WITH date_limit AS (SELECT max(datetime_created::DATE) AS maxdate FROM ph_transactional.application),
        app_base AS (
            SELECT
                app_pid,
                coalesce(foreign_pid, app_pid::VARCHAR) AS foreign_pid,
                any_value(carrier_id) AS carrier_id,
                any_value(product_type) AS product_type,
                min(datetime_created) AS datetime_created,
                max(datetime_modified) AS datetime_modified,
                min(effective_date) AS effective_date,
                least(min(date_cancelled), min(termination_date)) AS termination_date,
                max(expiration_date) AS expiration_date,
                sum(coverage_duration) AS duration_sold,
                sum(CASE 
                    WHEN effective_date <= maxdate - 1 
                    AND least(date_cancelled, termination_date, expiration_date) > effective_date
                    THEN least(maxdate - 1, date_cancelled, termination_date, expiration_date) -  effective_date + 1
                    ELSE null END) AS duration_utilized,
                max(max_months_rated_on) AS max_months_rated_on,
                max(sequence_max) AS sequence_max,
                sum(premium_amount * coverage_duration) / sum(coverage_duration) AS premium_health,
                nullif(sum(billing_fee * coverage_duration) / sum(coverage_duration), 0.0) AS billing_fee_health,
                nullif(sum(association_fee * coverage_duration) / sum(coverage_duration), 0.0) AS association_fee_health,
                nullif(sum(admin_fee * coverage_duration) / sum(coverage_duration), 0.0) AS admin_fee_health,
                nullif(max(application_fee), 0.0) AS enrollment_fee_health,
                nullif(sum(extras_amount * coverage_duration) / sum(coverage_duration), 0.0) AS premium_rider
            FROM ph_transactional.application
            CROSS JOIN date_limit
            WHERE agency_id NOT ILIKE '%test%'
                AND upper(status) IN ('NEW', 'PENDING','IN_PROGRESS', 'PROCESSED', 'CANCELLED')
                AND upper(product_type) IN ('BTM', 'STM')
            GROUP BY 1,2
            ),
        app_addon AS (
            SELECT
                app_pid,
                coalesce(foreign_pid, app_pid::VARCHAR) AS foreign_pid,
                count(DISTINCT foreign_pid) AS fpids,
                any_value(product_type) AS product_type,
                any_value(carrier_id) AS carrier_id,
                min(datetime_created) AS datetime_created,
                max(datetime_modified) AS datetime_modified,
                min(effective_date) AS effective_date,
                least(min(date_cancelled), min(termination_date)) AS termination_date,
                max((product_type = 'DENTAL')::INT) AS has_dental,
                max((product_type = 'VISION')::INT) AS has_vision,
                max((product_type = 'SUPPLEMENTAL')::INT) AS has_supp,
                sum(CASE WHEN product_type = 'DENTAL' THEN premium_amount ELSE null END) AS premium_dental,
                sum(CASE WHEN product_type = 'VISION' THEN premium_amount ELSE null END) AS premium_vision,
                sum(CASE WHEN product_type = 'SUPPLEMENTAL' THEN premium_amount ELSE null END) AS premium_supp,
                sum(CASE WHEN product_type = 'DENTAL' THEN nullif(association_fee, 0.0) ELSE null END) AS association_fee_dental,
                sum(CASE WHEN product_type = 'DENTAL' THEN nullif(application_fee, 0.0) ELSE null END) AS enrollment_fee_dental,
                sum(CASE WHEN product_type = 'DENTAL' THEN admin_fee ELSE null END) AS admin_fee_dental,
                sum(CASE WHEN product_type = 'VISION' THEN admin_fee ELSE null END) AS admin_fee_vision
            FROM ph_transactional.application
            CROSS JOIN date_limit
            WHERE agency_id NOT ILIKE '%test%'
                AND upper(status) IN ('NEW', 'PENDING','IN_PROGRESS', 'PROCESSED', 'CANCELLED')
                AND upper(product_type) IN ('DENTAL', 'VISION', 'SUPPLEMENTAL')
            GROUP BY 1,2
            ),
        tracking AS (
            SELECT DISTINCT
                app_pid,
                first_value(session_id IGNORE NULLS) OVER (
                    PARTITION BY app_pid ORDER BY datetime_created ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) AS session_id,
                first_value(session_user_id IGNORE NULLS) OVER (
                    PARTITION BY app_pid ORDER BY datetime_created ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) AS user_id,
                md5(first_value(email IGNORE NULLS) OVER (
                    PARTITION BY app_pid ORDER BY datetime_created ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    )) AS email_md5,
                md5(last_value(phone_primary IGNORE NULLS) OVER (
                    PARTITION BY app_pid ORDER BY datetime_created ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    )) AS phone_md5,
                last_value(agent_id IGNORE NULLS) OVER (
                    PARTITION BY app_pid ORDER BY datetime_created ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) AS agent_id,
                last_value(agency_id IGNORE NULLS) OVER (
                    PARTITION BY app_pid ORDER BY datetime_created ASC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) AS agency_id,
                first_value(regexp_replace(trim(zip ), '[\n|\r|\t]*','') IGNORE NULLS) OVER (
                    PARTITION BY app_pid ORDER BY datetime_created DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) AS zip,
                first_value(city IGNORE NULLS) OVER (
                    PARTITION BY app_pid ORDER BY datetime_created DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) AS city,
                first_value(state IGNORE NULLS) OVER (
                    PARTITION BY app_pid ORDER BY datetime_created DESC ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                    ) AS state
            FROM ph_transactional.application 
            WHERE agency_id NOT ILIKE '%test%'
                AND upper(status) IN ('NEW', 'PENDING','IN_PROGRESS', 'PROCESSED', 'CANCELLED')
            ),
        app_premium AS (
            SELECT
                coalesce(s.app_pid, a.app_pid) AS app_pid,
                coalesce(s.foreign_pid, a.foreign_pid) AS foreign_pid,
                coalesce(s.datetime_created, a.datetime_created) AS datetime_created,
                coalesce(s.carrier_id, a.carrier_id) AS carrier_id,
                CASE 
                    WHEN s.product_type IS NOT null THEN s.product_type
                    WHEN s.product_type IS null AND a.has_dental = 1 THEN 'DENTAL - SA'
                    WHEN s.product_type IS null AND a.has_dental = 0 AND a.has_supp = 1 THEN 'SUPP - SA'
                    ELSE null END AS product_type,
                coalesce(a.has_dental, 0) AS has_dental,
                coalesce(a.has_vision, 0) AS has_vision,
                coalesce(a.has_supp, 0) AS has_supp,
                coalesce(s.effective_date, a.effective_date) AS effective_date,
                coalesce(s.termination_date, a.termination_date) AS termination_date,
                s.expiration_date,
                s.duration_sold,
                coalesce(
                    s.duration_utilized, 
                    CASE WHEN a.effective_date <= maxdate - 1
                        THEN least(maxdate - 1, a.termination_date) -  a.effective_date + 1
                        ELSE null END
                    ) AS duration_utilized,
               CASE
                   WHEN s.max_months_rated_on IS null OR s.sequence_max IS null THEN null
                   WHEN s.max_months_rated_on = 0 OR s.sequence_max = 0 then '90x1'
                   WHEN s.max_months_rated_on = 12 THEN '364x' || s.sequence_max
                   WHEN s.max_months_rated_on IN ('3', '6') THEN s.max_months_rated_on * 30 || 'x' || s.sequence_max
                   END AS duration_category,
                s.premium_health,
                s.premium_rider,
                a.premium_dental,
                a.premium_vision,
                a.premium_supp,
                s.billing_fee_health,
                s.association_fee_health,
                a.association_fee_dental,
                s.admin_fee_health,
                a.admin_fee_dental,
                a.admin_fee_vision,
                s.enrollment_fee_health,
                a.enrollment_fee_dental,
                t.session_id,
                t.user_id,
                t.email_md5,
                t.phone_md5,
                t.agent_id,
                t.agency_id,
                t.zip,
                t.city,
                t.state,
                least(s.datetime_modified, a.datetime_modified) AS datetime_modified
            FROM app_base AS s
            FULL JOIN app_addon AS a
                ON s.app_pid = a.app_pid
                AND s.foreign_pid = a.foreign_pid
            LEFT JOIN tracking AS t
                ON coalesce(s.app_pid, a.app_pid) = t.app_pid
            CROSS JOIN date_limit
            ),
        member_ages AS (
            SELECT
                r.app_pid,
                CASE 
                    WHEN lower(r.relationship) = 'primary' AND (t.effective_date - r.dob) < 365 * 18
                    THEN 'child' ELSE lower(r.relationship)
                    END AS relationship,
                round((t.effective_date - r.dob)::NUMERIC / 365.25, 2) AS age
            FROM internal_raw.ph_policy_member AS r
            INNER JOIN app_premium AS t
                ON r.app_pid = t.app_pid
            ),
        members_consolidated AS (
            SELECT
                app_pid,
                count(1) AS plan_members,
                count(CASE WHEN relationship = 'spouse' THEN 1 ELSE null END) AS spousal_members,
                count(CASE WHEN relationship = 'child' THEN 1 ELSE null END) AS children_members,
                max(CASE WHEN relationship = 'primary' THEN age ELSE null END) AS primary_age,
                max(CASE WHEN relationship = 'spouse' THEN age ELSE null END) AS spouse_age,
                min(CASE WHEN relationship = 'child' THEN age ELSE null END) AS child_age_youngest,
                max(CASE WHEN relationship = 'child' THEN age ELSE null END) AS child_age_oldest,
                avg(CASE WHEN relationship = 'child' THEN age ELSE null END) AS child_age_average
            FROM member_ages
            GROUP BY 1
            ),
        members_summary AS (
            SELECT
                p.app_pid,
                m.plan_members,
                m.spousal_members,
                m.children_members,
                m.plan_members = m.children_members AS children_only,
                CASE WHEN m.primary_age IS NOT null
                    THEN p.gender ELSE null 
                    END AS primary_gender,
                m.primary_age,
                m.spouse_age,
                CASE WHEN m.children_members > 1 
                    THEN m.child_age_youngest ELSE null
                    END AS child_age_youngest,
                m.child_age_oldest,
                m.child_age_average
            FROM internal_raw.ph_policy_member AS p
            INNER JOIN members_consolidated AS m
                ON p.app_pid::INT = m.app_pid::INT
            WHERE p.relationship = 'primary'
            )
        SELECT
            a.app_pid,
            a.foreign_pid,
            a.datetime_created,
            a.carrier_id,
            a.product_type,
            a.has_dental,
            a.has_vision,
            a.has_supp,
            a.effective_date,
            a.termination_date,
            a.expiration_date,
            a.duration_sold,
            a.duration_utilized,
            a.duration_category,
            a.premium_health,
            a.premium_rider,
            a.premium_dental,
            a.premium_vision,
            a.premium_supp,
            a.billing_fee_health,
            a.association_fee_health,
            a.association_fee_dental,
            a.admin_fee_health,
            a.admin_fee_dental,
            a.admin_fee_vision,
            a.enrollment_fee_health,
            a.enrollment_fee_dental,
            a.zip,
            a.city,
            a.state,
            coalesce(m.plan_members, 1) AS plan_members,
            coalesce(m.spousal_members, 0) AS spousal_members,
            coalesce(m.children_members, 0) AS children_members,
            m.primary_age,
            m.spouse_age,
            m.child_age_youngest,
            m.child_age_oldest,
            m.child_age_average,
            a.session_id,
            a.user_id,
            a.email_md5,
            a.phone_md5,
            a.agent_id,
            a.agency_id,
            a.datetime_modified AS app_last_modified
        FROM app_premium AS a
        LEFT JOIN members_consolidated AS m
            ON a.app_pid = m.app_pid
        CROSS JOIN date_limit
        ;
    """

    with PivotDW() as db_context:
        db_context.exec(f"DROP MATERIALIZED VIEW IF EXISTS {schema}.{table};")
        db_context.exec(create)

if __name__ == '__main__':
    view_create()

