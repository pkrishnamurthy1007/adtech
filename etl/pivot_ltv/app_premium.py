from ds_utils.db.connectors import PivotDW


schema = 'tracking'
table = 'app_premium'


def view_create():

    create = f"""
        CREATE MATERIALIZED VIEW {schema}.{table} AS
        WITH
            date_limit AS (
                SELECT max(datetime_created::DATE) AS maxdate 
                FROM ph_transactional.application
                ),
            bad_pids AS (
                SELECT DISTINCT app_pid 
                FROM ph_transactional.application 
                WHERE (
                    status IN ('FAKE', 'VOID') OR
                    agency_id ILIKE '%test%'
                    )
                    AND app_pid IS NOT null
                ),
            health_fixes AS (
                SELECT DISTINCT app_pid, 
                    first_value(product_type IGNORE NULLS) OVER (
                        PARTITION BY app_pid ORDER BY effective_date ASC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS product_type_base,
                    first_value(premium_amount IGNORE NULLS) OVER (
                        PARTITION BY app_pid ORDER BY effective_date ASC
                        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                        ) AS premium_amount_base
                FROM ph_transactional.application
                WHERE app_pid IS NOT null
                    AND upper(product_type) IN ('STM', 'BTM')
                ),
            base_filtered AS (
                SELECT
                    upper(coalesce(a.product_type, f.product_type_base)) AS product_fixed,
                    a.expiration_date - a.effective_date + 1 AS coverage_duration_fixed,
                    coalesce(a.premium_amount, f.premium_amount_base) AS premium_fixed,
                    a.*
                FROM ph_transactional.application AS a
                LEFT JOIN bad_pids AS b
                    ON a.app_pid = b.app_pid
                LEFT JOIN health_fixes AS f
                    ON a.app_pid = f.app_pid
                WHERE coalesce(a.product_type, f.product_type_base) IS NOT null
                    AND a.app_pid IS NOT null
                    AND b.app_pid IS null
                 ),
            health_cancelled AS (
                SELECT DISTINCT
                    app_pid,
                    min(effective_date) AS health_min_effective_date,
                    min(least(termination_date, date_cancelled)) AS health_lapse_date
                FROM base_filtered AS a
                WHERE app_pid IS NOT null
                    AND product_fixed IN ('STM', 'BTM')
                    AND coalesce(termination_date, date_cancelled) IS NOT null
                GROUP BY 1
                ),
            addon_cancelled AS (
                SELECT 
                    app_pid,
                    min(effective_date) AS addon_min_effective_date,
                    min(least(termination_date, date_cancelled)) AS addon_lapse_date
                FROM base_filtered
                WHERE app_pid IS NOT null
                    AND product_fixed IN ('DENTAL', 'VISION', 'SUPPLEMENTAL')
                    AND coalesce(termination_date, date_cancelled) IS NOT null
                GROUP BY 1
                ),
        base_fixed AS (
            SELECT
                b.app_pid,
                coalesce(b.foreign_pid, b.app_pid::VARCHAR) AS foreign_pid,
                b.product_fixed AS product_type,
                b.coverage_duration_fixed AS coverage_duration,
                greatest(CASE
                    WHEN upper(b.payment_frequency) = 'PREPAID' THEN b.expiration_date + 1
                    WHEN b.expiration_date <= d.maxdate THEN coalesce(h.health_lapse_date, b.expiration_date + 1) 
                    WHEN b.expiration_date > d.maxdate THEN coalesce(h.health_lapse_date, greatest(d.maxdate, b.effective_date)) 
                    ELSE h.health_lapse_date END - b.effective_date, 0) AS health_duration_inforce,
                greatest(coalesce(a.addon_lapse_date, greatest(d.maxdate, b.effective_date)) - b.effective_date, 0) AS addon_duration_inforce,
                CASE WHEN upper(b.payment_frequency) = 'PREPAID' THEN 1 ELSE 0 END AS is_prepaid,
                b.status,
                b.carrier_id,
                b.effective_date,
                b.expiration_date,
                h.health_min_effective_date,
                h.health_lapse_date,
                a.addon_min_effective_date,
                a.addon_lapse_date,
                b.datetime_created,
                b.datetime_modified,
                least(h.health_lapse_date, a.addon_lapse_date) AS termination_date,
                CASE WHEN upper(payment_frequency) = 'PREPAID'
                    THEN (b.premium_fixed / b.coverage_duration_fixed::NUMERIC * 30.436875)::NUMERIC(9,4)
                    ELSE b.premium_fixed END AS premium_amount,
                b.billing_fee,
                b.association_fee,
                b.application_fee,
                b.admin_fee,
                b.extras_amount
            FROM base_filtered AS b
            CROSS JOIN date_limit AS d
            LEFT JOIN health_cancelled AS h
                ON h.app_pid = b.app_pid
            LEFT JOIN addon_cancelled AS a
                ON a.app_pid = b.app_pid
            ),
        app_base AS (
            SELECT
                app_pid,
                coalesce(foreign_pid, app_pid::VARCHAR) AS foreign_pid,
                any_value(carrier_id) AS carrier_id,
                any_value(product_type) AS product_type,
                min(datetime_created) AS datetime_created,
                max(datetime_modified) AS datetime_modified,
                min(effective_date) AS effective_date,
                min(health_lapse_date) AS termination_date,
                max(expiration_date) AS expiration_date,
                sum(coverage_duration) AS duration_sold,
                sum(health_duration_inforce) AS duration_inforce,
                max(is_prepaid) AS is_prepaid,
                count(nullif(coverage_duration, 0)) AS sequence_max,
                sum(premium_amount * coverage_duration) / sum(coverage_duration) AS premium_health,
                nullif(sum(billing_fee * coverage_duration) / sum(coverage_duration), 0.0) AS billing_fee_health,
                nullif(sum(association_fee * coverage_duration) / sum(coverage_duration), 0.0) AS association_fee_health,
                nullif(sum(admin_fee * coverage_duration) / sum(coverage_duration), 0.0) AS admin_fee_health,
                nullif(max(application_fee), 0.0) AS enrollment_fee_health,
                nullif(sum(extras_amount * coverage_duration) / sum(coverage_duration), 0.0) AS premium_rider
            FROM base_fixed
            WHERE product_type IN ('BTM', 'STM')
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
                    min(addon_lapse_date) AS termination_date,
                    avg(nullif(addon_duration_inforce, 0)) AS duration_inforce,
                    max(is_prepaid) AS is_prepaid,
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
                FROM base_fixed
                WHERE product_type IN ('DENTAL', 'VISION', 'SUPPLEMENTAL')
                    AND premium_amount IS NOT null
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
                    WHEN s.product_type IS null AND a.has_dental = 0 AND a.has_supp = 1 THEN 'SUPPLEMENTAL - SA'
                    ELSE null END AS product_type,
                coalesce(a.has_dental, 0) AS has_dental,
                coalesce(a.has_vision, 0) AS has_vision,
                coalesce(a.has_supp, 0) AS has_supp,
                coalesce(s.effective_date, a.effective_date) AS effective_date,
                coalesce(s.termination_date, a.termination_date) AS termination_date,
                s.expiration_date,
                s.duration_sold,
                coalesce(s.duration_inforce, a.duration_inforce) AS duration_inforce,
                coalesce(s.is_prepaid, a.is_prepaid) AS is_prepaid,
                s.sequence_max,
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
            a.termination_date,
            CASE 
                WHEN coalesce(a.termination_date, a.expiration_date) - a.effective_date <= 10 
                    THEN 'CANCELLED'
                WHEN coalesce(a.termination_date, a.expiration_date) - a.effective_date > 10
                    AND coalesce(a.termination_date, a.expiration_date) <= maxdate
                    THEN 'LAPSED'
                WHEN coalesce(a.termination_date, a.expiration_date) IS null
                    OR coalesce(a.termination_date, a.expiration_date) > maxdate
                    THEN 'ACTIVE'
                ELSE 'UNKNOWN' END AS status,
            a.carrier_id,
            a.product_type,
            a.has_dental,
            a.has_vision,
            a.has_supp,
            a.effective_date,
            a.expiration_date,
            CASE
                WHEN a.duration_sold < 45 THEN '30'
                WHEN a.duration_sold BETWEEN 45 AND 74 THEN '60'
                WHEN a.duration_sold BETWEEN 75 AND 134 THEN '90'
                WHEN a.duration_sold BETWEEN 135 AND 264 THEN '180'
                WHEN a.duration_sold BETWEEN 265 AND 364 THEN '364'
                WHEN a.duration_sold > 364 THEN '364+'
                ELSE null END AS plan_group,
            CASE
                WHEN a.duration_sold / a.sequence_max < 45 THEN '30'
                WHEN a.duration_sold / a.sequence_max BETWEEN 45 AND 74 THEN '60'
                WHEN a.duration_sold / a.sequence_max BETWEEN 75 AND 134 THEN '90'
                WHEN a.duration_sold / a.sequence_max BETWEEN 135 AND 264 THEN '180'
                WHEN a.duration_sold / a.sequence_max > 264 THEN '364'
                ELSE null END || 'x' || sequence_max::VARCHAR AS plan_type,
            CASE
                WHEN a.duration_sold / a.sequence_max < 45 THEN 1
                WHEN a.duration_sold / a.sequence_max BETWEEN 45 AND 74 THEN 2
                WHEN a.duration_sold / a.sequence_max BETWEEN 75 AND 134 THEN 3
                WHEN a.duration_sold / a.sequence_max BETWEEN 135 AND 264 THEN 4
                WHEN a.duration_sold / a.sequence_max > 264 THEN 5
                ELSE 6 END AS plan_type_order,
            a.sequence_max,
            a.duration_sold,
            a.duration_inforce,
            a.is_prepaid,
            (a.duration_inforce::NUMERIC / a.duration_sold::NUMERIC)::NUMERIC(5,4) AS inforce_percent,
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
            a.datetime_modified AS app_last_modified,
            d.maxdate AS asof_date
        FROM app_premium AS a
        LEFT JOIN members_consolidated AS m
            ON a.app_pid = m.app_pid
        CROSS JOIN date_limit AS d
        WHERE a.app_pid NOT IN (301093, 315478, 314141, 308897)
        ;
    """

    with PivotDW() as db_context:
        db_context.exec(f"DROP MATERIALIZED VIEW IF EXISTS {schema}.{table};")
        db_context.exec(create)

if __name__ == '__main__':
    view_create()

