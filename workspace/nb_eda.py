#%%
import typing
from ds_utils.db.connectors import \
    HealthcareDW, PivotDW, AnalyticsDB, \
    MySqlContext, RedshiftContext

import tqdm
import traceback as tb

def fetch_schema(src: RedshiftContext):
    with src() as db:
        currdb = db.to_df("select CURRENT_DATABASE();")
        dbs = db.to_df("select * from pg_database")
        table_df = db.to_df("SELECT * FROM information_schema.tables")
        schemas = {*table_df["table_schema"]}
        with db.conn.cursor() as c:
            c.execute(f"SET search_path TO {','.join(schemas)}")
        sp = db.to_df("show search_path;")["search_path"][0]
        schemadf = db.to_df(
            """
            SELECT * FROM pg_table_def
            """
        )
        schemadf = schemadf.set_index(
            ["schemaname", "tablename", "column"], drop=False)
    schemadf.to_csv(f"{TABLE_DUMPS}/{src.__name__}.schema.csv")

    schemadf \
        [["schemaname","tablename","column"]]  \
        .reset_index(drop=True) \
        .groupby(["schemaname","tablename"]) .count() \
        .to_csv(f"{TABLE_DUMPS}/{src.__name__}.schema2table.csv")

from pkg_resources import resource_filename as rscfn
import os
TABLE_DUMPS = rscfn(__name__,"TABLE_DUMPS")
os.makedirs(TABLE_DUMPS,exist_ok=True)
def fetch_head(rsc,limit=1000,src=PivotDW):
    with src() as db:
        db \
            .to_df(f"select * from {rsc} limit {limit}") \
            .to_csv(f"{TABLE_DUMPS}/{src.__name__}.{rsc}.csv")


# PIVOT
fetch_schema(PivotDW)
# fetch_head("internal_raw.pivothealth_com_user")
# fetch_head("internal_raw.session_enroll")
# fetch_head("data_science.pivot_plan_sales")
# fetch_head("ph_transactional.application")
# fetch_head("salesforce.accounts")
# fetch_head("salesforce.opportunities")
# fetch_head("salesforce.policies")
# fetch_head("session.master_table")
# fetch_head("sales_center.call_performance")
# fetch_head("sales_center.multicarrier_sales")
# fetch_head("sales_center.sales")
# fetch_head("session.user_attribution_model")
# fetch_head("session.application_touch_points")
# fetch_head("reports.sales_view", src=PivotDW)

# HC
fetch_schema(HealthcareDW)
# fetch_head("log_upload.tron_session_revenue",src=HealthcareDW) 
#%%

#%%
with PivotDW() as db:
    df = db.to_df(
        """
        select 
            conversion,
            (app_id IS NOT NULL) as applied,
            (DATEDIFF(day, session_creation_date, conversion_date) > 0) as lag1,
            count(*)
        from 
            session.user_attribution_model
        group by 
            conversion, applied, lag1
        """
    )
df
#%%
with PivotDW() as db:
    df = db.to_df(
        """
        with t as (
            select
                user_id,app_id
                sum(coalesce(conversion,False)::int) as num_conversions,
                count(*) as num_sessions
            from 
                session.user_attribution_model
            group by 
                user_id,app_id
        )
        select t.* from t where t.num_conversions > 0   
        """
    )
df
#%%
with PivotDW() as db:
    df = db.to_df(
        """
        select 
            (overflow_campaign IS NOT NULL) as overflow,
            overflow_is_sale,
            count(*),
            avg(queue_time::interval) as queue_time
        from 
            sales_center.call_performance
        group by 
            overflow,overflow_is_sale
        """
    )
df
#%%
with PivotDW() as db:
    df = db.to_df(
        """
        select 
            (overflow_campaign IS NOT NULL) as overflow,
            count(*),
            avg(queue_time::interval) as queue_time
        from 
            sales_center.call_performance
        group by 
            overflow
        """
    )
df
#%%
with PivotDW() as db:
    appdf = db.to_df(
        """
        select 
            s.*,
            t.*
        from 
            session.master_table s
        right join
            ph_transactional.application t 
        on s.user_id = REGEXP_REPLACE(UPPER(t.user_id),'-','')
        limit 100
        """
    )
appdf.shape
#%%
with PivotDW() as db:
    cnt = db.to_df(
        """
        select 
            
        from 
            session.master_table s
        right join 
            ph_transactional.application t 
        on s.user_id = REGEXP_REPLACE(UPPER(t.user_id),'-','')
        right join
            salesforce.opportunities o
        on s.user_id = o.account_user_id
        """
    )
cnt
#%%
with PivotDW() as db:
    cnt = db.to_df(
        """
        select 
            
        from 
            session.master_table s
        right join 
            ph_transactional.application t 
        on s.user_id = REGEXP_REPLACE(UPPER(t.user_id),'-','')
        right join
            salesforce.opportunities o
        on s.user_id = o.account_user_id
        """
    )
cnt
#%%
with PivotDW() as db:
    cnt = db.to_df(
        """
        select 
            count(*)
        from 
            session.master_table s
        right join 
            internal_raw.session_enroll e 
        on s.user_id = e.user_id
        """
    )
cnt
#%%
def summarize(rsc,src=PivotDW):
    with src() as db:
        cnt = db.to_df(
            f"select count(*) from {rsc}")
        schema = db.to_df(
            f"select * from {rsc} where False")
        return cnt,schema.shape,schema.columns
#%%
summarize("salesforce.opportunities")
#%%
summarize("salesforce.accounts_all")
#%%
summarize("internal_raw.ph_transactional_application")
#%%
summarize("internal_raw.ph_session_detail")
#%%
summarize("internal_raw.session_enroll")
#%%
summarize("ph_transactional.application")  # <--
#%%
summarize("session.master_table") # <--
#%%
summarize("tracking.app_premium_revised")
#%%
summarize("session.user_attribution_model")
#%%
summarize("sales_center.call_performance")
#%%
def inspect_mysql_db(mysqlsrc: typing.Callable):
    database2table2info = {}
    
    with mysqlsrc() as sqldb:
        c = sqldb.conn.cursor()
        c.execute("select database();")
        c.fetchall()
        c.execute(
            """
            select table_name from information_schema.tables
            where table_schema not in ('information_schema', 'mysql', 'performance_schema')
            """)
        tables, *_ = zip(*c.fetchall())
        c.execute("show databases;")
        dbnms, *_ = zip(*c.fetchall())

        fieldC = ["Field", "Type", "Null", "Key", "Default", "Extra"]
        for dbnm in tqdm.tqdm(dbnms):
            print(dbnm)
            c.execute(f"use {dbnm};")
            c.execute("show tables;")
            tables = c.fetchall()
            if len(tables) == 0:
                print(f"no tables for {dbnm} - continuing!")
                continue
            tables, *_ = zip(*tables)
            for tnm in tables:
                # print(t)
                try:
                    globals()["query"] = f'DESCRIBE "{tnm}";'
                    c.execute(f'DESCRIBE `{tnm}`;')
                    fields = c.fetchall()
                    schema = {f: tnm for f, t, *_ in fields}
                    # c.execute(f'SELECT * FROM `{tnm}` LIMIT 3')
                    # rows = c.fetchall()
                    # objs = [dict(zip(schema.keys(), r)) for r in rows]
                    # info = {
                    #     "schema": schema,
                    #     "rows": rows,
                    #     "objs": objs,
                    # }
                    database2table2info \
                        .setdefault(dbnm, {})[tnm] = schema
                except Exception as e:
                    print(f"describe {tnm} failed w/ {type(e)}: {e}")
    
    return database2table2info

def inspect_redshift_db(rssrc: typing.Callable):
    database2table2info = {}

    with rssrc() as rsdb:
        c = rsdb.conn.cursor()
        c.execute("SELECT * FROM pg_database;")
        dbnms, *_ = zip(*c.fetchall())

    for dbnm in tqdm.tqdm(dbnms):
        try:
            with rssrc(database=dbnm) as tmpdb:
                TABLE_DEF_C = ["schemaname",
                               "tablename",
                               "column",
                               "type",
                               "encoding",
                               "distkey",
                               "sortkey",
                               "notnull"]
                with tmpdb.conn.cursor() as c:
                    c.execute(
                        """
                        SELECT "tablename", "column", "type"
                        FROM pg_table_def
                        WHERE schemaname = 'public'
                        ORDER BY tablename;
                        """
                    )
                    # c.execute(
                    #     """
                    #     SELECT "tablename", "column", "type"
                    #     FROM pg_table_def
                    #     ORDER BY tablename;
                    #     """
                    # )
                    columndefs = c.fetchall()
                tnm2schema = {}
                for tnm, cnm, tpe in columndefs:
                    tnm2schema \
                        .setdefault(tnm,{}) \
                        [cnm] = tpe
                for tnm,schema in tnm2schema.items():
                    # try:
                    #     with tmpdb.conn.cursor() as c:
                    #         c.execute(f'SELECT * FROM "{tnm}" LIMIT 3')
                    #         rows = c.fetchall()
                    #         objs = [dict(zip(schema.keys(), r)) for r in rows]
                    # except Exception as e:
                    #     print(f"select failed w/ {type(e)}: {e}")
                    #     tb.print_exc()
                    #     tmpdb.conn.rollback()
                    #     rows = []
                    #     objs = []
                    # info = {
                    #     "schema": schema,
                    #     "rows": rows,
                    #     "objs": objs,
                    # }
                    database2table2info \
                        .setdefault(dbnm, {}) \
                        [tnm] = schema
        except Exception as e:
            print(f"connection failed w/ {type(e)}: {e}")
            tb.print_exc()
            continue
    
    return database2table2info


srcs = [AnalyticsDB, HealthcareDW, PivotDW]
src2database2table2info = {}
for src in srcs:
    print("inspecting:",src)
    if issubclass(src,MySqlContext):
        # continue
        src2database2table2info[src.__name__] = inspect_mysql_db(src)
    elif issubclass(src,RedshiftContext):
        src2database2table2info[src.__name__] = inspect_redshift_db(src)
    else:
        print("unrecognized src:",src)

import json
json.dump(
    src2database2table2info,
    open("data_itinerary.json","w"),
    indent=2,
    default=str,
)
#%%
# auto commit after each exec on by default
with HealthcareDW() as db_context:
    db_context.exec("CREAT TABLE _temp (a INT);")
    db_context.exec("INSERT...")

# manual control of commit (only commit when the with statement exits without errors)
with PivotDW(auto_commit=False) as db_context:
    db_context.exec("UPDATE...")
    db_context.exec("INSERT...")
    db_context.commit()
