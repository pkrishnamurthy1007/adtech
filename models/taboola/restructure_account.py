#%%
1/0
#%%
from models.taboola.common import *
active_camp_df
# %%
camp_management_df = active_camp_df[['created',"id"]]
camp_management_df['created_date'] = TODAY
camp_management_df["created_datetime"] = NOW
camp_management_df["baseline_cpc"] = 1
camp_management_df["active"] = False
camp_management_df["automatically_managed"] = True
camp_management_df
#%%
table_creation_sql = f"""
    CREATE TABLE IF NOT EXISTS
    {DS_SCHEMA}.{TABOOLA_CAMPAIGN_MANAGEMENT_TABLE}
    (
        "created"               BOOL,
        "id"                    INT,
        "created_date"          DATE,
        "created_datetime"      DATETIME,
        "baseline_cpc"          FLOAT,
        "active"                BOOL,
        "automatically_managed" BOOL

    );
"""
with HealthcareDW() as db:
    db.exec(table_creation_sql)
    db.load_df(camp_management_df, schema=DS_SCHEMA, table=TABOOLA_CAMPAIGN_MANAGEMENT_TABLE)
# %%
