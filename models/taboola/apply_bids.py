#%%
1/0
#%%
from models.taboola.common import *

from pytaboola import TaboolaClient
from pytaboola.services import CampaignService
from ds_utils.db.connectors import HealthcareDW

sql = f"""
    SELECT 
        *
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY account_id,campaign_id
                ORDER BY datetime DESC
            ) as rn
        FROM 
            {DS_SCHEMA}.{TABOOLA_CAMPAIGN_UPDATE_TABLE}
    )
    WHERE 
        rn = 1
    ;
"""
with HealthcareDW() as db:
    updatedf = db.to_df(sql)

updatedf.loc[~updatedf['update'].isna(),'update'] = \
    updatedf.loc[~updatedf['update'].isna(),'update'].apply(json.loads)
updatedf.loc[~updatedf['schedule'].isna(), 'schedule'] = \
    updatedf.loc[~updatedf['schedule'].isna(), 'schedule'].apply(json.loads)
#%%
sql = f"""
    SELECT 
        *
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY body.id
                ORDER BY datetime DESC
            ) as rn
        FROM 
        {DS_SCHEMA}.{TABOOLA_CAMPAIGN_TABLE}
    )
    WHERE
        rn = 1 AND 
        body.is_active = TRUE AND 
        date >= {TODAY}
"""
with HealthcareDW() as db:
    df = db.to_df(sql)
df

pd.DataFrame(df["body"].apply(json.loads).tolist())

#%%
sql = f"""
    SELECT DISTINCT ON (body.id)
        *
    FROM {DS_SCHEMA}.{TABOOLA_CAMPAIGN_TABLE}
    WHERE
        body.is_active = TRUE
    ORDER BY body.id,datetime DESC
"""
with HealthcareDW() as db:
    df = db.to_df(sql)
df
"""
DatabaseError: Execution failed on sql '
    SELECT DISTINCT ON (body.id)
        *
    FROM data_science.taboola_campaign_test
    WHERE
        body.is_active = TRUE
    ORDER BY body.id,datetime DESC
': SELECT DISTINCT ON is not supported
"""
#%%
sql = f"""
    SELECT
        h,
        *
    FROM (
        SELECT 
            *,
            ROW_NUMBER() OVER (
                PARTITION BY account_id,campaign_id
                ORDER BY datetime DESC
            ) as rn
        FROM 
            {DS_SCHEMA}.{TABOOLA_CAMPAIGN_UPDATE_TABLE}
    ) u, u.schedule.hour h
    WHERE
        rn = 1
    ;
"""
with HealthcareDW() as db:
    df = db.to_df(sql)
df
#%%
for _, r in updatedf.iterrows():
    client = TaboolaClient(**TABOOLA_HC_CREDS)
    camp_service = CampaignService(client, r["account_id"])
    camp_service.update(r["campaign_id"], **r["update"])
