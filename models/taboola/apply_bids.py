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
                PARTITION BY account_id,campaign_id,schedule
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
updatedf['update'] = updatedf['update'].apply(json.loads)
#%%
for _, r in updatedf.iterrows():
    client = TaboolaClient(**TABOOLA_HC_CREDS)
    camp_service = CampaignService(client, r["account_id"])
    camp_service.update(r["campaign_id"], **r["update"])
