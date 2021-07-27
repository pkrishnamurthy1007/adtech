#%%
from models.taboola.common import *
from pytaboola import TaboolaClient
from pytaboola.services import AccountService, CampaignService, CampaignSummaryReport
# d = CampaignSummaryReport(client, O65_ACCNT_ID).fetch(
#     dimension="campaign_day_breakdown",start_date=TODAY-7*DAY, end_date=TODAY)
# import jmespath
# jmespath.search("results[?cpc > `0`].{cpc: cpc,campaign_id: campaign, utc_dt: date}",d)

client = TaboolaClient(**TABOOLA_HC_CREDS)
acct_service = AccountService(client)
accnts = acct_service.list()["results"]
id2accnt = {a["account_id"]: a for a in accnts}

camps = []
for aid in [TEST_ACCNT_ID, O65_ACCNT_ID]:
    camp_service = CampaignService(client, aid)
    camps += camp_service.list()

campdf = pd.DataFrame([camps],index=["body"]).T
campdf["body"] = campdf["body"].apply(json.dumps)
campdf["date"] = TODAY
campdf["datetime"] = NOW

table_creation_sql = f"""
    CREATE TABLE IF NOT EXISTS
    {DS_SCHEMA}.{TABOOLA_CAMPAIGN_TABLE}
    (
        "date"                  DATE,
        "datetime"              DATETIME,
        "body"                  SUPER
    );
""" 
with HealthcareDW() as db:
    db.exec(table_creation_sql)
    db.load_df(campdf, schema=DS_SCHEMA, table=TABOOLA_CAMPAIGN_TABLE)
# %%
