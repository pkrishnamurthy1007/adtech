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

campaign_data_df = pd.DataFrame([camps],index=["body"]).T
campaign_data_df["body"] = campaign_data_df["body"].apply(json.dumps)
campaign_data_df["date"] = TODAY
campaign_data_df["datetime"] = NOW

upload_taboola_campaign_data_to_redshift(campaign_data_df)
# %%
