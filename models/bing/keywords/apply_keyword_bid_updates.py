#%%
import json
import logging
import os
import glob
import datetime
import pandas as pd
from api.bingads.bingapi.client import BingClient
from models.bing.keywords.common import *

bing_creds = json.loads(os.getenv("BING_CREDS"))

# from ds_utils.db.connectors import HealthcareDW,AnalyticsDB
# with AnalyticsDB() as db:
#     bing_accnt_df = db.to_df("select * from dev_ent_d1_gold.adtech.bingads.account")
# bing_accnt_df
bing_accnt_df = pd.DataFrame(
    [
        ["B013P57C", "43030867"],
        ["X000EWRC", "3196099"],
        ["B013D68T", "43060164"],
    ],
    columns=["account_number", "account_id"],
)

# TODAY = datetime.datetime.now().date()
TODAY = "2021-05-12"
todays_output = glob.glob(f"{OUTPUT_DIR}/**/*{TODAY}.csv")
df_out = pd.concat((pd.read_csv(fpth) for fpth in todays_output))

# Debug requests and responses
LOGLEVEL = logging.DEBUG
logging.basicConfig(level=LOGLEVEL)
logging.getLogger('suds.client').setLevel(LOGLEVEL)
logging.getLogger('suds.transport.http').setLevel(LOGLEVEL)

#%%
for accnt in df_out["account"].unique():
    print(f"Updating bids for account# `{accnt}`")
    accnt_id = bing_accnt_df \
        .set_index("account_number").loc[accnt,"account_id"]
    print(f"Got id `{accnt_id}` for account# `{accnt}`")
    accnt_client = BingClient(
        account_id=accnt_id,
        customer_id=bing_creds['BING_CUSTOMER_ID'],
        dev_token=bing_creds['BING_DEVELOPER_TOKEN'],
        client_id=bing_creds['BING_CLIENT_ID'],
        refresh_token=bing_creds['BING_REFRESH_TOKEN']
    )
    accntI = df_out["account"] == accnt

    import tqdm
    adgps = df_out.loc[accntI, "adgroup_id"].unique()
    print(f"got {len(adgps)} adgroups for account# {accnt}")
    for adgp in tqdm.tqdm(adgps):
        adgpI = df_out["adgroup_id"] == adgp
        keyword_ids, keyword_bids = df_out \
            .loc[accntI & adgpI, ['keyword_id', "max_cpc_old"]].values.T
        response = accnt_client.update_keyword_bids(
            adgp, [*keyword_ids.astype(int)], [*keyword_bids.astype(float)])
        break
        # print(f"Adgroup {adgroup} success: {response}")
#%%
keyword_ids = keyword_ids.astype(int)
len(keyword_ids),len({*keyword_ids})
#%%
accnt
#%%
bing_accnt_df
#%%
cid = df_out.loc[accntI,"campaign_id"].unique()[0]
accnt_client.campaign_service.GetAdGroupsByCampaignId(
    CampaignId=cid
)
#%%
accnt_client.get_campaign_by_id(cid)
#%%
cid
#%%
client.campaign_service.GetKeywordsByIds(
    AdGroupId=adgp,
    KeywordIds=[*keyword_ids.astype(int)],
)
#%%
kwids = client.campaign_service.factory.create('KeywordIds')
for kwid in keyword_ids.astype(int):
    kwids.long.append(kwid)
client.campaign_service.GetKeywordsByIds(
    AdGroupId=adgp,
    KeywordIds=kwids,
)
kwids
#%%
kws = accnt_client.campaign_service.GetKeywordsByAdGroupId(
    AdGroupId=int(adgp))
kws
#%%
type(adgp)
#%%
# Debug requests and responses
logging.basicConfig(level=logging.INFO)
logging.getLogger('suds.client').setLevel(logging.DEBUG)
logging.getLogger('suds.transport.http').setLevel(logging.DEBUG)
#%%
key
#%%

client = BingClient(
    account_id=bing_accnt_df.set_index("account_number").loc["B013P57C","account_id"],
    # account_id=credentials['BING_ACCOUNT_ID'],
    customer_id=bing_creds['BING_CUSTOMER_ID'],
    dev_token=bing_creds['BING_DEVELOPER_TOKEN'],
    client_id=bing_creds['BING_CLIENT_ID'],
    refresh_token=bing_creds['BING_REFRESH_TOKEN']
)


adgroups = df['adgroup_id'].unique().tolist()

for adgroup in adgroups:
    temp_df = df[df['adgroup_id'] == adgroup]
    keyword_ids = temp_df['keyword_id'].tolist()
    keyword_bids = temp_df['bid'].tolist()

    response = client.update_keyword_bids(
        adgroup, keyword_ids, keyword_bids)
    print(f"Adgroup {adgroup} success: {response}")

#%%
