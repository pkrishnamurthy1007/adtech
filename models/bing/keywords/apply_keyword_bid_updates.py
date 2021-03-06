
#%%
import json
import logging
import os
import glob
import datetime
import pandas as pd
import numpy as np
from api.bingads.bingapi.client import BingClient,LocalAuthorizationData
from models.bing.keywords.common import *

# Debug requests and responses
LOGLEVEL = logging.WARN
logging.basicConfig(level=LOGLEVEL)
# logging.getLogger('suds.client').setLevel(logging.DEBUG)
# logging.getLogger('suds.transport.http').setLevel(logging.DEBUG)

bing_creds = json.loads(os.getenv("BING_CREDS"))
# bing_creds["BING_REFRESH_TOKEN"] = "M.R3_BAY.-CeefO6bNKWF4el5x*Edtx4dBCeilSbm0YDBj0UO5Q28x9oVk1!SBMUyqbf1snTjVyWi6tilC0ALPANuHLtKkZEkR5etP9ax9WFxGGjVHOIDsUUo8EwRdk5QXwHrfMa4XMdk40ib4ZyYF3Fm4S8PpE3IkKzPUfNQCibqnpDVGFxjArMFJaQVJ7xhXTu0fZUjinEy3ZSuWx4y87Jw9UUWVnHMV*VFadK60SeTsg!lziXve5dB96xTb74ieV!YlKxU63GnxlDC3*UoK7Te1k1gh5heZZE7j*WIm*ODSrvRHP1NYdmpKW3vq7T5VDVhem5dKklEMLmcoH1XjDdkpHMernABvQS16B7Gkmv5ci3tAJkLh"

# from ds_utils.db.connectors import HealthcareDW,AnalyticsDB
# with AnalyticsDB() as db:
#     bing_accnt_df = db.to_df("select * from dev_ent_d1_gold.adtech.bingads.account_id")
# bing_accnt_df
bing_accnt_df = pd.DataFrame(
    [
        ["B013P57C", "43030867"],
        ["X000EWRC", "3196099"],
        ["B013D68T", "43060164"],
    ],
    columns=["account_number", "account_id"],
)

import boto3
ls_resp = boto3.client("s3").list_objects(Bucket=S3_OUTPUT_BUCKET,Prefix=S3_OUTPUT_PREFIX)
prev_output_keys = [o["Key"] for o in ls_resp["Contents"]]
todays_output_keys = [k for k in prev_output_keys if k.endswith(f"{TODAY}.csv")]

todays_output = [pd.read_csv(f"s3://{S3_OUTPUT_BUCKET}/{k}") for k in todays_output_keys]
df_out = pd.concat(todays_output)
old_len = df_out.__len__()
df_out = df_out.drop_duplicates() 
assert df_out.__len__() * 2 == old_len, """
We should have 2 records for each kw b/c we break bids out by account_id and write them,
but we also write the bids for all accounts.
"""
TODAY
#%%
from api.bingads.bingapi.client import *

for accnt_id in df_out["account_id"].astype(int).unique():
    print(f"Updating bids for account_id `{accnt_id}`")
    accntI = df_out["account_id"] == accnt_id
    accnt_client = BingClient(
        account_id=accnt_id,
        customer_id=bing_creds['BING_CUSTOMER_ID'],
        dev_token=bing_creds['BING_DEVELOPER_TOKEN'],
        client_id=bing_creds['BING_CLIENT_ID'],
        refresh_token=bing_creds['BING_REFRESH_TOKEN'],
        loglevel=LOGLEVEL,
    )
    adgroup_ids, keyword_ids, keyword_bids_old, keyword_bids_new = df_out \
        .loc[accntI, ['adgroup_id','keyword_id',"max_cpc_old","max_cpc_new"]].values.T
    keyword_bids_test = (
        keyword_bids_old + np.random.rand(len(keyword_bids_new)) * 0.02 - 0.01).round(2)
    df_out.loc[accntI,"max_cpc_test"] = keyword_bids_test

    adgroup_ids = [*adgroup_ids.astype(int)]
    keyword_ids = [*keyword_ids.astype(int)]
    keyword_bids_old = [*keyword_bids_old.astype(float)]
    keyword_bids_new = [*keyword_bids_new.astype(float)]
    keyword_bids_test = [*keyword_bids_test.astype(float)]
    # keyword_bids = keyword_bids_test # only update w/ a random push of +- 0.01$ to test
    keyword_bids = keyword_bids_new # actually update keyword bids
    keyword_updates = accnt_client.bulk_update_keyword_bids(adgroup_ids,keyword_ids,keyword_bids)
    keyword_bid_updates = [kwu.keyword.Bid.Amount for kwu in keyword_updates]
    assert all(np.array(sorted(keyword_bid_updates)) == np.array(sorted(keyword_bids)))
# %%
# as part of auth flow a new refresh token is generated - store in AWS secret manager
bing_creds["BING_REFRESH_TOKEN"] = accnt_client.authentication._oauth_tokens._refresh_token
os.environ["BING_CREDS"] = json.dumps(bing_creds)
from utils.env import dump_env_to_aws
dump_env_to_aws()