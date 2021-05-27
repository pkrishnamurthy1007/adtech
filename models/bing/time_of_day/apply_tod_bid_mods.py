#%%
import sys
sys.exit(1)
#%%
import boto3
import json
import logging
import os
import glob
import datetime
import pandas as pd
import numpy as np
from api.bingads.bingapi.client import BingClient, LocalAuthorizationData
from models.bing.time_of_day.common import *

# Debug requests and responses
LOGLEVEL = logging.WARN
logging.basicConfig(level=LOGLEVEL)
# logging.getLogger('suds.client').setLevel(logging.DEBUG)
# logging.getLogger('suds.transport.http').setLevel(logging.DEBUG)

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

TODAY = datetime.datetime.now().date()
# todays_output = glob.glob(f"{OUTPUT_DIR}/**/*{TODAY}.csv")
# df_out = pd.concat((pd.read_csv(fpth) for fpth in todays_output))
# ls_resp = boto3.client("s3").list_objects(
#     Bucket=S3_OUTPUT_BUCKET, Prefix=S3_OUTPUT_PREFIX)
# TODO: break out TOD modifiers by prefix
# for now - only update TOD modifiers for product=None
#   which corresponds to all product types
PRODUCT_TYPE = None
ls_resp = boto3.client("s3").list_objects(
    Bucket=S3_OUTPUT_BUCKET, Prefix=f"{S3_OUTPUT_PREFIX}/{PRODUCT_TYPE}")
[o["Key"] for o in ls_resp["Contents"]]
#%%
prev_output_keys = [o["Key"] for o in ls_resp["Contents"]]
todays_output_keys = [
    k for k in prev_output_keys if k.endswith(f"{TODAY}.csv")]

todays_output = [pd.read_csv(f"s3://{S3_OUTPUT_BUCKET}/{k}")
                 for k in todays_output_keys]
df_out = pd.concat(todays_output)
old_len = df_out.__len__()
df_out = df_out.drop_duplicates()
#%%
accnt_id = bing_accnt_df["account_id"].unique()[0]
accnt_client = BingClient(
    account_id=accnt_id,
    customer_id=bing_creds['BING_CUSTOMER_ID'],
    dev_token=bing_creds['BING_DEVELOPER_TOKEN'],
    client_id=bing_creds['BING_CLIENT_ID'],
    refresh_token=bing_creds['BING_REFRESH_TOKEN'],
    loglevel=LOGLEVEL,
)
accnt_client
#%%

#%%

for accnt_id in bing_accnt_df["account_id"].unique():
    print(accnt_id)
#%%

assert df_out.__len__() * 2 == old_len, """
We should have 2 records for each kw b/c we break bids out by account and write them,
but we also write the bids for all accounts.
"""
#%%
