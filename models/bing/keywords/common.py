#%%
#### RETRIEVE ENVIRON BASE FROM SECRET MANAGER ####
from utils.env import load_env_from_aws
load_env_from_aws()

import boto3
import json
import os
import base64
from pkg_resources import resource_filename as rscfn

from models.bing.keywords.config import *
#%%
S3_OUTPUT_BUCKET = "hc-data-lake-storage"
S3_OUTPUT_PREFIX = "prod/data-science/bing-keyword-bids"
OUTPUT_DIR = rscfn(__name__, "OUTPUT")
os.makedirs(OUTPUT_DIR, exist_ok=True)
#%%
import pytz
import datetime
NOW = datetime.datetime.now(pytz.timezone('EST'))
TODAY = NOW.date()
#%%
from ds_utils.db.connectors import HealthcareDW
# TODO: pull down keywords for all active campaigns in data window as kw df
with HealthcareDW(database="adtech") as db:
    accnt_df = db.to_df("select * from dl_gold.adtech_bingads_account")
# with HealthcareDW(database="adtech") as db:
#     kw_df = db.to_df("select * from dl_gold.adtech_bingads_keyword")
accnt_df = accnt_df .set_index("account_number")
accnt_df.loc[["B013D68T", "B013P57C", "X000EWRC"],
            ["keyword_bidder_enabled","keyword_geo_bidder_enabled"]] = "Y"
accnt_df = accnt_df.reset_index()
#%%
