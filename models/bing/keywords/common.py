#%%
#### RETRIEVE ENVIRON BASE FROM SECRET MANAGER ####
import boto3
import json
import os
import base64
from pkg_resources import resource_filename as rscfn
from utils.env import load_env_from_aws

load_env_from_aws()
#%%
S3_OUTPUT_BUCKET = "hc-data-lake-storage"
S3_OUTPUT_PREFIX = "prod/data-science/bing-keyword-bids"
OUTPUT_DIR = rscfn(__name__, "OUTPUT")
os.makedirs(OUTPUT_DIR, exist_ok=True)
#%%
ROI_TARGET = 1.15  # target we are aiming
CLICKS = 120  # click threshold. level at which kw uses all of its own data
MAX_PUSH = 0.2
MAX_CUT = -0.3
CPC_MIN = 0.05
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
