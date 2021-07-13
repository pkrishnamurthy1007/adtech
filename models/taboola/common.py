#### RETRIEVE ENVIRON BASE FROM SECRET MANAGER ####
from utils.env import load_env_from_aws
load_env_from_aws()

import boto3
import json
import os
import base64
from pkg_resources import resource_filename as rscfn

from models.taboola.config import *
#%%
S3_OUTPUT_BUCKET = "hc-data-lake-storage"
S3_OUTPUT_PREFIX = "prod/data-science/taboola-bids"
OUTPUT_DIR = rscfn(__name__, "OUTPUT")
os.makedirs(OUTPUT_DIR, exist_ok=True)
#%%
