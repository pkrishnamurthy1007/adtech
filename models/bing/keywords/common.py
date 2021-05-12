#%%
#### RETRIEVE ENVIRON BASE FROM SECRET MANAGER ####
import boto3
import json
import os
import base64
from pkg_resources import resource_filename as rscfn


secretsmanager = boto3.client('secretsmanager')
sm_env_base_secret = secretsmanager.get_secret_value(
    SecretId='SM_ENV_BASE')
sm_env_base = json.loads(
    base64.b64decode(
        sm_env_base_secret["SecretBinary"]))
# already set os env vars take precendence over aws vals
os.environ.update({**sm_env_base, **os.environ})
#%%
S3_OUTPUT_BUCKET = "hc-data-lake-storage"
S3_OUTPUT_PREFIX = "prod/data-science/bing-keyword-bids"
OUTPUT_DIR = rscfn(__name__, "OUTPUT")
os.makedirs(OUTPUT_DIR, exist_ok=True)
#%%
