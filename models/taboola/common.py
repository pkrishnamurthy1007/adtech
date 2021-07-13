#%%
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
TABOOLA_HC_CREDS = json.loads(os.getenv("TABOOLA_HC_CREDS"))
TABOOLA_PIVOT_CREDS = json.loads(os.getenv("TABOOLA_PIVOT_CREDS"))

NETWORK_ACCNT_ID = "healthcareinc-network"
TEST_ACCNT_ID = "healthcareinc-sc2"
O65_ACCNT_ID = "taboolaaccount-rangaritahealthcarecom"
