#%%
import os
import sys
# make sure we first look in `adtech` root folder
while os.path.split(sys.path[0])[1] != "adtech":
    sys.path.pop(0)
assert sys.path[0].endswith("adtech")

#### RETRIEVE ENVIRON BASE FROM SECRET MANAGER ####
from utils.env import load_env_from_aws
load_env_from_aws()

from pkg_resources import resource_filename as rscfn
import importlib
import datetime
import itertools
import collections
import pprint
import re
import json

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display as ipydisp

from models.taboola.config import *
#%%
TABOOLA_BASE = "https://backstage.taboola.com/backstage/api/1.0"

TABOOLA_HC_CREDS = json.loads(os.getenv("TABOOLA_HC_CREDS"))
TABOOLA_PIVOT_CREDS = json.loads(os.getenv("TABOOLA_PIVOT_CREDS"))

NETWORK_ACCNT_ID = "healthcareinc-network"
TEST_ACCNT_ID = "healthcareinc-sc2"
O65_ACCNT_ID = "taboolaaccount-rangaritahealthcarecom"

TABOOLA_MAX_PUBLISHER_MODS_PER_CAMPAIGN = 1500
TABOOLA_MAX_PUBLISHER_EXCL_PER_CAMPAIGN = 1500
#%%
import datetime
NOW = datetime.datetime.utcnow()
TODAY = NOW.date()
DAY = datetime.timedelta(days=1)

start_date = TODAY - LOOKBACK*DAY
eval_date = TODAY - EVAL_LOOKBACK*DAY
end_date = TODAY
#%%
MODEL_DIR = rscfn(__name__,"VERSIONS/v1")
os.makedirs(MODEL_DIR,exist_ok=True)
MODEL_PTH = f"{MODEL_DIR}/.pkl"
#%%
DS_SCHEMA = "data_science"
TABOOLA_CAMPAIGN_MANAGEMENT_TABLE = "taboola_campaign_management_test"
TABOOLA_CAMPAIGN_UPDATE_TABLE = "taboola_campaign_updates_test"
TABOOLA_CAMPAIGN_TABLE = "taboola_campaign_test"