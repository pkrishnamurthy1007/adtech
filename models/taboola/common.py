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
from ds_utils.db.connectors import HealthcareDW
DS_SCHEMA = "data_science"
TABOOLA_CAMPAIGN_MANAGEMENT_TABLE = "taboola_campaign_management_test"
TABOOLA_CAMPAIGN_UPDATE_TABLE = "taboola_campaign_updates_test"

def upload_taboola_updates_to_redshift(updatedf):
    table_creation_sql = f"""
        CREATE TABLE IF NOT EXISTS
        {DS_SCHEMA}.{TABOOLA_CAMPAIGN_UPDATE_TABLE}
        (
            "account_id"            VARCHAR(256),
            "campaign_id"           VARCHAR(256),
            "date"                  DATE,
            "datetime"              DATETIME,
            "update"                SUPER,
            "schedule"              SUPER
        );
    """
    with HealthcareDW() as db:
        db.exec(table_creation_sql)
        db.load_df(updatedf, schema=DS_SCHEMA,table=TABOOLA_CAMPAIGN_UPDATE_TABLE)
#%%
active_camp_df = pd.read_csv(rscfn(__name__,"active_campaigns.csv"))
active_camps = active_camp_df["id"]
# activeI = campdf["attrs"]["is_active"]
# active_camps = campdf.loc[activeI].index
#%%