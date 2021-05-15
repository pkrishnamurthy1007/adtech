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

TODAY = datetime.datetime.now().date()
# TODAY = "2021-05-12"
todays_output = glob.glob(f"{OUTPUT_DIR}/**/*{TODAY}.csv")
df_out = pd.concat((pd.read_csv(fpth) for fpth in todays_output))
old_len = df_out.__len__()
df_out = df_out.drop_duplicates() 
assert df_out.__len__() * 2 == old_len, """
We should have 2 records for each kw b/c we break bids out by account and write them,
but we also write the bids for all accounts.
"""

# Debug requests and responses
LOGLEVEL = logging.WARN
logging.basicConfig(level=LOGLEVEL)
# logging.getLogger('suds.client').setLevel(logging.DEBUG)
# logging.getLogger('suds.transport.http').setLevel(logging.DEBUG)

# import sys
# sys.exit(0)

DRY=True
def get_kw_bids(adgp,keyword_ids):
    kws = accnt_client.get_keywords(adgp,keyword_ids)
    kw_bids = {kw["Id"]: kw.Bid.Amount for kw in kws["Keywords"]["Keyword"]}
    return kw_bids
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
        refresh_token=bing_creds['BING_REFRESH_TOKEN'],
        loglevel=LOGLEVEL,
    )
    accntI = df_out["account"] == accnt

    import tqdm
    adgps = df_out.loc[accntI, "adgroup_id"].unique()
    print(f"got {len(adgps)} adgroups for account# {accnt}")
    for adgp in tqdm.tqdm(adgps):
        adgpI = df_out["adgroup_id"] == adgp
        keyword_ids, keyword_bids = df_out \
            .loc[accntI & adgpI, ['keyword_id', "max_cpc_new"]].values.T
        keyword_ids = [*keyword_ids.astype(int)]
        keyword_bids = [*keyword_bids.astype(float)]
        old_kw_bids = get_kw_bids(adgp, keyword_ids)
        response = accnt_client.update_keyword_bids(
            adgp, keyword_ids, keyword_bids, dry=DRY)
        new_kw_bids = get_kw_bids(adgp, keyword_ids)
        if DRY:
            assert new_kw_bids==old_kw_bids, (old_kw_bids, new_kw_bids)
        else:
            assert new_kw_bids==dict(zip(keyword_ids,keyword_bids)),(new_kw_bids,keyword_ids,keyword_bids)
#%%