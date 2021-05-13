#%%
# hc creds
hccreds = {
    "client_id": "3cc90f549fd44a9fbe36f70aa6e7cfde",
    "client_secret": "eaa74a94108648579e65167bf276c3a1",
}
# pivot creds
pivotcreds = {
    "client_id": "31d7f068eb3d43acb85b5ad64eef2ca9",
    "client_secret": "c5010044df194010817e76edac0fd426",
}

from pytaboola import TaboolaClient
client = TaboolaClient(**hccreds)
client.token_details

from pytaboola.services import AccountService
from pytaboola.services import CampaignService

import itertools
import tqdm
import pandas as pd
import os
from pkg_resources import resource_filename as rscfn


acct_service = AccountService(client)
accnts = acct_service.list()["results"]
NETWORK_ACCNT_ID = "healthcareinc-network"
TEST_ACCNT_ID = "healthcareinc-sc2"
O65_ACCNT_ID = "taboolaaccount-rangaritahealthcarecom"
id2accnt = {a["account_id"]: a for a in accnts}

def accnt_camps(accnt):
    camp_service = CampaignService(client, accnt["account_id"])
    return camp_service.list()
aid2cid2camp = {}
for aid,a in tqdm.tqdm(id2accnt.items()):
    cid2camp = {c["id"]: c for c in accnt_camps(a)}
    aid2cid2camp[aid] = cid2camp

import json
json.dump(aid2cid2camp, open("camps.json", "w"))
O65_accnt_camps = accnt_camps(id2accnt[O65_ACCNT_ID])
print("|065_accnt_camps|:",len(O65_accnt_camps))
#%%
import itertools
cross = itertools.product
import jmespath
get = jmespath.search

active_camps = get(
    "*.*[] | [?is_active]",
    aid2cid2camp,
)
print("|active campaigns|:",len(active_camps))
#%%
1/0
o65_camp_svc = CampaignService(client,O65_ACCNT_ID)
test_camp_svc = CampaignService(client,TEST_ACCNT_ID)
test_accnt_camps = aid2cid2camp[TEST_ACCNT_ID].values()
test_accnt_camp_names = {c["name"] for c in test_accnt_camps}
print("|test_accnt_camps|:",len(test_accnt_camps))
taboola_req_fields = [
    "name","branding_text","cpc","spending_limit",
    "spending_limit_model","marketing_objective"]
for camp in active_camps:
    # skip already created test account campaigns
    if camp["name"] in test_accnt_camp_names: 
        print(f"Campaign `{camp['name']}` already exists in account `{TEST_ACCNT_ID}`")
    camp_base = {f: camp[f] for f in taboola_req_fields}
    camp_copy = test_camp_svc.create(**camp_base)
    print(f"created base for `{camp['name']}` in account `{TEST_ACCNT_ID}` w/ id:", camp_copy["id"])
    updated_camp = test_camp_svc.update(camp_copy["id"], is_active=False)
    assert updated_camp["is_active"] == False
    print("...succesfully deactivated newly created campaign")
#%%
import pytaboola
pytaboola.errors.TaboolaError
#%%
import traceback
for k,attr in camp.items():
    print(f"attempting to copy over field `{k}`...")
    try:
        test_camp_svc.update(updated_camp["id"],**{k: attr})
        print("...Sucess!")
    except (Exception,pytaboola.errors.TaboolaError) as ex:
        print(f"Failed w/ {type(ex)}:{ex}\n{traceback.format_exc()}")
#%%
#%%
from pytaboola.services.report import \
    CampaignSummaryReport, RecirculationSummaryReport, \
    TopCampaignContentReport, RevenueSummaryReport, VisitValueReport
import datetime

DAY = datetime.timedelta(days=1)
MONTH = 30 * DAY
YEAR = 365 * DAY
NOW = datetime.datetime.now()

"""
Dimension 10 is not allowed for CampaignSummaryReport. Must be one of 
('day', 'week', 'month', 'content_provider_breakdown', 
'campaign_breakdown', 'site_breakdown', 'country_breakdown', 
'platform_breakdown', 'campaign_day_breakdown', 'campaign_site_day_breakdown')
"""
# for accnt in accnts:
#     d = CampaignSummaryReport(client, accnt["account_id"]) \
#         .fetch("site_breakdown", NOW-DAY, NOW)
#     print(accnt["name"], len(d["results"]))

# #%%
# # cmpaigns always on 
# import jmespath
# get = jmespath.search
# # evercamps = get("[?activity_schedule.mode=='ALWAYS' && is_active].[activity_schedule.mode,is_active]", camps)
# evercamps = get(
#     "[?activity_schedule.mode=='ALWAYS' && is_active]", camps)
# c = evercamps[0]
# c
# [*yield_time_rules(camps[0])]
# # [*yield_time_rules(evercamps[0])]

import itertools
cross = itertools.product
import jmespath
get = jmespath.search
#%%

#%%
