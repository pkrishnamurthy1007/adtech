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
o65_camp_svc = CampaignService(client,O65_ACCNT_ID)
test_camp_svc = CampaignService(client,TEST_ACCNT_ID)
test_accnt_camps = aid2cid2camp[TEST_ACCNT_ID].values()
test_accnt_camp_names = {c["name"] for c in test_accnt_camps}
print("|test_accnt_camps|:",len(test_accnt_camps))
def copy_campaign_base(camp,to=TEST_ACCNT_ID):
  taboola_req_fields = [
    "name","branding_text","cpc","spending_limit",
    "spending_limit_model","marketing_objective"]
  camp_svc = CampaignService(client,to)
  camp_base = {f: camp[f] for f in taboola_req_fields}
  camp_copy = camp_svc.create(**camp_base)
  print(f"created base for `{camp['name']}` in account `{to}` w/ id:", camp_copy["id"])
  camp_copy = camp_svc.update(camp_copy["id"], is_active=False)
  assert camp_copy["is_active"] == False
  print("...succesfully deactivated newly created campaign")
  return camp_copy
for camp in active_camps:
  # skip already created test account campaigns
  if camp["name"] in test_accnt_camp_names: 
    print(f"Campaign `{camp['name']}` already exists in account `{TEST_ACCNT_ID}`")
  else:
    camp_copy = copy_campaign_base(camp,to=TEST_ACCNT_ID)
#%%
#%%
[c["status"] for c in test_accnt_camps]
#%%
def get_camp_by_name(name,accnt=TEST_ACCNT_ID):
  # TODO: figure out some way to avoid having to refetch all campaigns
  #       => will be able to use catalog tables eventually
  camp_svc = CampaignService(client, accnt)
  camps = camp_svc.list()
  match_camps = [c for c in camps if c["name"] == name]
  assert len(match_camps) == 1, (name,len(match_camps))
  return match_camps[0]
import pytaboola
pytaboola.errors.TaboolaError
import traceback
import tqdm
field_update_attempts = {}
field_update_errors = {}
def update_fields(camp,accnt,camp_id):    
  camp_svc = CampaignService(client, accnt)
  for k,attr in tqdm.tqdm([*camp.items()]):
    # print(f"attempting to copy over field `{k}`...")
    try:
      camp_svc.update(camp_id,**{k: attr})
      field_update_attempts.setdefault(k,[]).append(1)
      # print("...Sucess!")
    except (Exception,pytaboola.errors.TaboolaError) as ex:
      errd = {
        "ex": ex,
        "type": type(ex),
        "str": str(ex),
        "tb": traceback.format_exc()
      }
      # print(f"Failed w/ {type(ex)}:{ex}\n{traceback.format_exc()}")
      # print(f"Failed w/ {type(ex)}:{ex}")
      field_update_attempts.setdefault(k, []).append(0)
      field_update_attempts.setdefault(k, []).append(errd)

for camp in active_camps:
  dest_camp = get_camp_by_name(camp["name"])
  dest_id = dest_camp["id"]
  print("UPDATEING",camp["name"],dest_id)
  update_fields(camp,TEST_ACCNT_ID,dest_id)

update_attempt_df = pd.DataFrame(field_update_attempts)
update_attempt_df.mean(axis=0)
#%%
"""
TODO: first taboola test
- what to test?
  1. time of day modifiere
  2. location modifiers
  3. publisher modifiers
- testing infrastructure
  - different accounts for A/B groups? - would allow easier test management
    - these endpts may be useful for bulk updates of `is_active` and `location_targetting`:
      https://developers.taboola.com/backstage-api/reference#bulk-update-campaigns
  - flag in campaign catalog table indicating what group the campaign belogns to
- duplicate campaigns in test account: https://developers.taboola.com/backstage-api/reference#duplicate-a-campaign
  - refreshed via hourly or 1/2/4/6/8/12x daily ETL 
    -figure out what fields are writable and write all of those fields
    - NOTE: think this may not pick up stuff like `audience_targetting`
    - also - was getting errors writing our publisher targetting
      - may have been timeout errors - but need to check       
  - how are chagnes propagated when we have many campaigns?
    - probably dont want to use name to indicate structure
    - maybe some kind of tree structure in catalog tables
    - can skip for now
- cammpaign management daemon
  - only needed if we want to do TOD modifiers or on/off testing
- modifier script
  - compute:
    - publisher modifier - already written
    - location modifiers - very similar to publisher modifier 
    - time-of-day modfiiers 
    - run daily or weekly
  - validate:
    - TODO - sync w/ Alexa
  - apply
    - TODO
    - should this be done by Dan?
    - might have to consider catalog tables and A/B flags 
    - for TOD modifiers would be hitting a table consumed by campaign management daemon
- testing procedure
  - test setup
    - create duplicate campaigns and sets up ETL to maintain them
    - marks cmapaigns as A or B in catalog tables
    - sets up scheduled modifier script
    - maybe makes some entry in an A/B table
  - test-teardown
    - ?
"""
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
