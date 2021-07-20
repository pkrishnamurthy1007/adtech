#%%
import requests
import datetime

OWNER = "healthcarecom"
REPO = "adtech"
PAT = "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"

def trigger_workflow(workflow_id,kwargs):
    resp = requests.post(
        f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{workflow_id}/dispatches",
        json={
            "ref": "workspace_amal",
            "inputs": {
                **kwargs,
            }
        },
        headers={
            'Authorization': f'token {PAT}'
        },
    )
    resp.raise_for_status()
    return resp

def bing_update_bids_keyword(execution_date):
    return trigger_workflow(
        "bing_update_bids_keyword.yml",
        {
            "execution_date": execution_date
        }
    )

def bing_update_bids_tod(execution_date):
    return trigger_workflow(
        "bing_update_bids_tod.yml",
        {
            "execution_date": execution_date
        }
    )

def taboola_update_bids(execution_date):
    return trigger_workflow(
        "taboola_update_bids.yml",
        {
            "execution_date": execution_date
        }
    )
#%%
# import requests
# import datetime
# """
# https://api.github.com/repos/octocat/hello-world/actions/workflows/42/dispatches \
# """
# OWNER = "healthcarecom"
# REPO = "adtech"
# WORKFLOW_ID = "bing_update_bids_keyword.yml"
# PAT = "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"
# resp = requests.post(
#     f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/{WORKFLOW_ID}/dispatches",
#     json={
#         "ref": "workspace_amal",
#         "inputs": {
#             "execution_date": datetime.datetime.now().date().__str__()  
#         }
#     },
#     headers={
#         'Authorization': f'token {PAT}'
#     },
# )
# resp.raise_for_status()
# #%%
