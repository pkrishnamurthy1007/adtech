#%%
import requests
import base64
import os
BRAX_URI = "https://api.brax.io/v1"
USER = os.getenv("BRAX_USER")
PSWD = os.getenv("BRAX_PSWD")
TOK = base64.b64encode(f"{USER}:{PSWD}".encode("ascii")).decode()
BRAX_KWARGS = {
    "headers": {
        "Authorization": f"Basic {TOK}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
}

OUTBRAIN = "outbrain"
TABOOLA = "taboola"
REVCONTENT = "revcontent"
YAHOO = "yahoo"
CONTENT_AD = "content-ad"
SOURCES = {OUTBRAIN,TABOOLA,REVCONTENT,YAHOO,CONTENT_AD}
def _get_params(kwargs):
    kwargs = {
        **kwargs,
        **kwargs.get("pagination_kwargs",{}),
        "pagination_kwargs": None
    }
    params = {k: v for k, v in kwargs.items() if v}
    source = params.get("source")
    assert not source or source in SOURCES
    return {k: v for k, v in params.items() if v}

def get_accounts(**pagination_kwargs):
    params = _get_params(locals())
    r = requests.get(f"{BRAX_URI}/accounts",params=params,**BRAX_KWARGS)
    r.raise_for_status()
    return r.json()

def get_campaigns(source_account_id=None, source_campaign_id=None, **pagination_kwargs):
    params = _get_params(locals())
    r = requests.get(f"{BRAX_URI}/campaigns", params=params, **BRAX_KWARGS)
    r.raise_for_status()
    return r.json()

def get_ads(source_account_id=None,source_campaign_id=None,**pagination_kwargs):
    params = _get_params(locals())
    r = requests.get(f"{BRAX_URI}/ads", params=params, **BRAX_KWARGS)
    r.raise_for_status()
    return r.json()


#%%
O65_ACCNT_ID = "taboolaaccount-rangaritahealthcarecom"
accnts = get_accounts(source=TABOOLA)["results"]
id2accnt = {a["source_account_id"]: a for a in accnts}
#%%
accnt = id2accnt[O65_ACCNT_ID]
camps = get_campaigns(source_account_id=O65_ACCNT_ID,source=TABOOLA)["results"]
#%%
camp = camps[0]
#%%
ads = get_ads(accnt["source_account_id"])["results"]
ads
#%%
accnt["source_account_id"]
#%%
get_campaigns(source_account_id=aid)
#%%
camps[0]
#%%

#%%
ads = get_ads(accnts[0]["id"],source=TABOOLA)["results"]
#%%
camps = get_campaigns(accnts[0]["id"])["results"]
#%%
print(r.json())

# %%
