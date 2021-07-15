#%%
import numpy as np
import typing

def wavg(V, W):
    if W.sum() == 0:return np.NaN
    if V.shape.__len__() > 1:
        return (V*W.values.reshape(-1,1)).sum() / W.sum()
    else:
        return (V*W).sum() / W.sum()
def wvar(V,W):
    mu = wavg(V,W)
    var = wavg((V - mu)**2,W)
    return var
def wstd(V,W):
    return wvar(V,W)**0.5
def get_wavg_by(df, col):
    def wavg_by(V):
        return wavg(V, W=df.loc[V.index, col])
    return wavg_by
def get_wstd_by(df, col):
    def wstd_by(V):
        return wstd(V, W=df.loc[V.index, col])
    return wstd_by
#%%
GOOGLE = "GOOGLE"
BING = "BING"
PERFORMMEDIA = "PERFORMMEDIA"
MEDIAALPHA = "MEDIAALPHA"
FACEBOOK = "FACEBOOK"
SUREHITS = "SUREHITS"
TABOOLA = "TABOOLA"
YAHOO = "YAHOO"
MAJOR_TRAFFIC_SOURCES = [
    GOOGLE,BING,PERFORMMEDIA,MEDIAALPHA,FACEBOOK,
#     SUREHITS,
    TABOOLA,
#     YAHOO,
]

U65 = "HEALTH"
O65 = 'MEDICARE'
PRODUCTS = [U65,O65]
#%%