#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAILY BIDDING ALGORITHM FOR MEDIA ALPHA
"""


# import libraries
import numpy as np
import pandas as pd
import datetime as dt
import math
#import os
#os.getcwd()

#### DEFINE GLOBAL VARIABLES ####

CLICKS = 60 # click threshold. level at which kw uses all of its own data
ROI_TARGET = 1.2 # target we are aiming
MAX_PUSH = 0.2
MAX_CUT = -0.2
TODAY = dt.date.today()
DAYS_BACK = 1
WEEKDAY =  dt.date.today().weekday() 
# CAMPAIGN = "U65"
CAMPAIGN = "O65"

#### IMPORT DATA ####
df = pd.read_csv("ma.csv")


#### PROCESS DATE ####

# data cleaning
df['date'] = pd.to_datetime(df['date'])
YESTERDAY = df['date'].max()

#find days back from today
df['yesterday'] = YESTERDAY
df['yesterday'] = pd.to_datetime(df['yesterday'])
df['days_back'] =  df['yesterday'] - df['date']
df['days_back'] = df['days_back'].astype('timedelta64[D]').astype('int')


#### DATA MUNGING AND CLEANING ####

#choose under vs over
df = df[df['campaign'].str.contains(CAMPAIGN)]


# set clicks to numeric
df['clicks'] = df['clicks'].astype('float64')

# create a key
df2 =  df[["campaign","pub","channel"]].drop_duplicates()
ch_key = range(0, df2.shape[0])
df2["ch_key"] = range(0, df2.shape[0])
join_key = ["campaign","pub","channel"]
df = df.merge(df2, left_on=join_key, right_on = join_key, how='left')
#df = df.set_index("kw_key")

#kw_keys with clicks yesterday
ch_keys_with_clicks = list(df["ch_key"][(df["clicks"]>0) & (df["date"] == YESTERDAY)].unique())

#filter data in order not to mix weekday and weekend data
df["weekday"] =  df['date'].dt.weekday
day_of_week_today = TODAY.weekday()

if day_of_week_today == 0:
    df = df[df["weekday"]>=5]
if day_of_week_today == 6:
    df = df[df["weekday"]>=5]    
if (day_of_week_today < 5) & (day_of_week_today>0):
    df = df[(df["weekday"]<5)]    


#### PREPARED YESTERDAY DATA FOR USE IN BID CHANGES ####

#store yesterday data for kws with clicks yesterday for later use in bid calcs
aggregations = { "clicks":"sum" , "rev":"sum", "cost":"sum"}    
df_bid = df[df["date"] == YESTERDAY]      

df_bid = df_bid.groupby([ "ch_key","campaign","pub","channel"]).agg(aggregations).reset_index()   

#keep only kws with clicks yesterday
df_bid = df_bid[df_bid["ch_key"].isin(ch_keys_with_clicks)]


#### CREATE AGGREGATIONS FOR USE IN RPC ###

# simplify df
aggregations = { "clicks":"sum" , "rev":"sum", "cost":"sum"}    
df = df.groupby([ "ch_key","days_back","campaign","pub","channel"]).agg(aggregations).reset_index()   
 
#find pub level
aggregations = { "clicks":"sum" , "rev":"sum"}    
df_pub = df.groupby(["days_back","campaign","pub"]).agg(aggregations).reset_index()  
df_pub.rename(columns = {'clicks':'clicks_pub', 'rev':'rev_pub'}, inplace = True) 

#find campaign level
df_cmp = df.groupby(["days_back","campaign"]).agg(aggregations).reset_index()   
df_cmp.rename(columns = {'clicks':'clicks_cmp', 'rev':'rev_cmp'}, inplace = True) 


         
#### FIND DECAY MULTIPLIER ####

#to be used to decaty revenue, click data in our recency model

#find the decay multiplier
decay_factor = 0.03 
df_days_back = df["days_back"].unique() -1
days_back_list = df_days_back.tolist()
decay_multiplier_list = []
for days_back in days_back_list:
    decay_multiplier = math.exp( -decay_factor * days_back )
    decay_multiplier_list.append(decay_multiplier)


#### FUNCTION FOR RPC CALCULATION ####

def find_rpc(x):    
    
    #retrieve kw
    df2 = df[df["ch_key"]==x]
    
    # #make sure all days are present
    join_key = ["days_back"]
    df_days_back = pd.DataFrame({"days_back":df["days_back"].unique()})
    df2 = pd.merge(df_days_back,df2,how="left",on=join_key)
    
    # join adgroup data
    join_key = ["days_back","campaign","pub"]
    df2 = pd.merge(df2,df_pub,how="left",on=join_key)
    
    #join campaign data
    join_key = ["days_back","campaign"]
    df2 = pd.merge(df2,df_cmp,how="left",on=join_key)
        
    #remove clicks and rev from successive levels
    df2["clicks_cmp"] = df2["clicks_cmp"]-df2["clicks_pub"]
    df2["clicks_pub"] = df2["clicks_pub"]-df2["clicks"]
    
    df2["rev_cmp"] = df2["rev_cmp"]-df2["rev_pub"]
    df2["rev_pub"] = df2["rev_pub"]-df2["rev"]
    
    
    # # decay the data
    df2 = df2.sort_values("days_back")
    df2["decay_multiplier"] = decay_multiplier_list
    
    df2["clicks"] = df2["clicks"] * df2["decay_multiplier"] 
    df2["clicks_pub"] = df2["clicks_pub"] * df2["decay_multiplier"] 
    df2["clicks_cmp"] = df2["clicks_cmp"] * df2["decay_multiplier"] 
    
    df2["rev"] = df2["rev"] * df2["decay_multiplier"] 
    df2["rev_pub"] = df2["rev_pub"] * df2["decay_multiplier"] 
    df2["rev_cmp"] = df2["rev_cmp"] * df2["decay_multiplier"] 
    
    # #create 1 table with combined data
    clicks_arr = df2["clicks"].to_numpy()
    clicks_arr = np.append(clicks_arr, df2["clicks_pub"].to_numpy())
    clicks_arr = np.append(clicks_arr, df2["clicks_cmp"].to_numpy())
    
    rev_arr = df2["rev"].to_numpy()
    rev_arr = np.append(rev_arr, df2["rev_pub"].to_numpy())
    rev_arr = np.append(rev_arr, df2["rev_cmp"].to_numpy())
    
    clicks_arr = np.append(clicks_arr, df["clicks"].sum())
    rev_arr = np.append(rev_arr, df["rev"].sum()/2)
    
    df3 = pd.DataFrame({'clicks': clicks_arr, 'rev': rev_arr})
    df3 = df3.fillna(0)
    
    #find cumulative clicks
    df3["clicks_cum"] = df3["clicks"].cumsum()
    
    # #choose what data to use based on click thresold
    clicks_cum = df3["clicks_cum"].to_numpy()
    max_row = np.searchsorted(clicks_cum,CLICKS)
    
    df3 = df3.loc[:max_row]
    
    
    # limit number of clicks in last row
    if(df3.shape[0]>1):
        rpc_last = df3["rev"].loc[max_row]/df3["clicks"].loc[max_row] 
        clicks_previous = df3["clicks"].loc[:max_row-1].sum()
        clicks_last = CLICKS - clicks_previous
        rev_last = rpc_last * clicks_last
        df3["rev"].loc[max_row] = rev_last
        df3["clicks"].loc[max_row] = clicks_last
    
        
    #find and output rpc
    rpc = df3["rev"].sum()/df3["clicks"].sum()

    return(rpc)


#### FIND RPC ####

rpc_est = []

for ch_key in ch_keys_with_clicks:
    rpc_est.append(find_rpc(ch_key))

df_rpc = pd.DataFrame()
df_rpc["ch_key"] = ch_keys_with_clicks
df_rpc["rpc_est"] = rpc_est

#### FINALIZE BIDS ####

# reset index prior to merge
df_rpc = df_rpc.reset_index(drop=True)
df_bid = df_bid.reset_index(drop=True)

# merge rpc with bids
df_bid = pd.merge(df_bid,df_rpc,how="left",on=["ch_key"]).drop_duplicates()
df_bid["cpc_y"] = df_bid["cost"]/df_bid["clicks"]

#find cpc target
df_bid["cpc_target"] = df_bid["rpc_est"]/ROI_TARGET
df_bid["bid_change"]=  df_bid["cpc_target"]/df_bid["cpc_y"] -1

# then apply change to the current bid modifier
# currently pubslisher and channel bid mods not present in db
