# Bing Auto Bidding Algs

This repo is created in order to store all the inhouse algs we develop related to management of Bing. We will provide descriptions of each of the files and releavant documentation to understand how they work



## Scripts currently available

- Daily keyword bidding alg: bing_keyword_bidder.py
- Daily keyword bidding alg: bing_keyword_geo_granular.py


## bing_keyword_bidder.py

##### Objective

Our objective is to create a bid every day at the keyword level in order to achieve a given ROI target.

##### Data Processing

The input is a tron profitablility in redshift. I use the following query to retrieve data then perform some data processing in python.

~~~~sql
select transaction_date as date,transaction_hour ,transaction_date_time, data_source_type, account_id as account, campaign_id, ad_group_id as adgroup_id, 
           keyword_id, campaign_name as campaign, ad_group as adgroup, keyword, 
           paid_clicks as clicks, cost, revenue as rev, match_type as match, max_cpc
           from tron.intraday_profitability
           where date >= current_date - 84 and channel = 'SEM' and traffic_source = 'BING'
~~~~

Once the data is imported I perform the following operations:

- Fix a data issue which occured due to incorrect reporting in the bing api
- Fix the match type which is tracked incorrectly where `data_source_type = "COST"`
- Create a column called `days_back` which is the number of days back in time from today
- Keep data only for the account being bid on
- Create a key to identify a unique keyword by `campaign_id` , `adgroup_id`, `keyword_id`, `match` 
- Create a list `kw_keys_with_clicks` with list of `kw_keys` that generated clicks in the previous day
- Filter data to keep either weekday or weekends. If previous day was a weekday keep only weekday data, if it was a weekend keep only weekend data.
- Create a data frame with data for the previous day called `df_bid`. This contains data for keywords which will be used in bid calculations. See "bid finalization below"
- Create data aggregations by keyword, adgroup, campaign and account to be used in the model. We will call this "data levels". We always include "match" in the aggregation so that keywords of different match types cannot share data.

#### The model

The first step is to find the decay multiplier. We decay both click and revenue data in order to weigh more recent data more. We use exponential decay with a lambda value of 0.03.

The main objective of the model is to find a revenue per click for all keywords that generated a click in the previous day.  Previous to building this model we carried out data exploration to understand after how many clicks we get a stable estimate of revenue per click. Looking at keyword level data aggregated by week, we found that with a click threshold of 120 clicks the revenue per click distribution is approximately normal and has few outliers. Thus, we will say that any keyword need to have at least this number of clicks in order for us to use it in bidding.

If a keyword which generated a click yesterday has less than this threshold we will use data from previous days or from other keywords in the adgroup, campaign or account. This is performed in the function `find rpc`. The functionality is summarised as follows:

- retrieve daily keyword, adgroup, campaign, account data (click and revenue) for a given `kw_key`
- create a data frame which contains daily data at keyword, adgroup, campaign and account level with one row per day
- in order not to double count data when we borrow from other levels, remove keyword data from adgroup data, adgroup data from campaign data and campaign data from account data. 
- apply the decay multiplier on all data levels
- next we create a unique dataframe where all levels are combined. For example, the first 84 days of kw data (clicks and revenue) will be the first 84 rows (with the first row being yesterday and the last one being 84 days ago), then next 84 days of data (rows 85 - 168) will be adgroup level data, and so on. We then find the cumulative number of clicks across the whole data frame.
- next we find the row where we reach the click threshold of 120 and discard all successive rows
- in order not to have this last row weigh too we change the number of clicks to the difference between the cumulative clicks in the previous row and the click threshold, while the revenue is also adjusted to maintain the same revenue per click obtained prior to click reduction
- finally we take the sum of the revenue column and divide by the sum of the click column to find the revenue per click

The model was validated was validated using a bootsrap approach (code still needs to be migrated from R)

- Create an empty list
- Randomly sample kw_keys with replacement
- find the difference between the observed revenue per click and the estimted revnue per click weighted by clicks
- Repeat 1000 times to derive the average error

#### Bid Change Finalization

The objective is to converge on the revenue per click. Because we have many layers of bid modifiers including time of day and geographic modifiers we cannot set the bid directly. We thus use the following approach. Inputs defined in the global variables are as follows:

- ROI_TARGET - the target ROI of the account
- MAX_PUSH - the maximum amount in dollars that we can increase the bid
- MAX_CUT - amount in dollars by which we will decrease the bid
- CPC_MIN - the minimum bid under which we will not decrease further (usually $0.05)

Using the estimated revenue per click by `kw_key` and the current bid, we use the following approach to find the new bid:

- we find the cpc target using the estimated revenue per click divided by ROI_TARGET
- we use this to find a target cpc
- then we find the difference between the target cpc and yesterday's cpc to find the bid change in dollars
- we then apply MAX_PUSH, MAX_CUT and CPC_MIN as described above to find the final bid change in dollars
- this bid change is then applied to the current bid to find the new bid

The final output is a bid change for every `kw_key`  that generated a click yesterday




## bing_keyword_geo_granular.py

#### Objective

Our objective is to create a bid every day at the keyword level in order to achieve a given ROI target. However, for the geo granular bids we want to share data across keywords with the same structure but in different states. For example "obamacare california" and "obamacare minnesota" are considered to be the same.

#### Data Processing

Data import is the same as above but data processing slightly different

Once the data is imported I perform the following operations:

- Fix a data issue which occured due to incorrect reporting in the bing api
- Fix the match type which is tracked incorrectly where `data_source_type = "COST"`
- Create a column called `days_back` which is the number of days back in time from today
- Keep data only for the account being bid on
- keey only data for keywords in the geo granular campaign
- replace state names with the term "state" both in the adgroup and keyword
- create a lookup so I can later retrieve the original keyword and campaign names
- replace `adgroup_id` with the modified adgroup name
- replace `keyword_id` with the modified keyword name
- Create a key by `campaign_id` , `adgroup_id`, `keyword_id`, `match` 
- Create a list `kw_keys_with_clicks` with list of `kw_keys` that generated clicks in the previous day
- Filter data to keep either weekday or weekends. If previous day was a weekday keep only weekday data, if it was a weekend keep only weekend data.
- Create data aggregations by keyword, adgroup, campaign and account to be used in the model. We will call this "data levels". We always include "match" in the aggregation so that keywords of different match types cannot share data.


#### retrieve bids

for now in the script i import a csv with all keywords including those with no impressions. These are not present in redshift. This should be replaced with api to bing.


#### The model

this is the same as in `bing_keyword_bidder.py`

#### Bid Change Finalization

this is the same as in `bing_keyword_bidder.py` except the final output is a bid change for every `kw_key` that generated a click yesterday. which in this case is a group of similar kewyords. We then apply bid change to all keywords that were imported from the section above named "retrieve bids"





