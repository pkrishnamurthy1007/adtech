# Media alpha Auto Bidding Algs

This repo is created in order to store all the inhouse algs we develop related to management of Media Alpha. We will provide descriptions of each of the files and releavant documentation to understand how they work



## Scripts currently available

- Daily channels bidding alg: media-alpha_channels_bidder.py


## media-alpha_channels_bidder.py

##### Objective

Our objective is to create a bid every day at the channel level in order to achieve a given ROI target.

##### Data Processing

The data is obtained with the following query in MySQL db

~~~~sql
select date,account,
	  campaign,ad_group as adgroup,
	  publisher, ma_channel as channel,
	  sum(clicks) as clicks, sum(cost) as cost,
	  sum(all_conversions) as conv,
	  sum(all_conversions_rev) as rev
from acquisition.ma_cost_conversions
group by date,account,campaign,adgroup,publisher,channel
~~~~

Once the data is imported I perform the following operations:

- Create a column called `days_back` which is the number of days back in time from yesterday
- Keep data only for the campaign being bid on
- Create a key to identify a unique channels by `campaign` , `publisher`, `channel`
- Create a list `ch_keys_with_clicks` with list of `ch_keys` that generated clicks in the previous day
- Filter data to keep either weekday or weekends. If previous day was a weekday keep only weekday data, if it was a weekend keep only weekend data.
- Create a data frame with data for the previous day called `df_bid`. This contains data for channels which will be used in bid calculations. See "bid finalization below"
- Create data aggregations by `channel`, `publisher`, `campaign` to be used in the model. We will call this "data levels".

## The model

The first step is to find the decay multiplier. We decay both click and revenue data in order to weigh more recent data more. We use exponential decay with a lambda value of 0.03.

The main objective of the model is to find a revenue per click for all channels that generated a click in the previous day.  Previous to building this model we carried out data exploration to understand after how many clicks we get a stable estimate of revenue per click. Looking at keyword level data aggregated by week, we found that with **a click threshold of 60 clicks the revenue per click distribution is approximately normal** and has few outliers. Thus, we will say that any channel need to have at least this number of clicks in order for us to use it in bidding.

If a channel which generated a click yesterday has less than this threshold we will use data from previous days or from other channels. This is performed in the function `find rpc`. The functionality is summarised as follows:

- retrieve daily channel, publisher, campaign data (click and revenue) for a given `ch_key`
- create a data frame which contains daily data at channel, publisher, campaign level with one row per day
- in order not to double count data when we borrow from other levels, remove channel data from publisher data, publisher data from campaign data
- apply the decay multiplier on all data levels
- next we create a unique dataframe where all levels are combined. For example, the first 84 days of channel data (clicks and revenue) will be the first 84 rows (with the first row being yesterday and the last one being 84 days ago), then next 84 days of data (rows 85 - 168) will be publisher level data, and so on. We then find the cumulative number of clicks across the whole data frame.
- next we find the row where we reach the click threshold of 60 and discard all successive rows
- in order not to have this last row weigh too much we change the number of clicks to the difference between the cumulative clicks in the previous row and the click threshold, while the revenue is also adjusted to maintain the same revenue per click obtained prior to click reduction
- finally we take the sum of the revenue column and divide by the sum of the click column to find the revenue per click

The model was validated was validated using a bootstrap approach (code still needs to be migrated from R)

- Create an empty list
- Randomly sample ch_keys with replacement
- find the difference between the observed revenue per click and the estimted revenue per click weighted by clicks
- Repeat 1000 times to derive the average error

## Bid Change Finalization

The objective is to converge on the cpc target. Inputs defined in the global variables are as follows:

- ROI_TARGET - the target ROI of the account
- MAX_PUSH - the maximum amount in percentage to increase bids modifiers
- MAX_CUT - the maximum amount in percentage to decrease bid modifiers

Using the estimated revenue per click by `ch_key` and the current bid model, we apply the `bid_change` to converge on the target cpc. The final output is a bid change for every channel that generated a click yesterday.







