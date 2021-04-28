### 2021-03-25
- incorporate per-lead tracking info (e.g. source page)
- polish up the featurizign e.g. properly featurize `time_modifier=="ALWAYS"`
- instead of 1 hot encoding maybe do a virtual bid modifier
- variable pairs
- figure out how to fetch the audience targeting params
- see what else is available via reports api


"""
- taboola click id 
    1. talk to alexa
    2. if param sent - talk to DE
"""

- w.t.s
  - @david mtd: weighting by decay
  - @trevors mtd:
    - e.g. rpc (rev/click)
        => avg rpc / day 
        rpc_{t-2}, rpc{t-1}, rpc{t}, 
        ~= .98,1.01,1 # basically all 1 in steady state
    - calculate geometric change per day:
      - r = rpc_{i} / rpc{i-1} 
    - weight data (e.g. revenue p_{t-3}) from 3 days ago b
      - can weight old rpc data p_{t-10} * (r_{t-10})
  - @sperk mtd:
    - how to use lead score to weight old data
    - rpc,cpc all vary over time
    - lead score (more) time invariant (if trained on recent data - or data from a small time window)
    - normalize cpc,rpc => to get same time invariance of lead score
      - assume:
        - lead score \alpha monetary val
  - @trevor (kayak)
    - kwd bidding algo
    - pid controller 
    - lookback over 3 days
    - Q: doesnt PID stabilize values?
      - whatif u wanted to increase profit/roi/columne?
    - Q: PID tuning difficult - did it require A/B tests?
    - @trevor: what if u downbid aggressivley ==> no volume => how do u inform algo?
      - problem or all algos
      - 
  - @trevor: kpis
  	- w.t. maimize rev for ROI target
  	- e.g. might want to target ROI 0-day lag, ROI 10-day lag, .... ,etc...
  	- different roi targets give different marginal profits? 
  	- Q: why aim for roi target intead of optimizing for profit?
  		- A1: e.g. LTV driven by volume 
			- A2: e.g. possibly b/c max profit assumes impossible (operational capactiy,market depth, etc....)
	- @trevor: taboola history
  	- alexa recent hire 
  	- currently has audience segmentation
  	- prev was targetting a small, possibly overmined dataset 
  	- so current cpc/roi might be worse than it could be
		=>>> sched another mtg w/ alexa - agenda - understand brax
	- @trevor: 
		- focus on platforms first
		- need bid controller first
	- Q: could using statistical power as a mask over weight tuning solve problem of training over noise?
	- Q: A/B testing - how has u seen it work in past?
  	- scenario: testing different weighting mtds @trevor vs @sperk for e.g.
  	- a) is there any way besides A/B testing uve seen work?
    	- 
  	- b) how would u do an A/B test?
    	- splitting traffic hard
    	- can make groups of locations that are roughly similar (e.g. cali vs NE)
      	- can do it more scientifically by making goups such that attrs align b/w groups over time
      	- cpc,rpc,etc.....
      	- DMA (demographic marketing area - not discrete moving avg) - from Neilson surveys
        	- rev[DMA]/rev[DMA[:]], cpc, clicks, ..... , bunch of metrics
      	- u want to find 2 test DMA groups w/ very similar metrics 
    	- splitting on GEO best
    	- could also do an on-off TOD test
	- action items
  	- @amal
    	- check out brax (accnt, api ,etc...)
    	- provides connection to many diff apis - cuts down on boilerplate codes...
    	- watch taboola call
    	- find a small project that doesnt involve splitting campaigns - see if possible
    	- watch out for internally competing campaigns

### Interval fitting
#### A*
- hueristic
  - admissible: h(n) <= actual cost of reaching goal
    - never overestimates cost to reach goal
  - consistent: h(`n`) <= traverse-cost-to(neighbor `m`) + h(`m`) \forall neightrs of `n`
    - basically the path_cost + hueristic always increases (or stays same) => monotone
    - e.g. path finding could use manhattan or L2 distance from node to goal
      - works b/c distance to goal could increase b/c of obstacles or rough terrain 
- interval fitting problem:
  - nodes:
    - state: interval boundaries (non-overlapping)
    - neighbors: may move any interval boundary left/right by 1 inc
    - h(n): 
#### DP
- bid:	current bid for this timeslot
- cpc: 	arr of cpc targets - 1 per time slot 
- cnt: 	arr of click (session) cnts - 1 per time slot   
- i: 	index of time slot we have slotted until
- k: 	intervals left to place
  
##### `OPTMSE(bid,cpc,cnt,i,k,)`
	- ? cases
  1. i >= len(cpc|cnt): we have assigned a bid level to all time slots - terminate w/ 0 MSE 
	`return 0`
  2. k = 0: we have used too many intervals - terminate w/ `inf` MSE
	`return inf`
  3. general case: we may either place an interval boundary here or continue w/ current interval
	- place interval bound by updating bid level and decrementing remaning intervals - but not advancing time slot 
	`retA = OPTMSE(cpc[i],cpc,cnt,i,k-1)`
	- extend current interval by keeping current bid level and advancing time slot 
	`retB = cnt[i]*(cpc[i] - bid) + OPTMSE(bid,rev,cnt,i+1,k)`
	- return min of options
	`return min(retA,retB)`

#### complexity disc 
- `N`: # of time slots `~360`
- `I`: # of intervals `=7`
- ***brute force***: 	N choose (I-1)
- ***DP***: 			N * N * I 
  - memoize bid level 		  `N` possible values
  - memoize time slot		    `N` possible vlaues
  - memoize intervals left	`I` possible values

### DMA split
- daily correlation important
- want roughly equal revenue
- conversion rate, ctr, ctv, 
- focus on
  - how to get taboola location in reporting
  - associate taboola reporting w/ session revenue reporting
  - A-A test over past 30 days -> A-A test over past 30 days fitted over 60-30 days
  - if i want SEO - could try to only use sessoins where referrer == GOOGLE
    - b/c if referrer == NULL  could have come for somewhere else

### 2021-04-23 standup
- A/B testing split
  - reporting:
    - is `session.creation_date` a good approximation of click date
    - can multiple sessions come out of 1 click

  - loc
    - after convo w/ Alexa and @milton
      - user location not available via taboola tracking params 
      - encrypted cpc is though ---- which could simplify things
      - TOD also available
    - vulnerable to changes in location targetting and possible geoip mismatch 
    - requires some level of fitting 
    - still worth looking at if we ever run into a system where bids cant be modified in realtime easily (bing maybe?)

  - on/off
    - robust against TOD targetting regardless of what happened in past b/c groups will cycle their positions in 5 days - so an even 6 times a month
    - divided into groups via ((dt - EPOCH) // 2.5) % group_num
    - then if u want bigger time groupings u can go group_num % (whatever group num u want)

  - loc vs on/off
    - i mean on/off seems way easier right?

  - evaluation metrics
    - definitely shouldnt be filtering/smoothing/dealing w/ outliers when computing eval metrics right?
    - daily correlation vs AA testing?
      - 
    - why not rolling (weekly,2-weekly,monthly) correlation?
    - or rolling (weekly,2-weekly,monthly) AA testing?

    - AA testing daily aggregated kpis vs click level kpis?
      - AA test where u have 1 observeration per day => what i did
        - used by @trevor at kayak
      - AA test where 1 observatoin per conversion => bag method
        - used by @trevor at wanderoo at rpc estimation
      - AA tes where 1 oberversion per session => ??? didnt this , 
        - effect variable is then binary [0,1]

    - is there some kind of metric that would determine the difficulty of running a test off the groups we create
      - bayesian A/B testing w/ priors 
      - A/B testing priors, variancae, groups, 

    - AA significance testing vs "negative-power"
      - positive power gives p(detect real effect | there is a real effect)
      - negative power gives p(find no effect | there is no effect)
    
  - in general
    - are we going to have an A/B testing table/object?
      - sagemaker experiemtns/A/B testing afaict cant do this for us
      - should support creating/running/stopping/deleting/monitoring tests
      - will it be responsible for making the account structure and deployment changes to run the tests?
      - or will that happen elsewhere and this is more of a logging/retrospective tool?
      - could have separate classes for A/B and loc splitting

    - would be nice to have a plugin that provides a standardized interface for all platforms
      - at least supporting bid modifiers & blocking for  
        - TOD
        - loc
        - device type
      - then each adapter could provide the extra platform specific functionality like e.g. keyword/publisher targetting for bing/taboola

### Trevor 2021-04-23
- most granular taboola rerporting
  - date, hour of day, campaign, publisher, country, region, dma, os, platform, browser, audience, cost, clicks, impressions, viewable_impressions
  - bag mtd


### Dan 2021-04-27
- params
  - click thresh
  - roi_target
  - daily change/day
- automating the different param runs
- automating curtis checks
  - get @curtis on the horn 

### Dan/David 2021-04-28
- high level overview of how geo-granular bidding different than regular bid modifieres
- whats curtis's process for verifying bids
  - ask @curtis
  - what ideas does he have going forward
- want to consider kws like Obamacare-Californa and Obamacare-Wisconsin to be considered the same
  - Q: is curtis writign these keywords? - does bing prefine them? - is there a list of keywords somewhere?
    - TODO: ask @curtis
- TODO: exclude Geo-granular campaigns from regular bids and vice versa
- TODO: use more broad search?
- TODO: how loose is "exact match" - ask curtis
  - sep kws into different intents - how done?
- 