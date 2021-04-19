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