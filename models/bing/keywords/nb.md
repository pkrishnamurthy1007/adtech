### Curtis 2021-04-30
- audience reporting
  - it is possible - TODO: ask Byron,Milton
  - target_id: from google - maybe something for bing
- priorities for targetting
  - accnt,camp,adgp,kw,match 
  - TOD
  - location
  - device
  - audience
  - gender
  - age
  - occupation
- keyword creation 
  - diff than search term 
  - 3 types
    - exact
    - phrase
    - broad
      - broad match modified
  - e.g. cobra insurance south carolina
    - exact matches - mispelling, reordering, semantic match
    - phrase match - cobra insurance in my area, cobra match policies
    - broad match modified (what we use) - 
- validation 
  - WTS
    - cost up if we exceed ROAS - cost down if we miss ROAS
    - sort by clicks descending
      - dont want to see a lot of keywords up there w/ low ROAS
      - this only yest data @curtis will then go chceck e.g. 30 day ROAS 
        - TODO: WTS 7day, 14day, 30 day , 60 day ROAS per kw
    - compare overall cost change and expected revenue change
      - assume revenue (volume) proportional to cost
        - NOTE: bing has a way to better model this
      - want to see percentage cost change lower than percentage rev change 
  - TODO: have a way to let kws escape the low cpc trap
  - TODO: retrieve bings bid to volume estimates
  - TODO: bing ads explorer / microsoft ads explorer

### gh action
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519" pip install -r requirements.txt

sagify_base/build.sh  . . sagify_base/Dockerfile requirements.txt latest data-science-bing 3.6

src/sagify_base/build.sh  src src src/sagify_base/Dockerfile requirements.txt latest data-science-bing 3.6

### standup 2021-05-07
- geo-granular kw bids vary accross location w/ same geo-granular kw