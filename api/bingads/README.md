# Bing API Interaction

## Authentication

You need a `config.json` file with the following fields in order to authenticate properly:

```json
{
    "BING_ACCOUNT_ID": "",
    "BING_CLIENT_ID": "",
    "BING_CUSTOMER_ID": "",
    "BING_DEVELOPER_TOKEN": "",
    "BING_REFRESH_TOKEN": ""
}
```

## Default test

The default test loads a CSV file (at `data/bids.csv`) and updates the bids for specific keywords in specific adgroups.

The CSV file must contain at least the following columns:

* `adgroup_id`
* `keyword_id`
* `bid`

Updates are done in bulk for each adgroup. A success status is printed for each adgroup being processed.

## Campaign creation

An ID is returned.

Valid values for the `CampaignType` field:

* `Audience`
* `DynamicSearchAds`
* `Search`
* `Shopping`

**Note:** The `CampaignType` field cannot be updated.

Default value being `Search`. This field is **case sensitive**.

Valid values for the `Status` field:

* `Active`
* `Paused`

Default being `Active`.

**Note:** The field `Language` once udpated, does not append values to the list
of languages, but rather, raplaces the older list with the new one. Additionally,
you cannot delete all languages.

## Adgroup creation

Valid values for the `Status` field:

* `Active`
* `Expired`
* `Paused`

### Adgroup modifications

The main modifications that can be made is editing the bid value `CpcBid` (which takes a
double value) and the `StartDate` and `EndDate` values. Additionally, target audience criteria 
can be added. The `BiddingScheme` field will most likely have to be `ManualCpcBiddingScheme`,
but here are the other valid values:

* `EnhancedCpcBiddingScheme`
* `MaxClicksBiddingScheme`
* `MaxConversionsBiddingScheme`
* `TargetCpaBiddingScheme`
* `TargetRoasBiddingScheme`
* `ManualCpcBiddingScheme`

#### Target Audience Criteria

Valid criteria values:

* `AgeCriterion` x
* `AudienceCriterion`
* `DayTimeCriterion`
* `DeviceCriterion` x
* `GenderCriterion` x
* `LocationCriterion` x
* `LocationIntentCriterion`
* `ProductScope`
* `ProductPartition`
* `ProfileCriterion`
* `RadiusCriterion`
* `StoreCriterion`
* `Webpage`

However, only the marked ones are supported right now.

Additionally, the multiplier's format is a percentage, i.e. you need to use `300` instead of `3` if you want
the bid to up by a 300%.


Age criteria valid values:

* `EighteenToTwentyFour`
* `TwentyFiveToThirtyFour`
* `ThirtyFiveToFortyNine`
* `FiftyToSixtyFour`
* `SixtyFiveAndAbove`
* `Unknown`

Valid **location** IDs:

A CSV file will be downloaded with the valid IDs. The file's name will be
`geographicallocations.csv` and will be located at the root directory. This file
includes location IDs at a country, state, or city level.

Device criteria valid values:

* `Computers`
* `Smartphones`
* `Tablets`

**NOTE:** All wanted devices must be added, whatever is missing will have a bid adjustment of **zero**.




