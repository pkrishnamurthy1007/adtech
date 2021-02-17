# Bing API Interaction

## Authentication

A helper file (`client_helper.py`) includes all the needed methods to authenticate a
client with the Bing API

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




