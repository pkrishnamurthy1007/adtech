import bingads

import os.path
import urllib
from typing import Optional

from .client_helper import set_elements_to_none

from bingads import OAuthDesktopMobileAuthCodeGrant, AuthorizationData, ServiceClient

VALID_AGES = ["EighteenToTwentyFour",
                "TwentyFiveToThirtyFour",
                "ThirtyFiveToFortyNine",
                "FiftyToSixtyFour",
                "SixtyFiveAndAbove",
                "Unknown"]

CAMPAIGN_STATUSES = ["Active", "Paused"]
ADGROUP_STATUSES = ["Active", "Expired", "Paused"]
MATCH_TYPES = ["Exact", "Broad", "Phrase"]
CRITERION_STATUSES = ["Active", "Paused", "Deleted"]
DEVICES = ["Computers", "Tablets", "Smartphones"]
GENDERS = ["Female", "Male", "Unknown"]

# Based on the client used for ETLs
class LocalAuthorizationData(AuthorizationData):
    def __init__(self, account_id: int, customer_id: int, developer_token: str, client_id: str,
                 refresh_token: Optional[str]):
        super().__init__(account_id, customer_id, developer_token)

        auth = OAuthDesktopMobileAuthCodeGrant(client_id)
        auth.state = ''
        auth.token_refreshed_callback = lambda x: None

        if refresh_token is not None:
            auth.request_oauth_tokens_by_refresh_token(refresh_token)

        self.authentication = auth

import logging
from bingads.v13.bulk import *
from pkg_resources import resource_filename as rscfn
import datetime
BING_LOG_DIR = rscfn(__name__, "BING_LOGS")
os.makedirs(BING_LOG_DIR, exist_ok=True)
NOW = datetime.datetime.now()
class BingClient(LocalAuthorizationData):
    def __init__(self,
                 account_id: int,
                 customer_id: int,
                 dev_token: str,
                 client_id: str,
                 refresh_token: Optional[str],
                 env: Optional[str] = 'production',
                 loglevel=logging.WARNING):
        super().__init__(account_id, customer_id, dev_token, client_id, refresh_token)
        self.campaign_id = None
        self.adgroup_id = None
        self.campaign_service = ServiceClient(
            service='CampaignManagementService',
            version=13,
            authorization_data=self,
            environment=env,
        )

        self.bulk_service = BulkServiceManager(
            authorization_data=self,
            working_directory=BING_LOG_DIR,
            environment=env)

        if not os.path.isfile('geographicallocations.csv'):
            response = self.campaign_service.GetGeoLocationsFileUrl(
                Version='2.0',
                LanguageLocale='en')
            urllib.request.urlretrieve(
                response.FileUrl, 'geographicallocations.csv')

        logger = logging.getLogger("HCBingClient")
        logger.setLevel(loglevel)
        self.logger = logger

    def get_campaigns(self, campaign_type: Optional[str] = 'Audience'):
        response = self.campaign_service.GetCampaignsByAccountId(
            AccountId=self.account_id,
            CampaignType=campaign_type)

        return response

    def get_campaign_by_id(self, campaign_id, campaign_type: Optional[str] = 'Audience'):
        response = self.campaign_service.GetCampaignsByIds(
            AccountId=self.account_id,
            CampaignIds={'long': [campaign_id]}
        )

        if response.Campaigns is not None:
            return response.Campaigns.Campaign[0]

        return None

    def create_campaign(self,
                        name: str,
                        budget: float,
                        status: str):

        if status not in CAMPAIGN_STATUSES:
            status = "Paused"

        campaigns = self.campaign_service.factory.create('ArrayOfCampaign')
        campaign = set_elements_to_none(
            self.campaign_service.factory.create('Campaign'))
        campaign.BudgetType = 'DailyBudgetStandard'
        campaign.DailyBudget = budget
        languages = self.campaign_service.factory.create('ns3:ArrayOfstring')
        languages.string.append('All')
        campaign.Languages = languages
        campaign.Name = name
        campaign.TimeZone = 'PacificTimeUSCanadaTijuana'
        campaign.Status = status
        campaigns.Campaign.append(campaign)

        response = self.campaign_service.AddCampaigns(
            AccountId=self.account_id,
            Campaigns=campaigns
        )

        return response

    def update_campaign(self, campaign):
        # API doesn't allow the `CampaignType` field to be updated or even have a value
        campaign.CampaignType = None

        campaigns = self.campaign_service.factory.create('ArrayOfCampaign')
        campaigns.Campaign.append(campaign)

        response = self.campaign_service.UpdateCampaigns(
            AccountId=self.account_id,
            Campaigns=campaigns
        )

        return response

    def add_adgroup(self,
                    campaign_id: str,
                    name: str,
                    bid: float,
                    status: str,
                    keywords: list,
                    kw_types: list
                    ):
        ad_groups = self.campaign_service.factory.create('ArrayOfAdGroup')
        ad_group = set_elements_to_none(
            self.campaign_service.factory.create('AdGroup'))
        ad_group.Name = name

        if status not in ADGROUP_STATUSES:
            status = "Paused"

        ad_group.Status = status
        cpc_bid = self.campaign_service.factory.create('Bid')
        cpc_bid.Amount = bid
        ad_group.CpcBid = cpc_bid
        ad_groups.AdGroup.append(ad_group)

        response = self.campaign_service.AddAdGroups(
            CampaignId=campaign_id,
            AdGroups=ad_groups,
            ReturnInheritedBidStrategyTypes=False
        )

        adgroup_id = response.AdGroupIds['long'][0]
        ad_group.Id = adgroup_id

        if keywords is not None:
            keyword_arr = self.campaign_service.factory.create(
                'ArrayOfKeyword')
            for ix, kw in enumerate(keywords):
                temp_kw = set_elements_to_none(
                    self.campaign_service.factory.create('Keyword'))
                temp_kw.Text = kw
                temp_kw.BiddingScheme = self.campaign_service.factory.create(
                    "ManualCpcBiddingScheme")

                # Needs a `Bid` object, not just the float
                temp_bid = set_elements_to_none(
                    self.campaign_service.factory.create('Bid'))
                temp_bid.Amount = bid
                temp_kw.Bid = temp_bid

                if kw_types[ix] not in MATCH_TYPES:
                    kw_types[ix] = 'Broad'
                temp_kw.MatchType = kw_types[ix]

                keyword_arr.Keyword.append(temp_kw)

            response = self.campaign_service.AddKeywords(
                AdGroupId=adgroup_id,
                Keywords=keyword_arr)

            print(response)

        return ad_group

    def add_keywords_to_adgroup(self,
                                adgroup_id,
                                keywords: list,
                                bids: list,
                                types: list):
        keyword_arr = self.campaign_service.factory.create('ArrayOfKeyword')

        for ix, kw in enumerate(keywords):
            temp_kw = set_elements_to_none(
                self.campaign_service.factory.create('Keyword'))
            temp_kw.Text = kw
            temp_kw.BiddingScheme = self.campaign_service.factory.create(
                "ManualCpcBiddingScheme")

            # Needs a `Bid` object, not just the float
            temp_bid = set_elements_to_none(
                self.campaign_service.factory.create('Bid'))
            temp_bid.Amount = bids[ix]
            temp_kw.Bid = temp_bid

            if types[ix] not in MATCH_TYPES:
                types[ix] = 'Broad'
            temp_kw.MatchType = types[ix]

            keyword_arr.Keyword.append(temp_kw)

        response = self.campaign_service.AddKeywords(
            AdGroupId=adgroup_id,
            Keywords=keyword_arr
        )

        return response

    def add_adgroup_criterion(self,
                              adgroup_id: int,
                              status: str,
                              criterion_type: str,
                              value,
                              multiplier: float,
                              is_negative: bool):
        ad_group_criterions = self.campaign_service.factory.create(
            'ArrayOfAdGroupCriterion')
        if is_negative:
            ad_group_criterion = set_elements_to_none(
                self.campaign_service.factory.create('NegativeAdGroupCriterion'))
        else:
            ad_group_criterion = set_elements_to_none(
                self.campaign_service.factory.create('BiddableAdGroupCriterion'))
        ad_group_criterion.AdGroupId = adgroup_id
        bid_multiplier = set_elements_to_none(
            self.campaign_service.factory.create('BidMultiplier'))
        bid_multiplier.Type = 'BidMultiplier'
        bid_multiplier.Multiplier = multiplier
        ad_group_criterion.CriterionBid = bid_multiplier

        if status not in CRITERION_STATUSES:
            # Some only allow "Active" as a `Status`
            status = "Active"
        ad_group_criterion.Status = status

        criterion = set_elements_to_none(
            self.campaign_service.factory.create(criterion_type))

        if criterion_type == 'LocationCriterion':
            criterion.LocationId = value
        elif criterion_type == 'DeviceCriterion':
            if value not in DEVICES:
                return None
            criterion.DeviceName = value
        elif criterion_type == 'GenderCriterion':
            if value not in GENDERS:
                return None
            criterion.GenderType = value
        elif criterion_type == 'AgeCriterion':
            if value not in VALID_AGES:
                return None
            criterion.AgeRange = value

        ad_group_criterion.Criterion = criterion
        ad_group_criterions.AdGroupCriterion.append(ad_group_criterion)

        add_ad_group_criterions_response = self.campaign_service.AddAdGroupCriterions(
            AdGroupCriterions=ad_group_criterions,
            CriterionType='Targets'
        )

        return add_ad_group_criterions_response

    def delete_adgroup_criterion(self, criterion_id, adgroup_id, criterion_type):
        response = self.campaign_service.DeleteAdGroupCriterions(
            AdGroupCriterionIds={'long': [criterion_id]},
            AdGroupId=adgroup_id,
            CriterionType=criterion_type
        )

        return response

    def get_keywords(self, adgroup_id, keyword_ids):
        response = self.campaign_service.GetKeywordsByIds(
            AdGroupId=adgroup_id,
            KeywordIds={'long': keyword_ids}
        )
        return response

    def update_keyword_bids(self, adgroup_id, keyword_ids, new_bids, dry=False):
        raise NotImplementedError # i messed around too much w/ this mtd i think - dont trust
        kw_response = self.get_keywords(adgroup_id, keyword_ids)
        keywords = kw_response['Keywords']

        for idx, (kw_id, new_kw_bid) in enumerate(zip(keyword_ids, new_bids)):
            old_kw_bid = keywords["Keyword"][idx].Bid.Amount
            self.logger.info((
                f"Adjusting bid for adgroup {adgroup_id} and keyword {kw_id}"
                f"from {old_kw_bid} to {new_kw_bid}, DRY={dry}"))
            kw_object = set_elements_to_none(
                self.campaign_service.factory.create('Keyword'))
            bid_object = set_elements_to_none(
                self.campaign_service.factory.create('Bid'))
            bid_object.Amount = old_kw_bid if dry else new_kw_bid
            kw_object.Bid = bid_object
            kw_object.Id = keywords['Keyword'][idx].Id
            keywords['Keyword'][idx] = kw_object

        response = self.campaign_service.UpdateKeywords(
            AdGroupId=adgroup_id,
            Keywords=keywords,
            ReturnInheritedBidStrategyTypes=False
        )

        if response is None:
            raise Exception()
        elif getattr(response, 'PartialErrors', ()).__len__() > 0:
            raise Exception(response['PartialErrors'])
        else:
            return response

    def bulk_update_keyword_bids(self, adgroup_ids, keyword_ids, bid_amounts):
        def get_suds_kw_bid_update(soap_client, adgroup_id, keyword_id, bid_amount):
            kw = set_elements_to_none(soap_client.factory.create('Keyword'))
            kw.AdGroup = adgroup_id
            kw.Id = keyword_id
            bid = set_elements_to_none(soap_client.factory.create('Bid'))
            bid.Amount = bid_amount
            kw.Bid = bid
            return kw
        bulk_kws = []
        for adgroup_id,keyword_id,bid_amount in zip(adgroup_ids,keyword_ids,bid_amounts):
            kw_bid_update = get_suds_kw_bid_update(
                self.campaign_service,adgroup_id,keyword_id,bid_amount)
            bulk_kws.append(BulkKeyword(ad_group_id=adgroup_id,keyword=kw_bid_update))
        globals()["bulk_kws"] = bulk_kws
        bulk_resp = self.bulk_service.upload_entities(EntityUploadParameters(
            entities=bulk_kws,
            overwrite_result_file=True,
            result_file_directory=BING_LOG_DIR,
            result_file_name=f"{NOW.date()}.log",
            response_mode='ErrorsAndResults'
        ))
        updated_kws = [*bulk_resp]
        return updated_kws