import requests
import json
import os

class MediaAlphaAPIClient():
    def __init__(self, base_url: str, token: str, campaign: int = None):
        self.base_url = base_url
        self.token = token
        self.campaign = campaign
        self.header = {'X-API-TOKEN': self.token, "Content-Type": "application/json"}

    def make_request(self, url: str, payload: dict = None):
        if payload is not None:
            response = requests.patch(url, json=payload, headers=self.header)
        else:
            response = requests.get(url, headers=self.header)
        
        return response

    def get_adgroups(self, campaign: int = None):
        if campaign is None and base.campaign is None:
            raise ValueError('A campaign must be defined')

        response = self.make_request(url = f"{self.base_url}/campaigns/{campaign}")
        return response.json()['ad_groups']

    def set_adgroup_bid(self, bid: float, adgroup: int, campaign: int = None):
        if campaign is None and base.campaign is None:
            raise ValueError('A campaign must be defined')

        # Fall back to object's campaign
        campaign = self.campaign if campaign is None else campaign

        payload = {'bid': bid}

        response = self.make_request(url = f"{self.base_url}/campaigns/{campaign}/ad-groups/{adgroup}", payload=payload)
        return response

    def set_channel_multiplier(self, multiplier: float, adgroup: int, channel: int, campaign: int = None):
        if campaign is None and base.campaign is None:
            raise ValueError('A campaign must be defined')

        # Fall back to object's campaign
        campaign = self.campaign if campaign is None else campaign

        payload = {'multiplier': multiplier}

        response = self.make_request(url = f"{self.base_url}/campaigns/{campaign}/ad-groups/{adgroup}/channel/{channel}", payload=payload)
        return response

    def get_time_of_day_modifiers(self, campaign: int = None):
        """Sets time of day modifiers via MediaAlpha API

        Keyword arguments:
        schedule_modifiers -- a list [of lists], one element per day
                              and inside that list one element per hour (or a list of bids per 15 mins intervals)
        campaign -- the id of the campaign
        """
        if campaign is None and base.campaign is None:
            raise ValueError('A campaign must be defined')

        # Fall back to object's campaign
        campaign = self.campaign if campaign is None else campaign
        response = self.make_request(url = f"{self.base_url}/campaigns/{campaign}")
        return response.json()

    def set_time_of_day_modifiers(self, schedule_modifiers: dict, campaign: int = None):
        """Sets time of day modifiers via MediaAlpha API

        Keyword arguments:
        schedule_modifiers -- a list [of lists], one element per day
                              and inside that list one element per hour (or a list of bids per 15 mins intervals)
        campaign -- the id of the campaign
        """
        if campaign is None and base.campaign is None:
            raise ValueError('A campaign must be defined')

        # Fall back to object's campaign
        campaign = self.campaign if campaign is None else campaign

        if 'schedule' not in schedule_modifiers:
            temp_schedule = schedule_modifiers
            schedule_modifiers = {}
            schedule_modifiers['schedule_consumer'] = temp_schedule

        response = self.make_request(url = f"{self.base_url}/campaigns/{campaign}", payload=schedule_modifiers)
        return response

