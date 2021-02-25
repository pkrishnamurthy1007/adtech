import json
import logging
import os.path

import pandas as pd
from bingapi.client import BingClient

# Debug requests and responses
# logging.basicConfig(level=logging.INFO)
# logging.getLogger('suds.client').setLevel(logging.DEBUG)
# logging.getLogger('suds.transport.http').setLevel(logging.DEBUG)

def main():
    with open("config.json") as json_data_file:
        credentials = json.load(json_data_file)

    client = BingClient(account_id = credentials['BING_ACCOUNT_ID'],
                 customer_id = credentials['BING_CUSTOMER_ID'],
                 dev_token = credentials['BING_DEVELOPER_TOKEN'],
                 client_id = credentials['BING_CLIENT_ID'],
                 refresh_token = credentials['BING_REFRESH_TOKEN']
            )

    # print(client.get_campaigns('Audience'))

    # response = client.create_campaign('API Test Campaign', 2, "Paused")
    # print(response)
    # Test API Campaign ID: 361487821

    # test_camp = client.get_campaign_by_id(361487821)
    # test_camp.DailyBudget = 1.5

    # response = client.update_campaign(test_camp)
    # print(response)

    # response = client.add_adgroup(361487821,
    #             name = "Test Adgroup 1",
    #             bid = 1.5,
    #             status = "Paused")

    # print(response)
    # AdGroupId = 1289727467714555

    # response = client.add_keywords_to_adgroup(1289727467714555, ['list', 'list2'], [1.5, 2.3], ['Broad', 'Exact'])
    # print(response)

    # response = client.delete_adgroup_criterion(80608110921421, 1289727467714555, 'Targets')
    # print(response)

    # response = client.add_adgroup_criterion(1289727467714555, "Active", "GenderCriterion", "Female", 300, is_negative = False)
    # print(response)
    # Criterion ID: 80608110921341

    #######
    ### Bid updates at keyword level test
    #######
    FILE_PATH = 'data/bids.csv'

    if not os.path.isfile(FILE_PATH):
        return None

    df = pd.read_csv(FILE_PATH)
    adgroups = df['adgroup_id'].unique().tolist()

    for adgroup in adgroups:
        temp_df = df[df['adgroup_id'] == adgroup]
        keyword_ids = temp_df['keyword_id'].tolist()
        keyword_bids = temp_df['bid'].tolist()

        response = client.update_keyword_bids(adgroup, keyword_ids, keyword_bids)
        print(f"Adgroup {adgroup} success: {response}")


if __name__ == '__main__':
    main()


