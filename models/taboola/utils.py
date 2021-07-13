from models.taboola.common import *
from models.taboola.utils import *

import itertools
cross = itertools.product
import jmespath
get = jmespath.search

def yield_from_type_pivot(camp,k):
    for t,v in cross(
            get(f"[{k}.type]", camp),
            get(f"{k}.value", camp) or [""]):
        yield [(k,(t,v)),1]

def yield_time_rules(camp):
    DAYS = [
        'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY',
        'FRIDAY', 'SATURDAY', 'SUNDAY', ]
    HRS = range(24)
    k = "activity_schedule"
    time_mode = get(f"{k}.mode", camp)
    if time_mode == "ALWAYS":
        included_hrs = {
            day: {hr: 1 for hr in HRS}
            for day in DAYS
        }
    else:
        included_hrs = {
            day: {hr: 0 for hr in HRS}
            for day in DAYS
        }
        for rule, day, st, end in \
                get(f"{k}.rules[*].[type,day,from_hour,until_hour]", camp):
            for hr in range(st, end):
                included_hrs[day.upper()][hr] = int(rule == "INCLUDE")
    for day, hr2v in included_hrs.items():
        for hr, v in hr2v.items():
            yield [(k, (day, hr)), v]

def yield_bid_strategy_rules(camp):
    k = "publisher_bid_strategy_modifiers"
    for r in get(f"{k}.values[*].*",camp):
        yield [(k,tuple(r)),1]

def yield_bid_modifiers(camp):
    k = "publisher_bid_modifier"
    for site,mod in get(f"{k}.values[*].*", camp):
        yield [(k, site), mod]

flat_K = [
    "advertiser_id",
    "id",
    "cpc",
    "safety_rating",
    "daily_cap",
    "daily_ad_delivery_model",
    "bid_type",
    "bid_strategy",
    "traffic_allocation_mode",
    "marketing_objective",
    "is_active",
]
type_pivoted_K = [
    "country_targeting",
    "sub_country_targeting",
    "dma_country_targeting",
    "region_country_targeting",
    "city_targeting",
    "postal_code_targeting",
    "contextual_targeting",
    "platform_targeting",
    "publisher_targeting",
    "auto_publisher_targeting",
    # "os_targeting",
    "connection_type_targeting",
    "browser_targeting",
]

def flatten_camp(camp):
    campd = {("attrs",k): camp[k] for k in flat_K}
    for k in type_pivoted_K:
        campd.update(dict(yield_from_type_pivot(camp,k)))
    campd.update(dict(yield_time_rules(camp)))
    campd.update(dict(yield_bid_strategy_rules(camp)))
    campd.update(dict(yield_bid_modifiers(camp)))
    return campd