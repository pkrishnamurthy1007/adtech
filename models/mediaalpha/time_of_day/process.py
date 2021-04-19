import os
import ds_utils
import pandas as pd
import numpy as np
import datetime

from models.mediaalpha.time_of_day.data.queries import hc_session_conversions
from models.utils.time_of_day import get_lowess_spline

from api.mediaalpha.mediaalpha_client import MediaAlphaAPIClient

def main():
    TRAFFIC_SOURCE = 'MEDIAALPHA'
    PRODUCT = 'HEALTH'
    NOW = datetime.datetime.now()
    DAY = datetime.timedelta(days=1)

    start_date = NOW - 90 * DAY
    end_date = NOW - 0 * DAY

    # Get data according to date range
    session_revenue = hc_session_conversions(start_date, end_date, PRODUCT, TRAFFIC_SOURCE)
    
    # Get modifiers
    rev_spline, avg_rev = get_lowess_spline(session_revenue, 'user_ts', 'revenue', show_plots=False)
    modifiers = (rev_spline / avg_rev).tolist()

    # Terrible way of making Sunday be last
    modifiers = modifiers[96:] + modifiers[:96]

    schedule_payload = []
    temp_day = []
    temp_hour = []
    for day in range(7):
        for hour in range(24):
            temp_day.append(modifiers[(day * 96 + hour * 4):(day * 96 + hour * 4 + 4)])
        schedule_payload.append(temp_day)
        temp_day = []

    token = os.getenv("MEDIAALPHA_TOKEN")
    client = MediaAlphaAPIClient(base_url = "https://insurance-api.mediaalpha.com/220", token=token)
    client.set_time_of_day_modifiers(schedule_payload, campaign=23898)

if __name__ == '__main__':
    main()
