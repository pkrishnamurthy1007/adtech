import sys
import logging
from itertools import combinations
from scipy.interpolate import UnivariateSpline

sys.path.append('/Users/trevor/git/datascience-utils')
from db.redshift import RedshiftContextHc

import multiprocessing as mp

logging.basicConfig(level=logging.WARN)


def get_data():
    start_date = '20200912'
    end_date = '20200921'

    timeofday_query = f"""
        SELECT
            date_part(DOW, user_ts)::INT AS dayofweek,
            date_part(HOUR, user_ts) +
            CASE 
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 0 AND 14 THEN 0.0
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 15 AND 29 THEN 0.25
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 30 AND 44 THEN 0.5
                WHEN date_part(MINUTE, user_ts)::INT BETWEEN 45 AND 59 THEN 0.75
                END AS hourofday,
            count(DISTINCT user_ts::DATE) AS days_samplesize,
            count(session_id) AS sessions,
            count(revenue) AS conversions,
            (count(revenue)::NUMERIC / count(session_id)::NUMERIC)::NUMERIC(5,4) AS conv_rate,
            avg(revenue)::NUMERIC(8,4) AS avg_conv_value,
            (sum(revenue) / count(DISTINCT session_id))::NUMERIC(8,4) AS rpc
        FROM (
            SELECT
                s.creation_date AS utc_ts,
                extract(HOUR FROM convert_timezone('UTC', l.time_zone, s.creation_date) - s.creation_date)::INT AS utc_offset,
                l.time_zone,
                convert_timezone('UTC', l.time_zone, s.creation_date) AS user_ts,
                s.session_id,
                l.subdivision_1_iso_code AS state,
                r.revenue
            FROM tracking.session_detail AS s
            INNER JOIN data_science.maxmind_ipv4_geo_blocks AS b
                ON ip_index(s.ip_address) = b.netowrk_index
                AND inet_aton(s.ip_address) BETWEEN b.start_int AND b.end_int
            INNER JOIN data_science.maxmind_geo_locations AS l
                ON b.maxmind_id = l.maxmind_id
            LEFT JOIN (
                SELECT
                    session_id,
                    sum(revenue) AS revenue
                 FROM tron.session_revenue
                 WHERE session_creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
                 GROUP BY 1
                 HAVING sum(revenue) > 0.0
                 ) AS r
                 ON s.session_id = r.session_id
            WHERE nullif(s.ip_address, '') IS NOT null
                AND s.creation_date::DATE BETWEEN '{start_date}' AND '{end_date}'
                AND l.country_iso_code = 'US'
                --AND s.device = 'DESKTOP'
                --AND s.traffic_source = 'BING'
            ) AS sub
        GROUP BY 1,2
        ;
    """

    with RedshiftContextHc() as db_context:
        df_norm = db_context.to_df(timeofday_query)

    df_norm = df_norm \
        .sort_values(by=['dayofweek', 'hourofday'], ascending=True) \
        .set_index(['dayofweek', 'hourofday'])

    df_norm['int_ix'] = range(len(df_norm))


    spline_k_degrees = 5
    smoothing = 50

    spl_df = df_norm
    spl_df.reset_index(inplace=True)
    rps_spline = UnivariateSpline(x=spl_df['int_ix'], y=spl_df['rps_normalized'], k=spline_k_degrees, s=smoothing)
    spl_df.set_index('int_ix', inplace=True)
    spl_df['rps_interp'] = rps_spline(spl_df.index)

    spline_k_degrees = 5
    smoothing = 17500000

    spl_df.reset_index(inplace=True)
    spl_df['sessions'] = spl_df['sessions']
    sessions_spline = UnivariateSpline(x=spl_df['int_ix'], y=spl_df['sessions'], k=spline_k_degrees, s=smoothing)
    spl_df.set_index('int_ix', inplace=True)
    spl_df['sessions_interp'] = sessions_spline(spl_df.index)

    opt_df = spl_df[spl_df['dayofweek'] == 0].set_index('hourofday')[['sessions_interp', 'rps_interp']].copy()
    global_rps = (spl_df['sessions'] * spl_df['rps_interp']).sum() / spl_df['sessions_interp'].sum()

    return opt_df, global_rps


def increment(start, stop, inc, round_resolution=2, include_start=False, include_stop=False):
    i = round(start if include_start else start + inc, round_resolution)
    while (i < stop) or (i <= stop and include_stop):
        yield round(i, round_resolution)
        i += inc


def possible_time_segments(hour_resolution, periods=6):
    possible_times = increment(0.0, 24.0, hour_resolution)
    # possible_times --> [.25, .5, .75, ..., 23.75]

    for combo in combinations(possible_times, r=periods - 1):

        # start range at 0 hour
        possible_arrangement = [(0.0, combo[0])]

        # middle ranges
        for n in range(periods - 2):
            possible_arrangement.append(combo[n:n + 2])

        # end range
        possible_arrangement.append((combo[-1], 24.0))

        yield possible_arrangement


def eval_combinations(combo_list, df):

    winning_combo = None
    winning_error = None
    for combo in combo_list:
        errors = list()
        for time_interval in combo:
            df_slice = df.loc[(df.index >= time_interval[0]) & (df.index < time_interval[1])]
            combo_rps = (df_slice['sessions_interp'] * df_slice['rps_interp']).sum() / df_slice['sessions_interp'].sum()
            combo_sessions = df_slice['sessions_interp'].sum()
            combo_error = (abs(df_slice['rps_interp'] - combo_rps) * df_slice['sessions_interp']).sum()
            errors.append(combo_error)

        combo_total_error = sum(errors)

        if winning_error is None:
            winning_combo = combo
            winning_error = combo_total_error

        if combo_total_error < winning_error:
            winning_combo = combo
            winning_error = combo_total_error

    return [winning_combo, winning_error]


class TrackBest(object):

    def __init__(self):

        self.winning_combo = None
        self.winning_error = None

    def process_results(self, results, n):

        for combo, error in results:

            if self.winning_combo is None:
                self.winning_combo = combo
                self.winning_error = error

            if error < self.winning_error:
                self.winning_combo = combo
                self.winning_error = error
                print(combo, error, n)


if __name__ == '__main__':

    opt_df, global_rps = get_data()

    tracker = TrackBest()

    processes = mp.cpu_count()
    per_process_chunksize = 1000000

    stop_at = None

    problem_space = possible_time_segments(hour_resolution=.25, periods=7)

    n = 0
    payload_n = 0
    worker_n = 0
    worker_load = list()
    payload = list()
    print('starting_loop')
    while True:

        if stop_at is not None and n == stop_at:
            break

        try:
            if payload_n == per_process_chunksize:
                worker_load.append([payload, opt_df])
                worker_n += 1
                payload = list()
                payload_n = 0

            if worker_n == processes:
                print('iterations: ', n)
                with mp.Pool(processes) as p:
                    results = p.starmap(eval_combinations, worker_load)

                tracker.process_results(results, n)
                worker_load = list()
                worker_n = 0

            payload.append(next(problem_space))
            payload_n += 1

            n += 1

        except StopIteration:
            with mp.Pool(processes) as p:
                results = p.starmap(eval_combinations, worker_load)
            tracker.process_results(results, n)
            break

    print(tracker.winning_combo, tracker.winning_error, n)

