import logging
import multiprocessing as mp
import sys
from models.bing.time_of_day.data.generic_queries import hc_quarter_hour_tz_adjusted
from models.bing.time_of_day.data.model_storage import load_parameters, store_optimized_set
from models.bing.time_of_day.process import TrackBest, add_spline, eval_combinations
from models.bing.time_of_day.problem_space import possible_time_segments

sys.path.append('/Users/trevor/git/datascience-utils')
from generic import Timer

logging.basicConfig(level=logging.INFO)

load_param_set = 1602702206
# HEALTH or MEDICARE
PRODUCT = 'MEDICARE'

START_DATE = '20200912'
END_DATE = '20200921'
HOUR_RESOLUTION = .5
SEGMENTS_PER_DAY = 6
RPS_SPLINE_K = 5
RPS_SPLINE_S = 50
SESSIONS_SPLINE_K = 5
SESSIONS_SPLINE_S = 17500000

PROCESSES = mp.cpu_count() * 2
PER_PROCESS_CHUNKSIZE = 5000
DEBUG_STOP_AT_N = None



if __name__ == '__main__':

    if load_param_set is not None:
        config = load_parameters(model_run_epoch=load_param_set)
        print('config parameters overriden using database')
        print(config)
        START_DATE = config['start_date'].strftime('%Y%m%d')
        END_DATE = config['end_date'].strftime('%Y%m%d')
        HOUR_RESOLUTION = float(config['hour_resolution'])
        SEGMENTS_PER_DAY = config['segments_per_day']
        RPS_SPLINE_K = config['rps_spline_k']
        RPS_SPLINE_S = config['rps_spline_s']
        SESSIONS_SPLINE_K = config['sessions_spline_k']
        SESSIONS_SPLINE_S = config['sessions_spline_s']

    df = hc_quarter_hour_tz_adjusted(start_date=START_DATE, end_date=END_DATE, product=PRODUCT)
    df = add_spline(df, index_col='int_ix', smooth_col='rps', spline_k=5, spline_s=5)
    df = add_spline(df, index_col='int_ix', smooth_col='sessions', spline_k=5, spline_s=17500000)
    df.reset_index(inplace=True)
    global_rps = (df['sessions_spline'] * df['rps_spline']).sum() / df['sessions_spline'].sum()

    timer = Timer()

    optimal_intervals = dict()
    for day in range(7):

        day_df = df.loc[df['dayofweek'] == day].copy()[['hourofday', 'rps_spline', 'sessions_spline']].set_index('hourofday')
        problem_space = possible_time_segments(hour_resolution=HOUR_RESOLUTION, periods=SEGMENTS_PER_DAY)

        tracker = TrackBest(print_at_win=True)
        problem_space_n = 0
        payload_n = 0
        worker_n = 0
        worker_load = list()
        payload = list()

        while True:
            if DEBUG_STOP_AT_N is not None and problem_space_n == DEBUG_STOP_AT_N:
                break

            try:
                if payload_n == PER_PROCESS_CHUNKSIZE:
                    worker_load.append([payload, day_df, 'rps_spline', 'sessions_spline', 'hourofday'])
                    worker_n += 1
                    payload = list()
                    payload_n = 0

                if worker_n == PROCESSES:
                    print('day: ',day,', iterations: ', problem_space_n)
                    timer.set_timer()
                    with mp.Pool(PROCESSES) as p:
                        results = p.starmap(eval_combinations, worker_load)

                    tracker.process_set(results, problem_space_n)
                    runtime = timer.seconds_elapsed(textout=False)

                    worker_load = list()
                    worker_n = 0

                payload.append(next(problem_space))
                payload_n += 1
                problem_space_n += 1

            except StopIteration:
                print('day: ', day, ', iterations: ', problem_space_n)
                worker_load.append([payload, day_df, 'rps_spline', 'sessions_spline', 'hourofday'])
                with mp.Pool(PROCESSES) as p:
                    results = p.starmap(eval_combinations, worker_load)
                tracker.process_set(results, problem_space_n)
                break

        print(tracker.winning_combo, tracker.winning_error, problem_space_n)
        optimal_intervals[day] = tracker.winning_combo

    if load_param_set is not None:
        store_optimized_set(model_run_epoch=load_param_set, optimized_intervals=str(optimal_intervals))

    print(optimal_intervals)