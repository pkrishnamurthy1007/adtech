import sys
from scipy.interpolate import UnivariateSpline

sys.path.append('/Users/trevor/git/datascience-utils')
from ds_utils.lookups.time import DAYSDICT

def add_spline(df, index_col, smooth_col, spline_k, spline_s, postfix='_spline'):

    df = df.copy().reset_index()
    spline = UnivariateSpline(x=df[index_col], y=df[smooth_col], k=spline_k, s=spline_s)
    df.set_index(index_col, inplace=True)
    df[smooth_col + postfix] = spline(df.index)

    return df


class TrackBest(object):

    def __init__(self, print_at_win=False):
        self.print_at_win = print_at_win
        self.winning_combo = None
        self.winning_error = None

    def process_result(self, combo, error, n=None):
        if self.winning_combo is None:
            self.winning_combo = combo
            self.winning_error = error

        if error < self.winning_error:
            self.winning_combo = combo
            self.winning_error = error

            output = [combo, error]
            if n is not None: output.append(n)
            if self.print_at_win: print(output)

    def process_set(self, results, n=None):
        for combo, error in results:
            self.process_result(combo, error, n)

def eval_combinations(combo_list, df, fit_col, weight_col, index_col):

    df = df.reset_index().set_index(index_col)
    tracker = TrackBest()

    for combo in combo_list:

        combo_errors = list()
        for time_interval in combo:
            interval_df = df.loc[(df.index >= time_interval[0]) & (df.index < time_interval[1])]
            interval_weight = interval_df[weight_col].sum()
            interval_mean = (interval_df[weight_col] * interval_df[fit_col]).sum() / interval_weight
            interval_errors = (abs(interval_df[fit_col] - interval_mean) * interval_df[weight_col]).sum()

            combo_errors.append(interval_errors)

        combo_total_error = sum(combo_errors)
        tracker.process_result(combo, combo_total_error)

    return [tracker.winning_combo, tracker.winning_error]


def compute_bidmods(intervals, df, metric_col='rps_spline', weight_col='sessions_spline'):

    global_mean = (df[weight_col] * df[metric_col]).sum() / df[weight_col].sum()
    df['baseline'] = 1.0
    df['weight'] = df[weight_col] / df[weight_col].max()

    bidmods_ls = list()
    for day, optimized_set in intervals.items():

        for time_interval in optimized_set:
            df_slice = df.loc[
                (df['dayofweek'] == day) &
                (df['hourofday'] >= time_interval[0]) &
                (df['hourofday'] < time_interval[1])
                ]

            weighted_mean = (df_slice[weight_col] * df_slice[metric_col]).sum() / df_slice[weight_col].sum()
            bidmod = weighted_mean / global_mean

            bidmods_ls.append([DAYSDICT[day], time_interval[0], time_interval[1], bidmod])

            df.loc[
                (df['dayofweek'] == day) &
                (df['hourofday'] >= time_interval[0]) &
                (df['hourofday'] < time_interval[1]),
                'interval_mean'
            ] = weighted_mean

            df.loc[
                (df['dayofweek'] == day) &
                (df['hourofday'] >= time_interval[0]) &
                (df['hourofday'] < time_interval[1]),
                'modifier'
            ] = bidmod

            df.loc[
                (df['dayofweek'] == day) &
                (df['hourofday'] >= time_interval[0]) &
                (df['hourofday'] < time_interval[1]),
                'weekday'
            ] = DAYSDICT[day]

    return bidmods_ls, df