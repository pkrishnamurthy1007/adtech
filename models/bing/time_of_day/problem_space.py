from itertools import combinations

def increment(start, stop, inc, round_resolution=2, include_start=False, include_stop=False):
    i = round(start if include_start else start + inc, round_resolution)
    while (i < stop) or (i <= stop and include_stop):
        yield round(i, round_resolution)
        i += inc

def possible_time_segments(hour_resolution, periods=7):
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

