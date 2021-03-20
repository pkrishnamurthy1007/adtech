import sys
from scipy.interpolate import UnivariateSpline

def add_spline(df, index_col, smooth_col, spline_k, spline_s, postfix='_spline'):

    df = df.copy().reset_index()
    spline = UnivariateSpline(x=df[index_col], y=df[smooth_col], k=spline_k, s=spline_s)
    df.set_index(index_col, inplace=True)
    df[smooth_col + postfix] = spline(df.index)

    return df
