import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin

def add_spline(df, index_col, smooth_col, spline_k, spline_s, postfix='_spline'):

    df = df.copy().reset_index()
    spline = UnivariateSpline(x=df[index_col], y=df[smooth_col], k=spline_k, s=spline_s)
    df.set_index(index_col, inplace=True)
    df[smooth_col + postfix] = spline(df.index)

    return df

def calculate_bag_rps(df, session_date_field: str, revenue_field: str):
    # Not memory efficient but just in case
    df = df.copy()

    df.sort_values(by=[session_date_field], ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['has_conversion'] = df[revenue_field] > 0
    mask = pd.Series([True] + (~df['has_conversion']).tolist()[:-1], index=df.index)
    df['session_no'] = np.where(mask, df.groupby(mask.ne(mask.shift()).cumsum()).cumcount()+2, 1)

    # Get rid of first conversion as it's slightly biased
    first_conv = df[df['has_conversion']].index[0]
    df = df.iloc[(first_conv+1):len(df)]
    df['bag_rps'] = df[revenue_field] / df['session_no']
    df['bag_conversion_rate'] = 1 / df['session_no']
    df.loc[~df['has_conversion'], 'bag_rps'] = np.nan
    df.loc[~df['has_conversion'], 'bag_conversion_rate'] = np.nan

    return df

def split_to_intervals(df, weekday: str, hourofday:str, hourfraction:str, rps_field: str, conversion_rate_field: str):
    grouped_df = df.groupby(by=[weekday, hourofday, hourfraction]).agg({rps_field: 'mean', conversion_rate_field: 'mean', 'session_id': 'count'})
    grouped_df.reset_index(inplace=True)
    grouped_df.rename(columns={'session_id': 'sessions'}, inplace=True)
    grouped_df['ix'] = range(len(grouped_df))
    grouped_df[rps_field] = grouped_df[rps_field].fillna(method='ffill')

    return grouped_df

# From agramfort gist at https://gist.github.com/agramfort/850437
# We could also have a `numba` equivalent but this is fast enough for how many data points we have 
def calc_lin_reg_betas(x, y, weights=None):
    """Calculates the intercept and gradient for the specified local regressions"""
    if weights is None:
        weights = np.ones(len(x))

    b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
    A = np.array([[np.sum(weights), np.sum(weights * x)],
                  [np.sum(weights * x), np.sum(weights * x * x)]])

    betas = np.linalg.lstsq(A, b, rcond=None)[0]

    return betas

## NOTE:
## `moepy` (MIT license) modified code follows:
def get_weighting_locs(x, reg_anchors=None, num_fits=None): 
    """Identifies the weighting locations for the provided dataset"""
    num_type_2_dist_rows = {
        type(None) : lambda x, num_fits: x.reshape(-1, 1),
        int : lambda x, num_fits: num_fits_2_reg_anchors(x, num_fits).reshape(-1, 1),
    }

    if reg_anchors is None:
        weighting_locs = num_type_2_dist_rows[type(num_fits)](x, num_fits)
    else:
        weighting_locs = reg_anchors.reshape(-1, 1)

    return weighting_locs


get_frac_idx = lambda x, frac: int(np.floor(len(x) * frac))

get_dist_thresholds = lambda frac_idx, dist_matrix: np.sort(dist_matrix)[:, frac_idx]

def clean_weights(weights):
    """Normalises each models weightings and removes non-finite values"""
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = weights/weights.sum(axis=0)

    weights = np.where(~np.isfinite(weights), 0, weights)

    return weights

def dist_to_weights_matrix(dist_matrix, dist_thresholds):
    """Converts distance matrix and thresholds to weightings"""

    # Use tricube kernel to map distances to weights
    # Something to fine tune if we want to modify the regressions?
    weights = (1 - ((np.abs(dist_matrix)/dist_thresholds.reshape(-1, 1)).clip(0, 1) ** 3)) ** 3
    weights = clean_weights(weights)

    return weights

def create_dist_matrix(x, reg_anchors=None, num_fits=None): 
    """Constructs the distance matrix for the desired weighting locations"""
    weighting_locs = get_weighting_locs(x, reg_anchors=reg_anchors, num_fits=num_fits)
    dist_matrix = np.abs(weighting_locs - x.reshape(1, -1))

    return dist_matrix

def get_weights_matrix(x, frac=0.4, weighting_locs=None, reg_anchors=None, num_fits=None):
    """Wrapper for calculating weights from the raw data and LOWESS fraction"""
    frac_idx = get_frac_idx(x, frac)

    if weighting_locs is not None:
        dist_matrix = np.abs(weighting_locs - x.reshape(1, -1))
    else:
        dist_matrix = create_dist_matrix(x, reg_anchors=reg_anchors, num_fits=num_fits)

    dist_thresholds = get_dist_thresholds(frac_idx, dist_matrix)
    weights = dist_to_weights_matrix(dist_matrix, dist_thresholds)

    return weights

def fit_regressions(x, y, weights=None, reg_func=calc_lin_reg_betas, num_coef=2, **reg_params):
    """Calculates the design matrix for the specified local regressions"""
    if weights is None:
        weights = np.ones(len(x))

    n = weights.shape[0]

    y_pred = np.zeros(n)
    design_matrix = np.zeros((n, num_coef))

    for i in range(n):
        design_matrix[i, :] = reg_func(x, y, weights=weights[i, :], **reg_params)

    return design_matrix

num_fits_2_reg_anchors = lambda x, num_fits: np.linspace(x.min(), x.max(), num=num_fits)

def calc_robust_weights(y, y_pred, max_std_dev=6):
    """
    Calculates robustifying weightings that penalise outliers
    NCSS Cleveland, W. S. (1979) in his paper 'Robust Locally Weighted Regression and Smoothing Scatterplots'
    https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Robust_Regression.pdf
    """
    residuals = y - y_pred

    # Get std using quantiles (normal distribution)
    std_dev = np.quantile(np.abs(residuals), 0.682)

    # Default value for max_std_dev is 6, this could probably be fine-tuned
    clean_residuals = np.clip(residuals / (max_std_dev * std_dev), -1, 1)
    robust_weights = (1 - clean_residuals ** 2) ** 2

    return robust_weights


class Lowess(BaseEstimator, RegressorMixin):
    """
    This class provides a Scikit-Learn compatible model for Locally Weighted
    Scatterplot Smoothing, including robustifying procedures against outliers.

    For more information on the underlying algorithm please refer to
    * William S. Cleveland: "Robust locally weighted regression and smoothing
      scatterplots", Journal of the American Statistical Association, December 1979,
      volume 74, number 368, pp. 829-836.
    * William S. Cleveland and Susan J. Devlin: "Locally weighted regression: An
      approach to regression analysis by local fitting", Journal of the American
      Statistical Association, September 1988, volume 83, number 403, pp. 596-610.

    Initialisation Parameters:
        reg_func: function that accepts the x and y values then returns the intercepts and gradients

    Attributes:
        reg_func: function that accepts the x and y values then returns the intercepts and gradients
        fitted: Boolean flag indicating whether the model has been fitted
        frac: Fraction of the dataset to use in each local regression
        weighting_locs: Locations of the local regression centers
        loading_weights: Weights of each data-point across the localalised models
        design_matrix: Regression coefficients for each of the localised models
    """

    def __init__(self, reg_func=calc_lin_reg_betas):
        self.reg_func = reg_func
        self.fitted = False
        return


    def calculate_loading_weights(self, x, reg_anchors=None, num_fits=None, external_weights=None, robust_weights=None):
        """
        Calculates the loading weights for each data-point across the localised models

        Parameters:
            x: values for the independent variable
            reg_anchors: Locations at which to center the local regressions
            num_fits: Number of locations at which to carry out a local regression
            external_weights: Further weighting for the specific regression
            robust_weights: Robustifying weights to remove the influence of outliers
        """

        # Calculating the initial loading weights
        weighting_locs = get_weighting_locs(x, reg_anchors=reg_anchors, num_fits=num_fits)
        loading_weights = get_weights_matrix(x, frac=self.frac, weighting_locs=weighting_locs)

        # Applying weight adjustments
        if external_weights is None:
            external_weights = np.ones(x.shape[0])

        if robust_weights is None:
            robust_weights = np.ones(x.shape[0])

        weight_adj = np.multiply(external_weights, robust_weights)
        loading_weights = np.multiply(weight_adj, loading_weights)

        # Post-processing weights
        with np.errstate(divide='ignore', invalid='ignore'):
            loading_weights = loading_weights/loading_weights.sum(axis=0) # normalising

        loading_weights = np.where(~np.isfinite(loading_weights), 0, loading_weights) # removing non-finite values

        self.weighting_locs = weighting_locs
        self.loading_weights = loading_weights

        return 


    def fit(self, x, y, frac=0.4, reg_anchors=None, 
            num_fits=None, external_weights=None, 
            robust_weights=None, robust_iters=3,
            max_std_dev=6, **reg_params):
        """
        Calculation of the local regression coefficients for 
        a LOWESS model across the dataset provided. This method 
        will reassign the `frac`, `weighting_locs`, `loading_weights`,  
        and `design_matrix` attributes of the `Lowess` object.

        Parameters:
            x: values for the independent variable
            y: values for the dependent variable
            frac: LOWESS bandwidth for local regression as a fraction
            reg_anchors: Locations at which to center the local regressions
            num_fits: Number of locations at which to carry out a local regression
            external_weights: Further weighting for the specific regression
            robust_weights: Robustifying weights to remove the influence of outliers
            robust_iters: Number of robustifying iterations to carry out
        """

        self.frac = frac

        # Solving for the design matrix
        self.calculate_loading_weights(x, reg_anchors=reg_anchors, num_fits=num_fits, external_weights=external_weights, robust_weights=robust_weights)
        self.design_matrix = fit_regressions(x, y, weights=self.loading_weights, reg_func=self.reg_func, **reg_params)

        # Recursive robust regression
        if robust_iters > 1:
            y_pred = self.predict(x)
            robust_weights = calc_robust_weights(y, y_pred, max_std_dev)

            robust_iters -= 1
            y_pred = self.fit(x, y, frac=self.frac, reg_anchors=reg_anchors, num_fits=num_fits, external_weights=external_weights, robust_weights=robust_weights, robust_iters=robust_iters, **reg_params)

            return y_pred

        self.fitted = True

        return 


    def predict(self, x_pred):
        point_evals = self.design_matrix[:, 0] + np.dot(x_pred.reshape(-1, 1), self.design_matrix[:, 1].reshape(1, -1))
        pred_weights = get_weights_matrix(x_pred, frac=self.frac, reg_anchors=self.weighting_locs)

        y_pred = np.multiply(pred_weights, point_evals.T).sum(axis=0)

        return y_pred

    # Especially useful for our Time of Day modifier use case.
    def fit_predict(self, x, y, frac=0.4, reg_anchors=None, 
            num_fits=None, external_weights=None, 
            robust_weights=None, robust_iters=3, **reg_params):
        self.fit(x, y, frac, reg_anchors, num_fits, external_weights, robust_weights, robust_iters, **reg_params)
        return self.predict(x)


def get_lowess_spline(df, date_field, revenue_field, show_plots=False):
    df = calculate_bag_rps(df, date_field, revenue_field).copy()

    if 'weekday' not in df.columns or 'hourofday' not in df.columns or 'hourfraction' not in df.columns:
        print("`weekday`, `hourofday` and `hourfraction` fields are needed.")
        return None

    grouped_df = split_to_intervals(df, 'weekday', 'hourofday', 'hourfraction', 'bag_rps', 'bag_conversion_rate')

    quants = grouped_df['bag_rps'].quantile([.99])
    grouped_df.loc[grouped_df['bag_rps'] > quants[.99], 'bag_rps'] = quants[.99]
    grouped_df['dayhour'] = grouped_df['weekday'].astype(str) + (grouped_df['hourofday'] + grouped_df['hourfraction']).astype(str)
    grouped_df['mean_rps'] = grouped_df['bag_rps'].mean()
    grouped_df['mean_sessions'] = grouped_df['sessions'].mean()

    lowess = Lowess()
    y_pred = lowess.fit_predict(grouped_df['ix'].values, grouped_df['bag_rps'].values, frac=0.03, max_std_dev=5)

    if show_plots:
        plt.figure(figsize=(15,5))
        plt.scatter(grouped_df['ix'], grouped_df['bag_rps'], label='Original RPS', zorder=2)
        plt.plot(grouped_df['ix'], y_pred, '--', label='Robust LOWESS', color='k', zorder=3)
        plt.plot(grouped_df['ix'], grouped_df['mean_rps'], label='Mean RPS', color='yellow')
        plt.legend(frameon=True)
        plt.show()
    
    lowess = Lowess()
    session_pred = lowess.fit_predict(grouped_df['ix'].values, grouped_df['sessions'].values, frac=0.03)

    if show_plots:
        plt.figure(figsize=(15,5))
        plt.scatter(grouped_df['ix'], grouped_df['sessions'], label='Original session count', zorder=2)
        plt.plot(grouped_df['ix'], session_pred, '--', label='Robust LOWESS', color='k', zorder=3)
        plt.plot(grouped_df['ix'], grouped_df['mean_sessions'], label='Mean session count', color='yellow')
        plt.legend(frameon=True)
        plt.show()
    
    norm_revenue_mean = (y_pred * session_pred).sum() / session_pred.sum()
    
    if show_plots:
        plt.figure(figsize=(15,5))
        plt.plot(grouped_df['ix'], y_pred, label='Robust LOWESS', color='k', zorder=3)
        plt.plot(grouped_df['ix'], [norm_revenue_mean] * len(y_pred), label='Mean', color='yellow')
        plt.legend(frameon=True)
        plt.show()
    
    return y_pred, norm_revenue_mean
