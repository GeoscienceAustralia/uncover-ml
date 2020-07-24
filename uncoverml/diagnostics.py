"""
This module contains functionality for plotting validation scores
and other diagnostic information.
"""
import os
import json
import math
import logging
from collections import defaultdict
from typing import Dict

import rasterio
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
import seaborn as sns

_CACHE = dict()
_logger = logging.getLogger(__name__)

def _suptitle_buffer(axes, topmargin=1):
     """
     From 'ImportanceOfBeingErnest' on StackOverflow.
     """
     fig = axes.flatten()[0].figure
     s = fig.subplotpars
     w, h = fig.get_size_inches()

     figh = h - (1-s.top)*h  + topmargin
     fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
     fig.set_figheight(figh)


def _color_histogram(N, bins, patches):
     fracs = N / N.max()
     norm = matplotlib.colors.Normalize(fracs.min(), fracs.max())
     for f, p in zip(fracs, patches):
         color = plt.cm.viridis(norm(f))
         p.set_facecolor(color)

def _real_vs_pred_from_prediction(rc_path, pred_path):
    """
    Gives a pair of arrays, the first containing real target values
    and the second containing predicted values for the corresponding
    target coordinate.

    Parameters
    ----------
    rc_path : str
        Path to 'rawcovariats' CSV file.
    pred_path : str
        Path to 'prediction' TIF file.
    
    Returns
    -------
    tuple(:obj:numpy.ndarray, :obj:numpy.ndarray)
        Arrays of targets ([0]) and corresponding predictions ([1]).
    """
    if '_real_vs_pred_from_prediction' in _CACHE:
        return _CACHE['_real_vs_pred_from_prediction']
        
    df = pd.read_csv(rc_path, float_precision='round_trip')
    targets = [(x, y, obs) for x, y, obs in zip(df['X'], df['Y'], df['target'])]

    targets_ar = np.zeros(len(targets))
    predict_ar = np.zeros(len(targets))

    with rasterio.open(pred_path) as ds:
        ar = ds.read(1)
        for i, (x, y, obs) in enumerate(targets):
            ind = ds.index(x, y)
            targets_ar[i] = obs
            predict_ar[i] = ar[ind]

    _CACHE['_real_vs_pred_from_prediction'] = targets_ar, predict_ar
    return targets_ar, predict_ar

def _real_vs_pred_from_crossval(crossval_path):
    """
    Same as :func:`real_vs_pred_from_prediction` but generates arrays
    from crossvalidation results file.

    Parameters
    ----------
    crossval_path : Path to 'crossval_results' CSV file.
        
    Returns
    -------
    tuple(:obj:numpy.ndarray, :obj:numpy.ndarray)
        Arrays of targets ([0]) and corresponding predictions ([1]).
    """
    if '_real_vs_pred_from_crossval' in _CACHE:
        return _CACHE['_real_vs_pred_from_crossval']

    rvp = pd.read_csv(crossval_path, float_precision='round_trip')
    # Split into the length of columns to support old format of crossval results.
    arrays = np.hsplit(rvp.to_numpy(), len(rvp.columns))
    # Only need the first two arrays ([0] = targets, [1] = predictions).
    return arrays[0].flatten(), arrays[1].flatten()

def plot_residual_error_crossval(crossval_path, bins=20):
    return _plot_residual_error(crossval_path=crossval_path, bins=bins)

def plot_residual_error_prediction(rc_path, pred_path, bins=20):
    return _plot_residual_error(rc_path=rc_path, pred_path=pred_path, bins=bins)

def _plot_residual_error(crossval_path=None, rc_path=None, pred_path=None, bins=20,):
    """
    Plots a histogram of residual error. Residual is 
    abs(predicted value - target value). 
    
    Parameters
    ----------
    rc_path : str
        Path to 'rawcovariates' CSV file.
    pred_path : str
        Path to 'prediction' TIF file.
    bins : int
        Number of bins for histogram.

    Returns
    -------
    :obj:matplotlib.figure.Figure
        The histogram as a matplotlib Figure.
    """
    if crossval_path:
        targets_ar, predict_ar = _real_vs_pred_from_crossval(crossval_path)
    elif targets_ar and predict_ar:
        targets_ar, predict_ar = _real_vs_pred_from_prediction(rc_path, pred_path)
    else:
        raise ValueError("Must provide either 'crossval_path' or 'rc_path' and 'pred_path' to "
                         "Real vs Prediction scatter plot.")

    residuals = np.absolute(targets_ar - predict_ar)

    figsize = 13.1, 7.5
    fig, ax = plt.subplots(figsize=figsize)

    hist = ax.hist(residuals, bins=20)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Residual')
    _color_histogram(*hist)
    fig.suptitle('Residual Error', x=0.5, y=1.02, fontsize=16)
    fig.tight_layout()

    return fig

def plot_real_vs_pred_crossval(crossval_path, scores_path=None, bins=20,
                               overlay=False, hist_cm=None, scatter_color=None,
                               figsize=(25, 12.5), point_size=None):
    return _plot_real_vs_pred(crossval_path=crossval_path, scores_path=scores_path, bins=bins, 
                              overlay=overlay, hist_cm=hist_cm, scatter_color=scatter_color,
                              figsize=figsize, point_size=point_size)

def plot_real_vs_pred_prediction(rc_path, pred_path, scores_path=None, bins=20,
                                 overlay=False, hist_cm=None, scatter_color=None,
                                 figsize=(25, 12.5), point_size=None):
    return _plot_real_vs_pred(rc_path=rc_path, pred_path=pred_path, scores_path=scores_path, bins=bins,
                                overlay=overlay, hist_cm=hist_cm, scatter_color=scatter_color,
                                figsize=figsize, point_size=point_Size)

def _plot_real_vs_pred(crossval_path=None, rc_path=None, pred_path=None, 
                      scores_path=None, bins=20, overlay=False,
                      hist_cm=None, scatter_color=None, figsize=(15, 7.5),
                      point_size=None):
    """
    Plots a scatterplot and 2D histogram of real vs predicted values.

    Parameters
    ----------
    rc_path : str
        Path to 'rawcovariates' CSV file.
    pred_path : str
        Path to 'prediction' TIF file.
    scores_path : str, optional
        Path to 'crossval_scores' JSON file. If provided, plot will be
        annotated with validation scores.
    bins : int
        Number of bins for 2D histogram.
    overlay : bool
        If True, the histogram will be overlaid on the scatterplot. If False,
        they will be drawn side-by-side.

    Returns
    -------
    :obj:matplotlib.figure.Figure
        The plots as a matplotlib figure.
    """
    if crossval_path:
        targets_ar, predict_ar = _real_vs_pred_from_crossval(crossval_path)
    elif targets_ar and predict_ar:
        targets_ar, predict_ar = _real_vs_pred_from_prediction(rc_path, pred_path)
    else:
        raise ValueError("Must provide either 'crossval_path' or 'rc_path' and 'pred_path' to "
                         "Real vs Prediction scatter plot.")

    def _side_by_side(targets_ar, predict_ar, bins=bins):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, 
                                sharey=True, gridspec_kw={'wspace': 0})
        
        for ax in axs:
            ax.set_xlabel('Real', fontsize=20)
            ax.set_ylabel('Predicted', fontsize=20)
            ax.label_outer()

        bin_limits = ([targets_ar.min(), targets_ar.max()], [targets_ar.min(), targets_ar.max()])
        hist = axs[1].hist2d(targets_ar, predict_ar, bins=bins, range=bin_limits, cmap=hist_cm, cmin=1)
        divider = make_axes_locatable(axs[1])
        axs[1].tick_params(labelsize=18)
        cb_axis = divider.append_axes('right', size="5%", pad=0.1)
        cb_axis.tick_params(labelsize=18)
        fig.colorbar(hist[3], cax=cb_axis)
        
        axs[0].scatter(targets_ar, predict_ar, s=point_size, color=scatter_color, alpha=0.8)
        axs[0].plot([targets_ar.min(), targets_ar.max()], [targets_ar.min(), targets_ar.max()],
                    color='b', linewidth=2, label='One-to-one')
        b, m = polyfit(targets_ar, predict_ar, 1)
        axs[0].plot(targets_ar, b + m * targets_ar, color='r', linewidth=2, label='Regression')
        axs[0].legend(loc='lower right', fontsize=18)
        axs[0].tick_params(labelsize=18)

        return fig

    def _overlay(targets_ar, predict_ar, bins=bins):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('Real', fontsize=20)
        ax.set_ylabel('Predicted', fontsize=20)

        bin_limits = ([targets_ar.min(), targets_ar.max()], [targets_ar.min(), targets_ar.max()])
        hist = ax.hist2d(targets_ar, predict_ar, bins=bins, range=bin_limits, cmap=hist_cm, cmin=1)
        divider = make_axes_locatable(ax)
        cb_axis = divider.append_axes('right', size="2.8%", pad=0.1)
        cb_axis.tick_params(labelsize=18)
        fig.colorbar(hist[3], cax=cb_axis)
        
        ax.scatter(targets_ar, predict_ar, s=point_size, color=scatter_color, alpha=0.8)
        # 1 to 1 line
        ax.plot([targets_ar.min(), targets_ar.max()], [targets_ar.min(), targets_ar.max()],
                color='b', linewidth=2, label='One-to-one')
        # Regression line
        b, m = polyfit(targets_ar, predict_ar, 1)
        ax.plot(targets_ar, b + m * targets_ar, color='r', linewidth=2, label='Regression')
        ax.legend(loc='lower right', fontsize=18)
        ax.tick_params(labelsize=18)
        

        return fig
        
    if overlay:
        fig = _overlay(targets_ar, predict_ar, bins)
    else:
        fig = _side_by_side(targets_ar, predict_ar, bins)


    if scores_path:
        display_scores = {'R^2               ': 'r2_score',
                          'Adjusted R^2      ': 'adjusted_r2_score', 
                          'LCCC              ': 'lins_ccc', 
                          'Mean Log Loss     ': 'mll',
                          'Explained Vairance': 'expvar',
                          'Standarised MSE   ': 'smse'}
        
        with open(scores_path) as f:
            crossval_scores = json.load(f)
        if any('transformed' in s for s in crossval_scores):
            for k in display_scores.keys():
                display_scores[k] += '_transformed'
        
        display_string = ''
        for k, v in display_scores.items():
            try:
                display_string += f'{k} = {crossval_scores[v]:.3f}\n'
            except KeyError:
                pass
        
        axs = fig.axes
        fig.text(0.01, 0.8, display_string, fontsize=20, fontfamily='monospace', 
                    transform=axs[0].transAxes, wrap=True)
    
    # leg = fig.legend(loc='upper left', bbox_to_anchor=(0.065, 0.965))
    fig.suptitle('Real vs Predicted', x=0.5, y=0.96, fontsize=24)
    fig.tight_layout()
    return fig    
    
def plot_covariate_correlation(path, method='pearson'):
    """
    Plots matrix of correlation between covariates.

    Parameters
    ----------
    path : str
        Path to 'rawcovariates' CSV file.
    method : str, optional
        Correlation coefficient to calculate. Choices are
        'pearson', 'kendall', 'spearman'. Default is 'pearson'.

    Returns
    -------
    :obj:matplotlib.figure.Figure
        The matrix plot as a matplotlib Figure.
    """
    df = pd.read_csv(path)
    df.drop([n for n in df.columns.values if not n.endswith('.tif')], axis=1, inplace=True)
    data = df.corr(method=method)
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(15,12))
    ax.set_title("Covariate Correlation", fontsize=16)
    divider = make_axes_locatable(ax)
    cb_ax = divider.append_axes('right', size="2.8%", pad=0.1)

    ax = sns.heatmap(data, vmin=-1., vmax=1., cmap=plt.cm.coolwarm, 
                     center=0., linewidths=0.1, linecolor=(1, 1, 1), cbar_ax=cb_ax, ax=ax)    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    return fig

def plot_target_scaling(path, bins=20, title='Target Scaling', sharey=False):
    """
    Plots histograms of target values pre and post-scaling.

    Parameters
    ----------
    path : str
        Path to 'transformed_targets' CSV file.
    bins : int, optional
        The number of value bins for the histograms. Default is 20.
    title : str, optional
        The title of the plot. Defaults to 'Target Scaling'.
    sharey : bool
        Whether the plots will share a y-axis and scale. Default is False.

    Returns
    -------
    :obj:maplotlib.figure.Figure
        The histograms as a matplotlib Figure.
    """
    with open(path) as f:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
         
    nt = data[:, 0]
    t = data[:, 1]

    figsize = 16, 8
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=sharey, 
                            gridspec_kw={'wspace': 0} if sharey else {})

    for ax in axs:
        ax.set_xlabel('Target Value', fontsize=16)
        ax.set_ylabel('Frequency', fontsize=16)
        if sharey:
            ax.label_outer()

    axs[0].set_title('Pre-Scaling', fontsize=18)
    axs[1].set_title('Post-Scaling', fontsize=18)
    hist_nt = axs[0].hist(nt, bins=bins)
    hist_t = axs[1].hist(t, bins=bins)

    _color_histogram(*hist_nt)
    _color_histogram(*hist_t)

    fig.suptitle(title, x=0.5, y=0.98, fontsize=20)
    _suptitle_buffer(axs)
    return fig

def plot_covariates_x_targets(path, cols=2, subplot_width=8, subplot_height=4):
    """
    Plots scatter plots of each covariate intersected with target
    values. 
    
    Parameters
    ----------
    path : str
        Path to 'rawcovariates' CSV file containing intersection
        of targets and covariates.
    cols : int, optional
        The number of columns to split the figure into. Default is 1.
    subplot_width : int
        Width of each subplot in inches. Default is 8.
    subplot_height : int
        Width of each subplot in inches. Default is 4.

    Returns
    -------
    :obj:matplotlib.figure.Figure
        The scatter plots as a matplotlib Figure.
    """
    with open(path) as f:
        header = f.readline().strip().split(',')
        covs = [h.replace('.tif', '') if h.endswith('.tif') else h for h in header]
        data = np.loadtxt(f, delimiter=',', skiprows=1, dtype='<U254')

    rows = math.ceil(len(covs) / cols)
    figsize = cols * subplot_width, rows * subplot_height
    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=figsize)

    targets = data[:,header.index('target')].astype(float)
    for i, cov in enumerate(covs):
        if cols == 1 or rows == 1:
            ind = i
        else:
            x = math.floor(i / cols)
            y = i - cols * x
            ind = x, y
        axs[ind].set(xlabel='Target', ylabel='Covariate')
        try:
            axs[ind].scatter(targets, data[:,i].astype(float))
        except ValueError:
            # TODO: encode non-float covariates
            _logger.error("Covariate %s could not be converted to float, will not be included on "
                          "intersection plot" % cov)
            continue
        axs[ind].set_title(cov)

    fig.suptitle('Covariate-Target Intersection', x=0.52, y=1.01, fontsize=16)
    fig.tight_layout()
    return fig

def plot_feature_ranks(path, barwidth=0.08, figsize=(15, 9)):
    """
    Plots a grouped bar chart of feature rank scores, grouped by 
    covariate. Depending on the number of covariates and metrics being
    calculated you may need to tweak barwidth and figsize so the bars
    fit.

    Parameters
    ----------
    path : str
        Path to JSON file containing feature ranking results.
    barwidth : float, optional
        Width of the bars.
    figsize : tuple(float, float), optional
        The (width, height) of the figure in inches.

    Returns
    -------
    :obj:matplotlib.figure.Figure
        The bar chart as a matplotlib Figure.
    """
    with open(path, 'r') as f:
        fr_dict = json.load(f)
        
    covariates = sorted([os.path.split(c)[1] for c in next(iter(fr_dict['ranks'].values()))])
    metrics = fr_dict['ranks'].keys()
    labels = {
        'r2_score': 'R^2',
        'adjusted_r2_score': 'Adjusted R^2', 
        'lins_ccc': 'LCCC', 
        'mll': 'Mean Log Loss',
        'expvar': 'Explained Variance',
        'smse': 'Standardised MSE'
    }
    title = 'Feature Ranking'
    if any('transformed' in s for s in metrics):
        # Is transformed model so only display transformed metrics.
        title = 'Feature Ranking (Transformed Metrics)'
        for k in list(labels.keys()):
            labels[k + '_transformed'] = labels.pop(k)
    metrics = labels.keys()

    # Get scores grouped by metric and ordered by covariate
    scores = defaultdict(list)
    for m in metrics:
        for cp, s in list(zip(fr_dict['ranks'][m], fr_dict['scores'][m])) :
            c = os.path.split(cp)[1]
            scores[m].append((c, s))
        scores[m].sort(key=lambda a: a[0])
        scores[m] = list(zip(*scores[m]))[1]
            
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(covariates))
    for i, (m, s) in enumerate(scores.items()):
        position = positions + i * barwidth
        ax.bar(position, s, barwidth, label=labels[m])
    
    ax.set_xlabel('Covariate')
    ax.set_ylabel('Score')
    ax.set_title(title, loc='left', fontsize=16)
    ax.set_xticks([c + (barwidth * len(metrics) / 2) for c in range(len(covariates))])
    ax.set_xticklabels(covariates, rotation=-45, ha='left')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=False, shadow=True)

    return fig


def plot_feature_rank_curves(path, subplot_width=8, subplot_height=4):
    """
    Plots curves for feature ranking of each metric.

    Parameters
    ----------
    path : str
        Path to 'featureranks' JSON file.
    subplot_width : int, optional
        Width of each subplot. Default is 8.
    subplot_height : int, optional
        Height of each subplot. Default is 4.
    
    Returns
    -------
    :obj:matplotlib.figure.Figure
        The plots as a matplotlib Figure.
    """
    with open(path) as f:
        fr_dict = json.load(f)
    lower_is_better = ['mll', 'mll_transformed', 'smse', 'smse_transformed']
    covariates = sorted([os.path.split(c)[1] for c in next(iter(fr_dict['ranks'].values()))])
    metrics = fr_dict['ranks'].keys()
    labels = {
        'r2_score': 'R^2',
        'adjusted_r2_score': 'Adjusted R^2', 
        'lins_ccc': 'LCCC', 
        'mll': 'Mean Log Loss',
        'expvar': 'Explained Variance',
        'smse': 'Standardised MSE',
    }
    t_labels = {}
    for k, v in labels.items():
        t_labels[k + '_transformed'] = v + ' Transformed'
    labels.update(t_labels)

    # Get scores grouped by metric and ordered by score
    scores = defaultdict(list)
    for m in metrics:
        for cp, s in list(zip(fr_dict['ranks'][m], fr_dict['scores'][m])) :
            c = os.path.split(cp)[1]
            scores[m].append((c, s))
        scores[m].sort(key=lambda a: a[1])
        if m in lower_is_better:
            scores[m].reverse()
    
    rows = math.ceil(len(metrics) / 2)
    cols = 2
    figsize = cols * subplot_width, rows * subplot_height
    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=figsize)
    for i, m in enumerate(metrics):
        x = math.floor(i / cols)
        y = i - cols * x
        ind = x, y
        z = list(zip(*scores[m]))
        ticks = [i for i in range(len(z[0]))]
        axs[ind].plot(ticks, z[1])
        axs[ind].scatter(ticks, z[1])
        axs[ind].set_xticks(ticks)
        axs[ind].set_xticklabels(z[0], rotation=-45, ha='left')
        axs[ind].set(xlabel='Covariate', ylabel='Score')
        axs[ind].set_title(labels[m])
        
    fig.tight_layout()
    return fig
