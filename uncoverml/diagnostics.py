"""
This module contains functionality for plotting validation scores
and displaying other diagnostic information.
"""
import os
import json
import math
from collections import defaultdict

import rasterio
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import numpy as np
import seaborn as sns


def _real_vs_pred(rc_path, pred_path):
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
    targets = pd.read_csv(rc_path, float_precision='round_trip')
    targets.drop(list(targets.columns.values)[:-3], axis=1, inplace=True)
    tx, ty, tn = targets.columns.values
    targets = [(x, y, obs) for x, y, obs in zip(targets[tx], targets[ty], targets[tn])]

    targets_ar = np.zeros(len(targets))
    predict_ar = np.zeros(len(targets))

    with rasterio.open(pred_path) as ds:
        ar = ds.read(1)
        for i, (x, y, obs) in enumerate(targets):
            ind = ds.index(x, y)
            targets_ar[i] = obs
            predict_ar[i] = ar[ind]
    
    return targets_ar, predict_ar

def plot_real_vs_pred(rc_path, pred_path, bins=20, overlay=False):
    """
    Plots a scatterplot and 2D histogram of real vs predicted values.

    Parameters
    ----------
    rc_path : str
        Path to 'rawcovariates' CSV file.
    pred_path : str
        Path to 'prediction' TIF file.
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
    targets_ar, predict_ar = real_vs_pred(rc_path, pred_path)

    def _side_by_side(targets_ar, predict_ar, bins=bins):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5), sharey=True, gridspec_kw={'wspace': 0})
        
        for ax in axs:
            ax.set_xlabel('Real')
            ax.set_ylabel('Predicted')
            ax.label_outer()

            bin_limits = ([targets_ar.min(), targets_ar.max()], [targets_ar.min(), targets_ar.max()])
            hist = axs[1].hist2d(targets_ar, predict_ar, bins=bins, range=bin_limits, cmap=plt.cm.binary, alpha=1)
            divider = make_axes_locatable(axs[1])
            cb_axis = divider.append_axes('right', size="5%", pad=0.1)
            fig.colorbar(hist[3], cax=cb_axis)
            
            axs[0].scatter(targets_ar, predict_ar)
            axs[0].plot([targets_ar.min(), targets_ar.max()], [targets_ar.min(), targets_ar.max()],
                        color='r', linewidth=2, label='1:1')

            fig.legend(loc='upper left', bbox_to_anchor=(0.05, 0.96))
            fig.tight_layout()
            return fig

    def _overlay(targets_ar, predict_ar, bins=bins):
        fig, ax = plt.subplots(figsize=(15, 7.5))
        ax.set_xlabel('Real')
        ax.set_ylabel('Predicted')

        bin_limits = ([targets_ar.min(), targets_ar.max()], [targets_ar.min(), targets_ar.max()])
        hist = ax.hist2d(targets_ar, predict_ar, bins=bins, range=bin_limits, cmap=plt.cm.binary)
        divider = make_axes_locatable(ax)
        cb_axis = divider.append_axes('right', size="3%", pad=0.1)
        fig.colorbar(hist[3], cax=cb_axis)
        
        ax.scatter(targets_ar, predict_ar, alpha=0.8)
        ax.plot([targets_ar.min(), targets_ar.max()], [targets_ar.min(), targets_ar.max()],
                color='r', linewidth=2, label='1:1')
        

        fig.legend(loc='upper left', bbox_to_anchor=(0.05, 0.96))
        fig.tight_layout()
        return fig

    if overlay:
        return _overlay(targets_ar, predict_ar, bins)
    else:
        return _side_by_side(targets_ar, predict_ar, bins)
    
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
    df.drop(df.columns.values[-3:], axis=1, inplace=True)
    data = df.corr(method=method)
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(13,10))
    fig.suptitle('Covariate Correlation', fontsize=16, x=0.44, y=0.93)
    ax = sns.heatmap(data, vmin=-1., vmax=1., cmap=plt.cm.coolwarm, 
                     center=0., linewidths=0.1, linecolor=(1, 1, 1), ax=ax)    
    return fig

def plot_target_scaling(path, bins=20):
    """
    Plots histograms of target values pre and post-scaling.

    Parameters
    ----------
    path : str
        Path to 'transformed_targets' CSV file.
    bins : int, optional
        The number of value bins for the histograms. Default is 20.

    Returns
    -------
    :obj:maplotlib.figure.Figure
        The histograms as a matplotlib Figure.
    """
    def _color_histogram(N, bins, patches):
     fracs = N / N.max()
     norm = colors.Normalize(fracs.min(), fracs.max())
     for f, p in zip(fracs, patches):
         color = plt.cm.viridis(norm(f))
         p.set_facecolor(color)
        
    with open(path) as f:
        data = np.loadtxt(f, delimiter=',', skiprows=1)
         
    nt = data[:, 0]
    t = data[:, 1]

    figsize = 16, 8
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True, 
                            gridspec_kw={'wspace': 0})

    for ax in axs:
        ax.set_xlabel('Target Value')
        ax.set_ylabel('Frequency')
        ax.label_outer()

    axs[0].set_title('Pre-Scaling')
    axs[1].set_title('Post-Scaling')
    hist_nt = axs[0].hist(nt, bins=bins)
    hist_t = axs[1].hist(t, bins=bins)

    _color_histogram(*hist_nt)
    _color_histogram(*hist_t)

    fig.suptitle('Target Scaling', x=0.5, y=1.02, fontsize=16)
    fig.tight_layout()
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
        header = f.readline().strip().split(',')[:-3]
        header = [h.replace('.tif', '') for h in header]
        data = np.loadtxt(f, delimiter=',', skiprows=1)

    rows = math.ceil(len(header) / cols)
    figsize = cols * subplot_width, rows * subplot_height
    fig, axs = plt.subplots(ncols=cols, nrows=rows, figsize=figsize)
    if cols == 1 or rows == 1:
        for ax in axs:
            ax.set(xlabel='Target', ylabel='Covariate')
    else:
        for x in range(axs.shape[0]):
            for y in range(axs.shape[1]):
                axs[x, y].set(xlabel='Target', ylabel='Covariate')

    targets = data[:, -1]
    for i, cov in enumerate(header):
        if cols == 1 or rows == 1:
            ind = i
        else:
            x = math.floor(i / cols)
            y = i - cols * x
            ind = x, y
        axs[ind].scatter(targets, data[:, -i])
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
        ax.bar(position, s, barwidth, label=m)
        
    ax.set_xlabel('Covariate')
    ax.set_ylabel('Score')
    ax.set_title('Feature Ranking', loc='left', fontsize=16)
    ax.set_xticks([c + (barwidth * len(covariates) / 2) for c in range(len(covariates))])
    ax.set_xticklabels(covariates)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=False, shadow=True)

    return fig
