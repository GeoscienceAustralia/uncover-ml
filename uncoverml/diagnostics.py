"""
This module contains functionality for plotting validation scores
and displaying other diagnostic information.
"""
import os
import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

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

    fig.suptitle('Target Scaling', x=0.5, y=1.02, ha='center', fontsize=16)
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
    ax.set_title('Feature Ranking', loc='left')
    ax.set_xticks([c + (barwidth * len(covariates) / 2) for c in range(len(covariates))])
    ax.set_xticklabels(covariates)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=False, shadow=True)

    return fig
