"""
This module contains functionality for plotting validation scores
and displaying other diagnostic information.
"""
import os
import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

def plot_covariates_x_targets(path, cols=1, subplot_width=8, subplot_height=4):
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
        
    # Get list of sorted covariates
    covariates = sorted([os.path.split(c)[1] for c in next(iter(fr_dict['ranks'].values()))])
    # Get list of performance metrics
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
