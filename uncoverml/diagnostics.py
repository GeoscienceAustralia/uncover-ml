"""
This module contains functionality for plotting validation scores
and displaying other diagnostic information.
"""
import os
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

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
    barwidth : float, optiona;
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
            
    figsize = 15, 9
    fig, ax = plt.subplots(figsize=figsize)
    barwidth = 0.08
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
