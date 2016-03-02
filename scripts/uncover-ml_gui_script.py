#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import click


@click.command()
def gui():
    fig, ax = plt.subplots()
    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)

    # the histogram of the data
    n, bins, patches = ax.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

    ax.set_xlabel('Smarts')
    ax.set_ylabel('Probability')
    ax.set_title('Histogram of IQ')

    ax.axis([40, 160, 0, 0.03])
    ax.text(60, .025, r'$\mu=100,\ \sigma=15$')
    ax.grid(True)

    plt.show()
