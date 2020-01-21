import sys
import os
import csv
from pprint import pprint

import click
import rasterio
import numpy  as np

@click.command()
@click.argument('directory')
@click.option('-o', '--output', help='output diagnostics to specified file')
def cli(directory, output):
    """
    Will output some basic diagnostic information for each TIF found
    in the provided directory.
    """
    diags = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            diags.append(diagnostic(os.path.join(dirpath, f)))

    if output:
        fieldnames = ['name', 'driver', 'crs', 'dtype', 'width', 'height', 'bands', 'nodata', 'ndv_percent']
        with open(output, 'w') as csvfile:
            w = csv.DictWriter(csvfile, fieldnames=fieldnames)
            w.writeheader()
            for diag in diags:
                w.writerow(diag)

    for diag in diags:
        printer(diag)

def diagnostic(filename):
    src = rasterio.open(filename)
    diag = {}
    diag['name'] = os.path.basename(filename)
    diag.update(src.meta)
    del diag['transform']
    diag['crs'] = diag['crs'].to_string()
    diag['bands'] = diag['count']
    del diag['count']
    diag['ndv_percent'] = [_percentage(src.read(i), src.nodata) for i in range(1, diag['bands'] + 1)]
    src.close()
    return diag
 
def printer(diag):
    print(f"Name:   {diag['name']}")
    print(f"Driver: {diag['driver']}")
    print(f"CRS:    {diag['crs']}")
    print(f"Dtype:  {diag['dtype']}")
    print(f"Width:  {diag['width']}")
    print(f"Height: {diag['height']}")
    print(f"Bands:  {diag['bands']}") 
    print(f"NDV:    {diag['nodata']}")
    print("No data percentages:")
    for i in range(diag['bands']):
        print(f"\tBand {i + 1}: {diag['ndv_percent'][i]}")
   
def _percentage(band, value):
    return np.count_nonzero(band == value) / band.flatten().shape[0] * 100
