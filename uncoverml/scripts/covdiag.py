import sys
import os
import glob
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
    paths = glob.glob(os.path.join(directory, '*.tif'))
    print(f"Found {len(paths)} geotiffs, retrieving informtion...")
    for f in paths:
        diag = diagnostic(f)
        if diag is not None:
            diags.append(diag)
            print(f"Processed '{f}'")

    if output:
        fieldnames = ['name', 'driver', 'crs', 'dtype', 'width', 'height', 'bands', 'nodata', 'ndv_percent']
        with open(output, 'w') as csvfile:
            w = csv.DictWriter(csvfile, fieldnames=fieldnames)
            w.writeheader()
            for diag in diags:
                w.writerow(diag)

    for diag in diags:
        printer(diag)
        print()

    print("Finished")

def diagnostic(filename):
    try:
        src = rasterio.open(filename)
    except rasterio.errors.RasterioIOError:
        print(f"Couldn't load '{filename}'\n")
        return None
    diag = {}
    diag['name'] = os.path.basename(filename)
    diag.update(src.meta)
    del diag['transform']
    diag['crs'] = diag['crs'].to_string()
    diag['bands'] = diag['count']
    del diag['count']
    diag['ndv_percent'] = [_percentage(src.read(i), src.nodata, diag['width'] * diag['height']) for i in range(1, diag['bands'] + 1)]
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
   
def _percentage(band, ndv, n_elements):
    return np.count_nonzero(band == ndv) / n_elements * 100
