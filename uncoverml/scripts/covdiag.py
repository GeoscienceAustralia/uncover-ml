import sys
import os
import glob
import csv
import itertools
from pprint import pprint

import click
import rasterio
import numpy as np

from uncoverml import mpiops

@click.command()
@click.argument('directory')
def cli(directory):
    """
    Will output some basic diagnostic information for each TIF found
    in the provided directory.
    """
    diags = []
    paths = mpiops.run_once(glob.glob, os.path.join(directory, '*.tif'))
    if mpiops.chunk_index == 0:
        print(f"Found {len(paths)} geotiffs, retrieving information...")
    this_chunk_paths = np.array_split(paths, mpiops.chunks)[mpiops.chunk_index]
    for f in this_chunk_paths:
        diag = diagnostic(f)
        if diag is not None:
            diags.append(diag)
            print(f"Processed '{f}'")

    diags = mpiops.comm.gather(diags, root=0)
    mpiops.comm.barrier()

    if mpiops.chunk_index == 0:
        diags = list(itertools.chain.from_iterable(diags))

        fieldnames = ['name', 'driver', 'crs', 'dtype', 'width', 'height', 'bands', 'nodata', 'ndv_percent', 'min', 'max']
        with open('covdiag.csv', 'w') as csvfile:
            w = csv.DictWriter(csvfile, fieldnames=fieldnames)
            w.writeheader()
            for diag in diags:
                w.writerow(diag)

        with open('covdiag.txt', 'w') as txtfile:
            for diag in diags:
                pretty_string = printer(diag)
                print(pretty_string)
                txtfile.write(pretty_string + '\n')

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
    diag['ndv_percent'] = []
    diag['min'] = []
    diag['max'] = []
    for i in range(1, diag['bands'] + 1):
        band = src.read(i)
        diag['ndv_percent'].append(_percentage(band, src.nodata, diag['width'] * diag['height']))
        diag['min'].append(np.min(band))
        diag['max'].append(np.max(band))
    src.close()
    return diag
 
def printer(diag):
    pretty_string = (
        f"Name:   {diag['name']}\n"
        f"Driver: {diag['driver']}\n"
        f"CRS:    {diag['crs']}\n"
        f"Dtype:  {diag['dtype']}\n"
        f"Width:  {diag['width']}\n"
        f"Height: {diag['height']}\n"
        f"Bands:  {diag['bands']}\n" 
        f"NDV:    {diag['nodata']}\n"
        "Band stats:\n"
    )
    for i in range(diag['bands']):
        pretty_string += (
            f"\tBand {i + 1} NDV: {diag['ndv_percent'][i]}%\n"
            f"\tBand {i + 1} min: {diag['min'][i]}\n"
            f"\tBand {i + 1} max: {diag['max'][i]}\n"
        )
    return pretty_string
   
def _percentage(band, ndv, n_elements):
    return np.count_nonzero(band == ndv) / n_elements * 101
