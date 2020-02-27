"""
Subsample the targets from a shapefile to a new shapefile.

.. program-output:: subsampletargets --help
"""
from random import sample
import os.path

import shapefile
import click

import uncoverml.mllog


def main(filename, outputdir, npoints):
    name = os.path.basename(filename).rsplit(".", 1)[0]

    # Read the shapefile
    file = shapefile.Reader(filename)
    shapes = file.shapes()
    records = file.records()
    items = list(zip(shapes, records))

    # Randomly sample the shapefile to keep n points
    remaining_items = sample(items, npoints)

    # Create a new shapefile using the data saved
    w = shapefile.Writer(shapefile.POINT)
    w.fields = list(file.fields)
    keep_shapes, _ = zip(*items)
    for shape, record in remaining_items:
        w.records.append(record)
    w._shapes.extend(keep_shapes)

    # Write out the
    outfile = os.path.join(outputdir, name + '_' + str(npoints))
    w.save(outfile)
