"""
Convert a Geotiff to a KMZ archive for display on Google and Cesium maps (e.g.
National Map).

.. program-output:: tiff2kmz --help
"""
import os.path
import click as cl
import simplekml
from PIL import Image

from uncoverml import geoio
from uncoverml import mpiops

def main(tiff, outfile, overlayname):
    """
    Turn a geotiff into a KMZ that can be dragged onto an instance of Terria
    Map. This also constructs a JPEG of the Geotiff, as it is required for the
    KMZ.
    """
    # This runs on the root node only
    if not mpiops.leader_world:
        return

    # Get tiff info
    I = geoio.Image(tiff)

    # Save tiff as jpeg
    if outfile is not None:
        outfile = os.path.splitext(outfile)[0]
    else:
        outfile = os.path.splitext(tiff)[0]
    jpg = outfile + ".jpg"

    # Convert tiff to jpg
    Im = Image.open(tiff)
    Im.save(jpg)

    # Construct KMZ
    kml = simplekml.Kml()
    if overlayname is None:
        overlayname = os.path.basename(outfile)
    ground = kml.newgroundoverlay(name=overlayname)
    ground.icon.href = jpg
    ground.latlonbox.west = I.xmin
    ground.latlonbox.east = I.xmax
    ground.latlonbox.north = I.ymax
    ground.latlonbox.south = I.ymin

    kml.savekmz("{}.kmz".format(outfile))
