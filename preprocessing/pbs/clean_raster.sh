#!/bin/bash
# usage: make sure you have gdal_calc.py installed or download it from here:https://svn.osgeo.org/gdal/trunk/gdal/swig/python/scripts/gdal_calc.py
# then use: bash clean_raster.sh input.tif output.tif 0 -3.40282346638528E+038
# input.tif: input geotiff
# output.tif: output geotiff name
# Only keep values above '0'
# NoDataValue used for output geotiff: -3.40282346638528E+038
input=$(basename "$1")
gdal_calc.py -A $1 --outfile=bool_$input --calc="A>=$3"
rm -f $2
gdal_calc.py -A $1 -B bool_$input --outfile=$2 --calc="A*B" --NoDataValue=$4
rm bool_$input