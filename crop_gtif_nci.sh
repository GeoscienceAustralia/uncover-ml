#!/usr/bin/env bash
#PBS -P ge3
#PBS -q express
#PBS -l walltime=02:00:00,mem=32GB,ncpus=16,jobfs=100GB
#PBS -l wd
#PBS -j oe

# module load parallel
# module load gdal

inputdir=${PWD}
outdir=MBTest
mkdir -p ${outdir}

function crop {
        outdir=$1
        f=$2
        python crop_gtif.py -i ${f} -o ${outdir}/${f##*/} -e '138.1815078 -37.9222297 148.8765078 -29.7972297';
}

export -f crop

ls ${inputdir}/*.tif | parallel crop ${outdir}

