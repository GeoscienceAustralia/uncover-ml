#!/usr/bin/env bash
#-----------------------------------------------------------------------------
#          Description
#
#          This script uses `gnu parallel`. 
#          It calls the scrip `crop_gtif.py` to do the real work in parallel.
#          Change `inputdir`, `outdir`, and `extents` as required.
#          For help with `extents`, see `help` in `crop_gtif.py`.
#          
# Dependencies:
#     gdal
#     gnu parallel
#-----------------------------------------------------------------------------

#PBS -P ge3
#PBS -q express
#PBS -l walltime=02:00:00,mem=32GB,ncpus=16,jobfs=100GB
#PBS -l wd
#PBS -j oe

module load parallel                                                            
module load gdal                                                                
                                                                                
inputdir=/g/data/ge3/covariates/national                                        
outdir=/g/data/ge3/covariates/national/TISATest                                 
mkdir -p ${outdir}                                                              
                                                                                
function crop {                                                                 
    outdir=$1                                                                   
    f=$2                                                                        
    gdalwarp -overwrite -te 138.1815078 -37.9222297 148.8765078 -29.7972297 ${f} ${outdir}/${f##*/};
}                                                                               
                                                                                
export -f crop                                                                  
                                                                                
ls ${inputdir}/*.tif | parallel crop ${outdir}                  
                                                    
