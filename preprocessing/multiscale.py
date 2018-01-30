#!/bin/env python
"""
Description:
    Class for generating multiscale covariates based on 2D wavelet
    decomposition and reconstruction.
References:

CreationDate:   04/12/17
Developer:      rakib.hassan@ga.gov.au

Revision History:
    LastUpdate:     04/12/17   RH
    LastUpdate:     dd/mm/yyyy  Who     Optional description
"""

import os

import numpy as np
from mpi4py import MPI
import glob
from collections import defaultdict
import pywt
import click

import gdal
from gdalconst import *
import uuid

import logging
log = logging.getLogger('multiscale')

class Multiscale():
    def __init__(self, input, output_folder,
                 level=2, file_extension='.tif',
                 mother_wavelet_name='coif6',
                 extension_mode='smooth',
                 extrapolate=True,
                 max_search_dist=400,
                 smoothing_iterations=10,
                 keep_level=()):
        """
        :param input: a file containing a list of input files (with full path) or a folder containing
                      input files
        :param output_folder: output folder
        :param level: maximum decomposition level
        :param file_extension: file extension e.g. '.tif'
        :param mother_wavelet_name: name of mother wavelet
        :param extension_mode: method of signal extrapolation during computation of wavelet transforms
        :param extrapolate: note, this is separate to the extrapolation done internally by pywavelets,
                            controlled by extention_mode. This parameter controls whether image values
                            are extrapolated into masked regions with NO_DATA_VALUE assigned to them
        :param max_search_dist: this parameter sets the search radius -- in number of pixels -- of
                                extrapolation, controlled by the previous parameter
        :param smoothing_iterations: number of smoothing iterations to be performed after the extrapolation,
                                     controlled by the previous two parameters
        :param keep_level: list of integers that specify the levels to save, while the rest are culled. By
                           default all levels are saved
        """
        self._input = input
        self._output_folder = output_folder
        self._level = level
        self._file_extension = file_extension
        self._mother_wavelet_name = mother_wavelet_name
        self._extension_mode = extension_mode
        self._extrapolate = extrapolate
        self._max_search_dist = max_search_dist
        self._smoothing_iterations = smoothing_iterations
        self._keep_level = keep_level

        self._comm = MPI.COMM_WORLD
        self._nproc = self._comm.Get_size()
        self._chunk_index = self._comm.Get_rank()
        self._proc_files = defaultdict(list)
        self._rm_list = []

        self.__split_work()
    # end func

    def __get_files(self):
        """
        Function to get a list of input files from a text file or a folder.

        :return: list of files
        """
        files = None
        if(os.path.isdir(self._input)):
            log.info(' Searching for input files with extension %s in folder %s'%
                     (self._file_extension, self._input))
            # prepare case insensitive glob pattern,
            # e.g. for '.pdf', this will produce '*.[Pp][Dd][Ff]'
            if (self._file_extension.count('.') != 1):
                raise (RuntimeError, 'Invalid file extension')

            glob_pattern = '*' + ''.join(sum(map(lambda x: ['[%s%s]' % (a.upper(), a.lower())
                                                              if a.isalpha() else a for a in list(x)],
                                                 self._file_extension), []))

            files = glob.glob(os.path.join(self._input, glob_pattern))
        elif(os.path.isfile(self._input)):
            try:
                fh = open(self._input)
                files = fh.read().splitlines()
                fh.close()
            except:
                raise(RuntimeError, 'Failed to read input file')
        log.info(' Found %d files to process ' % len(files))
        return files
    # end func

    def __split_work(self):
        """
        Splits up workload and sends each processor a list of files to process.
        """

        if(self._chunk_index==0):
            files = self.__get_files()
            count = 0
            for iproc in np.arange(self._nproc):
                for ifile in np.arange(np.divide(len(files), self._nproc)):
                    self._proc_files[iproc].append([files[count], str(uuid.uuid4())])
                    count += 1
            # end for
            for iproc in np.arange(np.mod(len(files), self._nproc)):
                self._proc_files[iproc].append([files[count], str(uuid.uuid4())])
                count += 1
        # end if

        # broadcast workload to all procs
        log.info(' Distributing workload over %d processors'%(self._nproc))
        self._proc_files = self._comm.bcast(self._proc_files, root=0)

        #print 'proc: %d, %d files\n========='%(mpiops.chunk_index,
        #                                       len(self._proc_files[mpiops.chunk_index]))
        #for f in self._proc_files[mpiops.chunk_index]: print f
    # end func

    def __generate_reconstructions(self, fname, uuid):
        """
        Computes wavelet decompositions and reconstructions.

        :param fname: file name
        :param uuid: universally unique id to be used as a tag to avoid file name collisions when
                     creating temporary files -- this of course is only useful when multiple
                     parallel 'multicale' jobs are running on a given node
        """

        # need all data at once
        src_ds = gdal.Open(fname, gdal.GA_ReadOnly)
        od = None
        if(src_ds.GetRasterBand(1).GetMaskBand() != None):
            driver = gdal.GetDriverByName('GTiff')

            mem_driver = gdal.GetDriverByName('MEM')
            scratch = mem_driver.CreateCopy('', src_ds, strict=0)

            sb = scratch.GetRasterBand(1)
            nodataval = sb.GetNoDataValue()

            if(nodataval is not None and self._extrapolate==False):
                log.warning(' NO_DATA_VALUES found in raster %s, but not extrapolating values. This may'%(fname)+\
                            ' cause \'ringing\' artefacts at the edges')
            elif(nodataval is not None and self._extrapolate):
                log.info(' Extrapolating raster %s by %d pixels'%(fname, self._max_search_dist))
                result = gdal.FillNodata(targetBand=sb, maskBand=None,
                                         maxSearchDist=self._max_search_dist,
                                         smoothingIterations=self._smoothing_iterations,
                                         options=['TEMP_FILE_DRIVER=MEM'])

            od = sb.ReadAsArray()
            # set NO_DATA_VALUE pixels to the global mean. Note that pywavelets cannot handle
            # masked values
            od[od==nodataval] = np.mean(od[od!=nodataval])

            # clean up
            scratch = None
        else:
            od = src_ds.GetRasterBand(1).ReadAsArray()

        # generate wavelet decompositions up to required level
        d = od
        assert(d.ndim==2)
        #print('orig shape:', d.shape)

        for i in np.arange(self._level):
            r = pywt.dwt2(d, self._mother_wavelet_name,
                          mode=self._extension_mode)

            fn, _ = os.path.splitext(os.path.basename(fname))
            tfn = os.path.join(self._output_folder, '%s.dwt2.level_%03d.%s.npy'%(fn, i+1, uuid))

            np.save(tfn, r[0])
            d = r[0]
            #print(d.shape)
            self._rm_list.append(tfn)
        # end for

        # reconstruct each level, starting from the highest
        for l in np.arange(1, self._level+1)[::-1]:

            # Culling reconstructed levels based on user-selection
            if(len(self._keep_level)):
                if(l not in self._keep_level): continue

            fn, _ = os.path.splitext(os.path.basename(fname))
            tfn = os.path.join(self._output_folder, '%s.dwt2.level_%03d.%s.npy'%(fn, l, uuid))
            d = np.load(tfn)

            log.debug('\tReconstructing level: %d'%(l))
            #print(d.shape)
            for i in np.arange(1, l+1):
                r = pywt.idwt2([d, [np.zeros(d.shape),
                                    np.zeros(d.shape), np.zeros(d.shape)]],
                               self._mother_wavelet_name,
                               mode=self._extension_mode)
                d = r
                #print(l, i, r.shape)
            # end for

            p = np.array(d.shape) - np.array(od.shape)
            #print(p, d.shape, od.shape)
            psx = pex = psy = pey = None
            if (p[0] % 2):
                psx = np.floor(p[0] / 2.)
                pex = np.ceil(p[0] / 2.)
            else:
                psx = np.floor(p[0] / 2.)
                pex = np.floor(p[0] / 2.)

            if (p[1] % 2):
                psy = np.floor(p[1] / 2.)
                pey = np.ceil(p[1] / 2.)
            else:
                psy = np.floor(p[1] / 2.)
                pey = np.floor(p[1] / 2.)

            psx, pex, psy, pey = np.int_([psx, pex, psy, pey])
            #print psx,pex,psy,pey

            if(psx != 0 or pex != 0):
                d = d[psx:-pex, :]
            if(psy != 0 or pey != 0):
                d = d[:, psy:-pey]

            #print d.shape
            #print np.min(d), np.max(d)
            #print '\n\n'

            if(d.shape != od.shape):
                print d.shape, od.shape
                raise(RuntimeError, 'Error encountered in wavelet reconstruction.')

            fn,ext = os.path.splitext(os.path.basename(fname))
            ofn = os.path.join(self._output_folder, '%s.level_%03d%s'%(fn,l,ext))
            of = driver.CreateCopy(ofn, src_ds, strict=0)
            of.GetRasterBand(1).WriteArray(d)
            of = None
        # end for

        src_ds = None
        # clean up temporary files
        for fn in self._rm_list:
            os.system('rm -rf %s'%fn)
    # end func

    def process(self):
        """
        Iterates over a list of files and processes them
        """

        for f, uuid in self._proc_files[self._chunk_index]:
            log.info(' Processing %s..'%(f))
            self.__generate_reconstructions(f, uuid)
        # end for
    # end func
# end class

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument('input', required=True,
                type=click.Path(exists=True))
@click.argument('output-folder', required=True,
                type=click.Path(exists=True))
@click.argument('max-level', required=True,
                type=np.int8)
@click.option('--file-extension', default='.tif',
              help='File extension to use (e.g. \'.tif\') to search for input files; only applicable'
                   'if the \'input\' argument is a folder.',
              type=str)
@click.option('--mother-wavelet', default='coif6',
              help='Name of the mother wavelet',
              type=click.Choice(['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2',  'bior2.4',  'bior2.6',  'bior2.8',
                               'bior3.1', 'bior3.3',  'bior3.5',  'bior3.7',  'bior3.9',  'bior4.4',  'bior5.5',
                               'bior6.8',  'coif1',  'coif2',  'coif3',  'coif4',  'coif5',  'coif6',  'coif7',
                               'coif8', 'coif9', 'coif10',  'coif11',  'coif12',  'coif13',  'coif14',  'coif15',
                               'coif16', 'coif17', 'db1',  'db2',  'db3',  'db4',  'db5',  'db6',  'db7',  'db8',
                               'db9',  'db10',  'db11',  'db12',  'db13',  'db14',  'db15',  'db16',  'db17',
                               'db18', 'db19',  'db20',  'db21',  'db22',  'db23',  'db24',  'db25',  'db26',
                               'db27', 'db28', 'db29',  'db30',  'db31',  'db32',  'db33',  'db34', 'db35', 'db36',
                               'db37', 'db38', 'dmey', 'haar', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4',
                               'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4',
                               'rbio5.5', 'rbio6.8', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9',
                               'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18',
                               'sym19', 'sym20']))
@click.option('--extension-mode', default='smooth',
              help="Signal extension mode used for padding",
              type=click.Choice(['zero', 'constant', 'symmetric', 'reflect', 'periodic', 'smooth', 'periodization']))
@click.option('--extrapolate', default=True,
              type=bool,
              help="Extrapolate raster if NO_DATA_VALUES are found. 'Ringing' artefacts can result near sharp contrasts"
                   " in image values -- especially at the edges of NO_DATA_VALUE regions. By extrapolating image values"
                   " to regions of NO_DATA_VALUE, 'ringing' artefacts can be pushed further outward, away from the region"
                   " of interest in the original image. This parameter has no effect when the input raster has no masked"
                   " regions")
@click.option('--max-search-dist', default=500,
              help="Maximum search distance (in pixels) for extrapolating image values to regions of NO_DATA_VALUE in "
                   "input raster; not used if raster has no masked regions")
@click.option('--smoothing-iterations', default=10,
              help="Number of smoothing iterations used for smoothing extrapolated values; see option --max-search-dist")
@click.option('--keep-level', multiple=True, type=(int),
              help="Level to keep. Note that by default all levels up to max-level are saved, which may cause disk"
                   " space issues. This option allows users to save only those levels that are of interest; e.g. to "
                   "keep only levels 5 and 6, this option must be repeated twice for the corresponding levels")
@click.option('--log-level', default='INFO',
              help="Logging verbosity",
              type=click.Choice(['DEBUG', 'INFO', 'WARN']))
def process(input, output_folder, max_level, file_extension,
            mother_wavelet, extension_mode, extrapolate, max_search_dist,
            smoothing_iterations, keep_level, log_level):
    """
    INPUT: Path to raster files, or a file containing a list of raster file names (with full path)\n
    OUTPUT_FOLDER: Output folder \n
    MAX_LEVEL: Maximum level up to which wavelet reconstructions are to be computed

    Example usage:
    mpirun -np 2 python multiscale.py filelist.txt /tmp/output 10 --max-search-dist 500

    Running in Cluster Environments:
    This script requires all raster data to be loaded into memory for processing. Furthermore, each
    raster being processed requires 4 times as much memory, e.g. a 7 GB raster will require ~28 GB
    of RAM. Hence, for parallel runs, one must be judicious in terms of allocating processors,
    bearing in mind the amount of memory available on a given compute node on the cluster.
    """

    logMap = {'DEBUG':logging.DEBUG, 'INFO':logging.INFO, 'WARN':logging.WARNING}
    logging.basicConfig(level=logMap[log_level])

    m = Multiscale(input, output_folder, level=max_level, file_extension=file_extension,
                   mother_wavelet_name=mother_wavelet, extension_mode=extension_mode,
                   extrapolate=extrapolate, max_search_dist=max_search_dist,
                   smoothing_iterations=smoothing_iterations, keep_level=keep_level)
    m.process()
    return
# end

# =============================================
# Quick test
# =============================================
if __name__ == "__main__":
    # call main function
    process()
