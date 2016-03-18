#! /usr/bin/env python3
"""
A demo script that ties some of the command line utilities together in a
pipeline

TODO: Replicate this with luigi or joblib
"""

from os import path, mkdir
from glob import glob
from subprocess import run, Popen, PIPE, CalledProcessError


# Settings
data_dir = path.join(path.expanduser("~"), "data/GA-cover")
proc_dir = path.join(data_dir, "processed")

target_file = "geochem_sites.shp"
target_var = "Na_ppm_i_1"
target_json = path.join(proc_dir, "{}_{}"
                        .format(path.splitext(target_file)[0], target_var))


def main():

    # Make processed dir if it does not exist
    if not path.exists(proc_dir):
        mkdir(proc_dir)
        print("Made processed dir")

    # Make pointspec and hdf5 for targets
    cmd = ["maketargets", path.join(data_dir, target_file), target_var,
           "--outfile", target_json]

    if try_run_checkfile(cmd, target_json + ".json"):
        print("Made targets")

    # Extract feats for training
    tifs = glob(path.join(data_dir, "*.tif"))
    if len(tifs) == 0:
        raise PipeLineFailure("No geotiffs found in {}!".format(data_dir))

    for tif in tifs:
        outfile = path.join(proc_dir, path.splitext(path.basename(tif))[0])
        cmd = ["extractfeats", target_json + ".json", tif, outfile,
               "--splits", "1", "--standalone"]
        if try_run_checkfile(cmd, outfile + ".hdf5"):
            print("Made features for {}.".format(path.basename(tif)))

    print("Done!")


class PipeLineFailure(Exception):
    pass


def try_run_checkfile(cmd, checkfile):

    if not path.exists(checkfile):
        try_run(cmd)
        return True

    return False


def try_run(cmd):

    try:
        run(cmd, check=True)
    except CalledProcessError:
        print("\n--------------------\n")
        raise


if __name__ == "__main__":
    main()
