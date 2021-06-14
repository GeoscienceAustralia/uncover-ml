#! /usr/bin/env python
"""
Description:
    Gather Metadata for the uncover-ml prediction output results:

Reference: email 2019-05-24
Overview
Creator: (person who generated the model)
Model;
    Name:
    Type and date:
    Algorithm:
    Extent: Lat/long - location on Australia map?

SB Notes: None of the above is required as this information will be captured in the yaml file.

Model inputs:

1.      Covariates - list (in full)
2.      Targets: path to shapefile:  csv file
SB Notes: Only covaraite list file. Targets and path to shapefile is not required as this is available in the yaml file. May be the full path to the shapefile has some merit as one can specify partial path.

Model performance
       JSON file (in full)
SB Notes: Yes

Model outputs

1.      Prediction grid including path
2.      Quantiles Q5; Q95
3.      Variance:
4.      Entropy:
5.      Feature rank file
6.      Raw covariates file (target value - covariate value)
7.      Optimisation output
8.      Others ??
SB Notes: Not required as these are model dependent, and the metadata will be contained in each of the output geotif file.


Model parameters:
1.      YAML file (in full)
2.      .SH file (in full)
SB Notes: The .sh file is not required. YAML file is read as a python dictionary in uncoverml which can be dumped in the metadata.


CreationDate:   31/05/19
Developer:      fei.zhang@ga.gov.au

Revision History:
    LastUpdate:     31/05/19   FZ
    LastUpdate:     dd/mm/yyyy  Who     Optional description
"""

# import section
import os
import sys
import json
import pickle
import datetime
import getpass
import socket

from ppretty import ppretty

import uncoverml


class MetadataSummary():
    """
    Summary Description of the ML prediction output
    """
    def __init__(self, model, config):
        self.model = model
        self.description = "Metadata for the ML results"
        username = getpass.getuser()
        hostname = socket.gethostname()

        self.creator = username
        self.computename = hostname
        self.datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.version = uncoverml.__version__

        model_str = ppretty(self.model, indent='  ', show_protected=True, show_static=True,
                            show_address=False, str_length=50)

        self.config = config
        self.name = self.config.name  # 'demo_regression'
        self.algorithm = self.config.algorithm  # 'svr'

        self.extent = ((-10, 100),(-40, 140))

        if config.cross_validate and os.path.exists(config.crossval_scores_file):
            with open(config.crossval_scores_file) as sf:
                self.model_performance_metrics = json.load(sf)
        else:
            self.model_performance_metrics = None


    def write_metadata(self, out_filename):
        """
        write the metadata for this prediction result, into a human-readable txt file.
        in order to make the ML results traceable and reproduceable (provenance)
        """
        with open(out_filename, 'w') as outf:
            outf.write("# Metadata Profile for the Prediction Results")

            outf.write("\n\n############ Software Environment ###########\n\n")
            outf.write("Creator = %s \n"%self.creator)
            outf.write("Computer = %s \n"%self.computename)
            outf.write("ML Algorithm = %s \n"%self.algorithm)
            outf.write("Version = %s\n"%self.version)
            outf.write("Datetime = %s \n"%self.datetime)

            outf.write("\n\n############ Performance Metrics ###########\n\n")
            if self.model_performance_metrics:
                for keys, values in self.model_performance_metrics.items():
                    outf.write("%s = %s\n"%(keys, values))

            outf.write("\n\n############ Configuration ###########\n\n")

            conf_str = ppretty(self.config, indent='  ', width=200, seq_length=200,
                               show_protected=True, show_static=True, show_properties=True, 
                               show_address=False, str_length=200)

            outf.write(conf_str)

            outf.write("\n\n############ Model ###########\n\n")
            model_str = ppretty(self.model, indent='  ', show_protected=True, show_static=True, 
                                show_address=False, str_length=50)

            outf.write(model_str)

            outf.write("\n\n############ The End of Metadata ###########\n\n")

        return out_filename
