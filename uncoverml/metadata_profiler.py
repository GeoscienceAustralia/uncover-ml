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
import uncoverml.git_hash as gits

from ppretty import ppretty



class MetadataSummary():
    """
    Summary Description of the ML prediction output
    """

    def __init__(self, model_file):

        path2mf = os.path.dirname(os.path.abspath(model_file))

        print (path2mf)

        self.description = "Metadata for the ML results"
        username = getpass.getuser()
        hostname = socket.gethostname()

        self.creator = username
        self.computename = hostname
        self.datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.git_hash = gits.git_hash

        with open(model_file, 'rb') as f:
            state_dict = pickle.load(f)

        print(state_dict.keys())

        self.model = state_dict["model"]
        print("####################### Info about the prediction model  ####################")
        model_str = ppretty(self.model, indent='  ', show_protected=True, show_static=True, show_address=False, str_length=50)
        print(model_str)

        self.config = state_dict["config"]
        self.name = self.config.name  # 'demo_regression'
        self.algorithm = self.config.algorithm  # 'svr'

        self.extent= ((-10, 100),(-40, 140))


        # self.performance_metric= {"json_file": 0.99}
        jsonfilename = "%s_scores.json"%(self.name)

        jsonfile = os.path.join(path2mf, jsonfilename)

        with open(jsonfile) as json_file:
            self.model_performance_metrics = json.load(json_file)


    def write_metadata(self, out_filename):
        """
        write the metadata for this prediction result, into a human-readable txt file.
        in order to make the ML results traceable and reproduceable (provenance)
        :return:
        """

        with open(out_filename, 'w') as outf:
            outf.write("# Metadata Profile for the Prediction Results")

            outf.write("\n\n############ Software Environment ###########\n\n")
            outf.write("Creator = %s \n"%self.creator)
            outf.write("Computer = %s \n"%self.computename)
            outf.write("ML Algorithm = %s \n"%self.algorithm)
            outf.write("uncoverml git-hash = %s\n"%self.git_hash)
            outf.write("Datetime = %s \n"%self.datetime)

            outf.write("\n\n############ Performance Matrics ###########\n\n")
            #outf.write(str(self.model_performance_metrics))
            for keys, values in self.model_performance_metrics.items():
                outf.write("%s = %s\n"%(keys,values))

            outf.write("\n\n############ Configuration ###########\n\n")

            conf_str = ppretty(self.config, indent='  ', width=200, seq_length=200,
                             show_protected=True, show_static=True, show_properties=True, show_address=False,
                             str_length=200)

            outf.write(conf_str)

            outf.write("\n\n############ Model ###########\n\n")
            model_str = ppretty(self.model, indent='  ', show_protected=True, show_static=True, show_address=False,
                                str_length=50)
            outf.write(model_str)


            outf.write("\n\n############ The End of Metadata ###########\n\n")

        return out_filename



def main(mf):
    """
    define my main function
    :return:
    """

    obj= MetadataSummary(mf)
    obj.write_metadata(out_filename='metatest.txt')

    return


# =============================================
# Section for quick test of this script
# ---------------------------------------------
if __name__ == "__main__":
    # call main function
    main(sys.argv[1])