#! /usr/bin/env python
"""
Description:
    Gather Metadata for the uncover-ml prediction output results


CreationDate:   31/05/19
Developer:      fei.zhang@ga.gov.au

Revision History:
    LastUpdate:     31/05/19   FZ
    LastUpdate:     dd/mm/yyyy  Who     Optional description
"""

# import section
import os
import sys
import pickle
from ppretty import ppretty


class MetaSummary():
    """
    Summary Description of the ML prediction output
    """

    def __init__(self, model_file):
        self.description = "Metadata for the ML results"
        self.creator = "login_user_fxz547"

        with open(model_file, 'rb') as f:
            state_dict = pickle.load(f)

        print(state_dict.keys())

        self.model = state_dict["model"]
        config = state_dict["config"]

        self.name = config.name


    def write_metadata(self, out_filename):
        with open(out_filename, 'w') as outf:
            outf.write("####### Metadata for the prediction results \n\n")

            objstr = ppretty(self, indent='  ', width=200, seq_length=200,
                             show_protected=True, show_static=True, show_properties=True, show_address=False,
                             str_length=200)

            outf.write(objstr)

            outf.write("\n####### End of Metadata \n")

        return out_filename


def write_prediction_metadata(model_file, out_filename="metadata.txt"):
    """
    write the metadata for this prediction result, into a human-readable YAML file.
    in order to make the ML results traceable and reproduceable (provenance)
    :return:
    """

    from ppretty import ppretty

    with open(model_file, 'rb') as f:
        state_dict = pickle.load(f)

    print(type(state_dict))
    print(state_dict.keys())

    model = state_dict["model"]
    print(type(model))

    # print("####################### -------------------------------  ####################")
    print("####################### wrting the properties of the prediction model  ####################")
    model_str = ppretty(model, indent='  ', width=40, seq_length=10,
                        show_protected=True, show_static=True, show_properties=True, show_address=False, str_length=150)

    config = state_dict["config"]

    # print("#######################  --------------------------------  ####################")
    print("#######################  writing the properties of the config  ####################")

    config_str = ppretty(config, indent='  ', width=200, seq_length=200,
                         show_protected=True, show_static=True, show_properties=True, show_address=False,
                         str_length=200)

    with open(out_filename, 'w') as outf:
        outf.write("####### Metadata for the prediction results ")

        outf.write("\n####### Summary of the ML Result \n")

        outf.write("\n###### Configuration Info \n")
        outf.write(config_str)

        outf.write("\n####### Model Info \n")
        outf.write(model_str)

    return out_filename


def main(mf):
    """
    define my main function
    :return:
    """

    obj= MetaSummary(mf)
    obj.write_metadata(out_filename='metatest.txt')

    return


# =============================================
# Section for quick test of this script
# ---------------------------------------------
if __name__ == "__main__":
    # call main function
    main(sys.argv[1])
