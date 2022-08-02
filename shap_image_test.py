import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging

import uncoverml as ls

from uncoverml import predict

from uncoverml import features
from uncoverml import mpiops
from uncoverml import geoio
from uncoverml.models import apply_masked, modelmaps
from uncoverml.optimise.models import transformed_modelmaps
from uncoverml.krige import krig_dict
from uncoverml import transforms
from uncoverml.config import Config
from uncoverml import predict

from uncoverml.scripts import uncoverml as uncli

with open('gbquantile/gbquantiles.model', 'rb') as f:
    state_dict = joblib.load(f)

model = state_dict['model']
config = state_dict['config']


def look_shp(file, config):
    targets = geoio.load_targets(shapefile=file,
                                 targetfield=config.target_property,
                                 conf=config)
    print('got targets')


if __name__ == '__main__':
    look_shp('Sites_of_interest.shp', config)
