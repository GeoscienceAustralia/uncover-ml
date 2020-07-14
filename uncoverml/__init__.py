import os
import pkg_resources

import uncoverml

__author__ = 'Geoscience Australia, Mineral Systems Branch, ' \
             'NICTA Spatial Inference Systems Team (now Data 61)'
__email__ = 'daniel.steinberg@nicta.com.au, basaks@gmail.com, brenainn.moushall@ga.gov.au'
__version__ = pkg_resources.get_distribution('uncover-ml').version

# Turn off MPI warning about network interface
os.environ['OMPI_MCA_btl_base_warn_component_unused'] = '0'

