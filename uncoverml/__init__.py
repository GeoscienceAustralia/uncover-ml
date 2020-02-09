import os
import pkg_resources

import uncoverml

__author__ = 'Geoscience Australia Mineral Systems Group, ' \
             'NICTA Spatial Inference Systems Team'
__email__ = 'daniel.steinberg@nicta.com.au, basaks@gmail.com'
__version__ = pkg_resources.get_distribution('uncover-ml').version

# Turn off MPI warning about network interface
os.environ['OMPI_MCA_btl_base_warn_component_unused'] = '0'

