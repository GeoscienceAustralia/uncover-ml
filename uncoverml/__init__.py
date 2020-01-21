import os
import pkg_resources

import uncoverml

__author__ = 'Geoscience Australia Mineral Systems Group, ' \
             'NICTA Spatial Inference Systems Team'
__email__ = 'daniel.steinberg@nicta.com.au, basaks@gmail.com'
__version__ = pkg_resources.get_distribution('uncover-ml').version

if 'UNCOVERML_SRC' not in os.environ:
    os.environ['UNCOVERML_SRC'] = os.path.split(uncoverml.__path__[0])[0]

# Turn off MPI warning about network interface
os.environ['OMPI_MCA_btl_base_warn_component_unused'] = '0'

