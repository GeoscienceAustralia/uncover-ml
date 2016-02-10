import numpy as np
import sys
import shapefile
    
sf = shapefile.Reader(sys.argv[1])
fields = [f[0] for f in sf.fields[1:]]
usable_fields = [i for i,f in enumerate(sf.fields[1:]) if f[1] == 'N']

coords = []
properties = []
for shape, record in zip(sf.iterShapes(), sf.iterRecords()):
    # process_poly expects this format (geojson)
    coords.append(list(shape.__geo_interface__['coordinates']))
    properties.append([float(record[i]) for i in usable_fields])

label_coords = np.array(coords)
labels = np.array(properties)
field_names = np.array([fields[i] for i in usable_fields])


import IPython; IPython.embed()

