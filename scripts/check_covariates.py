from joblib import delayed, Parallel
from pathlib import Path
import rasterio as rio
import numpy as np
import pandas as pd

dir = "configs/data/sirsam"

mask = Path(dir).joinpath("dem_foc2.tif")

with rio.open(mask) as geotif:
    mask_raster = geotif.read(1, masked=True)


def _parallel_read(r):
    try:
        with rio.open(r) as geotif:
            raster = geotif.read(1, masked=True)
            m = geotif.meta
            m['crs'] = m['crs'].to_string()
            t = m.pop('transform')
            m['pixsize_x'] = t[0]
            m['top_left_x'] = t[2]
            m['pixsize_y'] = -t[4]
            m['top_left_y'] = t[5]
            raster.mask = raster.mask | mask_raster.mask  # we are not interested in masked areas
            m['all_finite'] = np.all(np.isfinite(raster))
            m['any_nan'] = np.any(np.isnan(raster))
            m['any_large'] = np.any(np.abs(raster) > 1e10)
            # raster_attrs[r.stem] = m
        return m
    except Exception as e:
        print(r)
        print(e)
        return [None] * 14


rets = Parallel(
    n_jobs=-1,
    verbose=100,
)(delayed(_parallel_read)(r) for r in Path(dir).glob("*.tif"))

raster_attrs = {r.stem: v for r, v in zip(Path(dir).glob("*.tif"), rets)}

df = pd.DataFrame.from_dict(raster_attrs)
df.to_csv("quality.csv")
