from functools import partial
from pathlib import Path
from scipy import interpolate
import numpy as np
import pandas as pd
import geopandas as gpd
from joblib import Parallel, delayed


def interpolate_1d(row, thicknesses):
    conductivities = row[conductivity_cols].values
    f = interpolate.interp1d(thicknesses, conductivities, bounds_error=False,
                             fill_value=(conductivities[0], conductivities[-1]), assume_sorted=True)
    new_conductivities = f(new_thicknesses)
    return new_conductivities


def interpolate_shape(shp):
    print(f"interpolating {shp}")
    df = gpd.read_file(shp)
    thicknesses = df.loc[0, thickness_cols].values.cumsum()
    interpolate_1d_local = partial(interpolate_1d, thicknesses=thicknesses)
    interpolated_conductivities = df.apply(
        func=interpolate_1d_local,
        axis=1,
        raw=False,
    )
    df.loc[:, conductivity_cols] = np.vstack(interpolated_conductivities)
    df.loc[:, thickness_cols] = np.tile(new_thicknesses, (df.shape[0], 1))
    target_shp = output_dir.joinpath(Path(shp).name)
    df.to_file(target_shp)
    print(f"wrote interpolated shapefile {target_shp}")


if __name__ == '__main__':

    ref_df = gpd.read_file("./NT_1_albers.shp", rows=1)

    shapes_to_convert = [
        './AEM_GAB_albers.shp',
        './Albany_1_albers_cleaned.shp',
        './Albany_2_albers_mod_cleaned.shp',
        './Murchison_1_albers_cleaned.shp',
        './Murchison_2_albers_cleaned.shp',
        './Murchison_3_albers.shp',
        './frome_1_albers.shp',
        './frome_2_albers.shp',
        './frome_3_albers.shp',
        './frome_4_albers.shp',
        './frome_5_albers.shp',
        './frome_6_albers.shp'
    ]
    output_dir = Path('interpolated')
    output_dir.mkdir(exist_ok=True, parents=True)

    conductivity_cols = [c for c in ref_df.columns if c.startswith('cond_')]
    thickness_cols = [t for t in ref_df.columns if t.startswith('thick_')]
    new_thicknesses = np.array(ref_df.loc[0, thickness_cols].values.cumsum(), dtype=np.float32)

    rets = Parallel(
        n_jobs=-1,
        verbose=100,
    )(delayed(interpolate_shape)(s) for s in shapes_to_convert)