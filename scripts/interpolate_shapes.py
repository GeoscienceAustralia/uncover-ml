from functools import partial
from pathlib import Path
from scipy import interpolate
import csv
import numpy as np
import pandas as pd
import geopandas as gpd
from joblib import Parallel, delayed


def read_list_file(list_path: str):
    files = []
    csvfile = Path(list_path)
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        tifs = list(reader)
        tifs = [f[0].strip() for f in tifs
                if (len(f) > 0 and f[0].strip() and
                    f[0].strip()[0] != '#')]
    for f in tifs:
        files.append(Path(f).as_posix())
    return files


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

    ref_df = gpd.read_file("/g/data/ge3/aem_sections/AEM_covariates/NT_1_albers.shp", rows=1)

    # shapes_to_convert = [
    #     # "/g/data/ge3/data/AEM_shapes/AEM_ERC_ALL_interp_data_Albers_Ceno.shp",
    #     "/g/data/ge3/data/AEM_shapes/East_C_1_1_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/East_C_1_2_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/East_C_1_3_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/East_C_2_1_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/East_C_2_2_albers_mod.shp",
    #     "/g/data/ge3/data/AEM_shapes/East_C_2_2_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/East_C_2_3_albers_mod.shp",
    #     "/g/data/ge3/data/AEM_shapes/East_C_2_3_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/East_C_3_1_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/frome_1_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/frome_2_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/frome_3_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/frome_4_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/frome_5_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/frome_6_albers.shp",
    #     "/g/data/ge3/data/AEM_shapes/Mundi_Mundi_albers.shp",
    # ]
    list_path = "/g/data/ge3/sudipta/jobs/intersect_covariates/aem_2d_model_shapes/all_shapes.txt"
    shapes_to_convert = read_list_file(list_path)
    output_dir = Path('interpolated')
    output_dir.mkdir(exist_ok=True, parents=True)

    conductivity_cols = [c for c in ref_df.columns if c.startswith('cond_')]
    thickness_cols = [t for t in ref_df.columns if t.startswith('thick_')]
    new_thicknesses = np.array(ref_df.loc[0, thickness_cols].values.cumsum(), dtype=np.float32)

    rets = Parallel(
        n_jobs=-1,
        verbose=100,
    )(delayed(interpolate_shape)(s) for s in shapes_to_convert)