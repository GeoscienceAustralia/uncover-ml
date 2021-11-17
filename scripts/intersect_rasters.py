from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from joblib import Parallel, delayed

data_location = \
    Path("/g/data/ge3/covariates/national_albers_filled_new/albers_cropped/")
# Read points from shapefile

shapefile_location = Path("/g/data/ge3/aem_sections/AEM_covariates/")

geotifs = {
    "relief_radius4.tif": "relief4",
    "national_Wii_RF_multirandomforest_prediction.tif": "mrf_pred",
    "MvrtpLL_smooth.tif": "mrvtpLL_s",
    "MvrtpLL_fin.tif": "mvrtpLL_f",
    "LOC_distance_to_coast.tif": "LOC_dis",
    "Gravity_land.tif": "gravity",
    "dem_fill.tif": "dem",
    "Clim_Prescott_LindaGregory.tif": "clim_linda",
    "clim_PTA_albers.tif": "clim_alber",
    "SagaWET9cell_M.tif": "sagawet",
    "ceno_euc_aust1.tif": "ceno_euc"
}


# local
# data_location = Path("configs/data")
# tif_local = data_location.joinpath('LATITUDE_GRID1.tif')
# shapefile_location = Path("configs/data")
# shp = shapefile_location.joinpath('geochem_sites.shp')

downscale_factor = 2  # keep 1 point in a 2x2 cell


def intersect_and_sample_shp(shp: Path):
    print("====================================\n", f"intersecting {shp.as_posix()}")
    pts = gpd.read_file(shp)
    coords = np.array([(p.x, p.y) for p in pts.geometry])
    geom = pd.DataFrame(coords, columns=['POINT_X', 'POINT_Y'], index=pts.index)
    pts = pts.merge(geom, left_index=True, right_index=True)
    tif_name = list(geotifs.keys())[0]
    tif = data_location.joinpath(tif_name)
    orig_cols = pts.columns
    with rasterio.open(tif) as src:
        # resample data to target shape
        data = src.read(
            out_shape=(
                src.count,
                int(src.height / downscale_factor),
                int(src.width / downscale_factor)
            ),
            resampling=rasterio.enums.Resampling.bilinear
        )
        # scale image transform
        transform = src.transform * src.transform.scale(
            (src.width / data.shape[-1]),
            (src.height / data.shape[-2])
        )
        pts["rows"], pts["cols"] = rasterio.transform.rowcol(transform, coords[:, 0], coords[:, 1])

    pts_deduped = pts.sort_values(
        by=['POINT_X', 'POINT_Y'], ascending=[True, True]
        ).groupby(by=['rows', 'cols'], as_index=False).first()[orig_cols]
    # pts_deduped = pts.drop_duplicates(subset=['rows', 'cols'])[orig_cols]
    coords_deduped = np.array([(p.x, p.y) for p in pts_deduped.geometry])

    for k, v in geotifs.items():
        print(f"adding {k} to output dataframe")
        with rasterio.open(data_location.joinpath(k)) as src:
            pts_deduped[v] = [x[0] for x in src.sample(coords_deduped)]

    pts_deduped = gpd.GeoDataFrame(pts_deduped, geometry=pts_deduped.geometry)
    pts_deduped.to_file(Path('out_resampled').joinpath(shp.name))
    # pts.to_csv(Path("out").joinpath(shp.stem + ".csv"), index=False)

intersect_and_sample_shp(shp)
rets = Parallel(
    n_jobs=-1,
    verbose=100,
)(delayed(intersect_and_sample_shp)(s) for s in shapefile_location.glob("*.shp"))

