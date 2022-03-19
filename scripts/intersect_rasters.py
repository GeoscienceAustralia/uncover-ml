from typing import Dict
import csv
from collections import defaultdict
from os import path
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from joblib import Parallel, delayed


def read_list_file(list_path: str):
    files = []
    csvfile = path.abspath(list_path)
    with open(csvfile, 'r') as f:
        reader = csv.reader(f)
        tifs = list(reader)
        tifs = [f[0].strip() for f in tifs
                if (len(f) > 0 and f[0].strip() and
                    f[0].strip()[0] != '#')]
    for f in tifs:
        files.append(path.abspath(f))
    return files


def generate_key_val(gl, shorts):
    eight_char_name = Path(gl).name[:8]
    if eight_char_name not in shorts:
        shorts[eight_char_name] += 1
    else:
        shorts[eight_char_name] += 1
    return gl, eight_char_name + str(shorts[eight_char_name])


# import IPython; IPython.embed(); import sys; sys.exit()


# data_location = \
#     Path("/g/data/ge3/covariates/national_albers_filled_new/albers_cropped/")
# Read points from shapefile

# shapefile_location = Path("/g/data/ge3/aem_sections/AEM_covariates/")

# geotifs = {
#     "relief_radius4.tif": "relief4",
#     "national_Wii_RF_multirandomforest_prediction.tif": "mrf_pred",
#     "MvrtpLL_smooth.tif": "mrvtpll_s",
#     "MvrtpLL_fin.tif": "mvrtpll_f",
#     "mrvbf_9.tif": "mrvbf_9",
#     "Rad2016K_Th.tif": "rad2016kth",
#     "Thorium_2016.tif": "th_2016",
#     "Mesozoic_older_raster_MEAN.tif": "meso_mean",
#     "LOC_distance_to_coast.tif": "loc_dis",
#     "be-30y-85m-avg-ND-RED-BLUE.filled.lzw.nodata.tif": "be_av_rb",
#     "water-85m_1.tif": "water_85m",
#     "clim_RSM_albers.tif": "clim_rsm",
#     "tpi_300.tif": "tpi_300",
#     "be-30y-85m-avg-ND-SWIR1-NIR.filled.lzw.nodata.tif": "be_av_swir",
#     "si_geol1.tif": "si_geol1",
#     "be-30y-85m-avg-CLAY-PC2.filled.lzw.nodata.tif": "be_av_clay",
#     "be-30y-85m-avg-GREEN.filled.lzw.nodata.tif": "be_av_gr",
#     "be-30y-85m-avg_BLUE+SWIR2.tif": "be_av_bl",
#     "Gravity_land.tif": "gravity",
#     "dem_fill.tif": "dem",
#     "Clim_Prescott_LindaGregory.tif": "clim_linda",
#     "slope_fill2.tif": "slopefill2",
#     "clim_PTA_albers.tif": "clim_alber",
#     "SagaWET9cell_M.tif": "sagawet",
#     "ceno_euc_aust1.tif": "ceno_euc",
#     "s2-dpca-85m_band1.tif": "s2_band1",
#     "s2-dpca-85m_band2.tif": "s2_band2",
#     "s2-dpca-85m_band3.tif": "s2_band3",
#     "3dem_mag0_finn.tif": "3dem_mag0",
#     "3dem_mag1_fin.tif": "3dem_mag1",
#     "3dem_mag2.tif": "3dem_mag2",
# }

# geotifs = {
#     "relief_apsect.tif": "relief4",
#     "LATITUDE_GRID1.tif": "latitude",
#     "LONGITUDE_GRID1.tif": "longitude",
#     "er_depg.tif": "er_depg",
#     "sagawet_b_sir.tif": "sagawet",
#     "dem_foc2.tif": "dem_foc2",
#     "outcrop_dis2.tif": "outcrop",
#     "k_15v5.tif": "k_15v5",
# }


def intersect_and_sample_shp(shp: Path, geotifs: Dict[str, str], dedupe: bool = False, intersect_parallel: bool = False):
    print("====================================\n", f"intersecting {shp.as_posix()}")
    pts = gpd.read_file(shp)
    coords = np.array([(p.x, p.y) for p in pts.geometry])
    geom = pd.DataFrame(coords, columns=geom_cols, index=pts.index)
    pts = pts.merge(geom, left_index=True, right_index=True)
    if dedupe:
        tif_name = list(geotifs.keys())[0]
        tif = Path(tif_name)
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

        # keep one only
        # pts_deduped = pts.sort_values(
        #     by=geom_cols, ascending=[True, True]
        # ).groupby(by=['rows', 'cols'], as_index=False).first()[orig_cols]

        # keep mean of repeated observations in a pixel
        #
        pts_count = pts.groupby(by=['rows', 'cols'], as_index=False).agg(pixel_count=('rows', 'count'))
        pts_mean = pts.groupby(by=['rows', 'cols'], as_index=False).mean()
        # import IPython; IPython.embed(); import sys; sys.exit()
        pts_deduped = pts_mean.merge(pts_count, how='inner', on=['rows', 'cols'])

        pts_deduped = gpd.GeoDataFrame(pts_deduped, geometry=gpd.points_from_xy(pts_deduped['POINT_X'],
                                                                                pts_deduped['POINT_Y']))
        # pts_deduped = pts.drop_duplicates(subset=['rows', 'cols'])[orig_cols]
        coords_deduped = np.array([(p.x, p.y) for p in pts_deduped.geometry])
    else:
        pts_deduped = pts
        coords_deduped = coords

    # for k, v in geotifs.items():
        # print(f"adding {i}/{len(geotifs)}: {k} to output dataframe")
        # pts_deduped[v] = __extract_raster_points(coords_deduped, k)
    if intersect_parallel:
        pts_deduped__ = Parallel(
                n_jobs=-1,
                verbose=100,
            )(delayed(__extract_raster_points)(r, coords_deduped) for r in geotifs.keys())

        for i, col_name in enumerate(geotifs.values()):
            pts_deduped[col_name] = pts_deduped__[i]
    else:
        for i, (k, v) in enumerate(geotifs.items()):
            print(f"adding {i}/{len(geotifs)}: {k} to output dataframe")
            pts_deduped[v] = __extract_raster_points(coords_deduped, k)

    # pts_deduped = gpd.GeoDataFrame(pts_deduped, geometry=pts_deduped.geometry)
    # pts.to_csv(Path("out").joinpath(shp.stem + ".csv"), index=False)
    return pts_deduped


def __extract_raster_points(raster, coords_deduped):
    with rasterio.open(Path(raster)) as src:
        print(f"---- intersecting {raster}--------")
        return [x[0] for x in src.sample(coords_deduped)]


def intersect_sample_and_clean(shp, dedupe: bool = False, write_dropped: bool = False,
                               intersect_parallel: bool = False):
    df2 = intersect_and_sample_shp(shp, geotifs, dedupe=dedupe, intersect_parallel=intersect_parallel)
    output_dir.mkdir(exist_ok=True, parents=True)
    output_dir.joinpath('clean')
    out_shp = output_dir.joinpath(shp.name)
    df3 = df2[list(geotifs.values())]
    df4 = df2.loc[(df3.isna().sum(axis=1) == 0) & ((np.abs(df3) < 1e10).sum(axis=1) == len(geotifs)), :]
    df5 = df2.loc[~((df3.isna().sum(axis=1) == 0) & ((np.abs(df3) < 1e10).sum(axis=1) == len(geotifs))), :]
    try:
        if df5.shape[0]:
            df4.to_file(out_shp.parent.joinpath(out_shp.stem + '_cleaned.shp'))
            print(f"Wrote clean shapefile {out_shp.parent.joinpath(out_shp.stem + '_cleaned.shp')}")
            if write_dropped:
                df5.to_file(out_shp.parent.joinpath(out_shp.stem + '_cleaned_dropped.shp'))
        else:
            df2.to_file(out_shp.as_posix())
            print(f"saved intersected shapefile at {out_shp.as_posix()}")
            print(f"No points dropped and there for _cleaned.shp file is not createed'")
            print(f"No points dropped and there for _cleaned_dropped.shp file is not created")
    except:
        print(f"Check this shapefile {shp}")


if __name__ == '__main__':
    # local
    output_dir = Path('out_resampled')
    shapefile_location = Path("configs/data")
    downscale_factor = 2  # keep 1 point in a 2x2 cell

    geom_cols = ['POINT_X', 'POINT_Y']

    covariastes_list = "configs/data/sirsam/covariates_list.txt"
    geotifs_list = read_list_file(list_path=covariastes_list)
    shorts = defaultdict(int)

    geotifs = {}

    for gl in geotifs_list:
        k, v = generate_key_val(gl, shorts)
        geotifs[k] = v

    # check all required files are available on disc
    for k, v in geotifs.items():
        print(f"checking if {k} exists")
        assert Path(k).exists()

    print("Add under the 'intersected_features:' section in yaml")
    print('='*100)
    for k, v in geotifs.items():
        print(f"\"{Path(k).name}\": \"{v}\"")

    print('='*100)
    for s in shapefile_location.glob("*.shp"):
        intersect_sample_and_clean(s, True, True)
    # rets = Parallel(
    #         n_jobs=-1,
    #         verbose=100,
    #     )(delayed(intersect_sample_and_clean)(s, True) for s in shapefile_location.glob("geochem_sites.shp"))
