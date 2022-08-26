from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd


def dedupe_raster(shp: Path, tif: Path, deduped_shp: Path):
    """
    :param shp: input shapefile with dense points
    :param tif: sample tif to read resolution details
    :param deduped_shp: output shapefile with one point per down-sampled raster resolution
    :return:
    """
    print("====================================\n", f"deduping {shp.as_posix()}")
    geom_cols = ['POINT_X', 'POINT_Y']
    pts = gpd.read_file(shp)
    for g in geom_cols:
        if g in pts.columns:
            pts = pts.drop(g, axis=1)
    coords = np.array([(p.x, p.y) for p in pts.geometry])
    geom = pd.DataFrame(coords, columns=geom_cols, index=pts.index)
    pts = pts.merge(geom, left_index=True, right_index=True)

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

    pts_count = pts.groupby(by=['rows', 'cols'], as_index=False).agg(pixel_count=('rows', 'count'))
    pts_mean = pts.groupby(by=['rows', 'cols'], as_index=False).mean()
    pts_deduped = pts_mean.merge(pts_count, how='inner', on=['rows', 'cols'])

    pts_deduped = gpd.GeoDataFrame(pts_deduped,
                                   geometry=gpd.points_from_xy(pts_deduped['POINT_X'], pts_deduped['POINT_Y']),
                                   crs="EPSG:3577"   # Australian Albers
                                   )
    pts_deduped.to_file(deduped_shp.as_posix())
    return pts_deduped


if __name__ == '__main__':
    shapefiles = Path("configs/data/")
    downscale_factor = 6  # keep 1 point in a 6x6 cell

    dem = Path('/home/my_dem.tif')
    output_dir = Path('1in6')
    output_dir.mkdir(exist_ok=True, parents=True)

    # for s in shapefiles.glob("*.shp"):
    #     deduped_shp = output_dir.joinpath(s.name)
    #     dedupe_raster(shp=s, tif=dem, deduped_shp=deduped_shp)

    Parallel(
            n_jobs=-1,
            verbose=100,
        )(delayed(dedupe_raster)(s, dem, output_dir.joinpath(s.name)) for s in shapefiles.glob("geochem_sites.shp"))
