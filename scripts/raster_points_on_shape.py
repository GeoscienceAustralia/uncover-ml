from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd


def intersect_and_sample_shp(raster: Path, shp: Path):
    print("====================================\n", f"intersecting {shp.as_posix()}")
    pts = gpd.read_file(shp)
    for g in geom_cols:
        if g in pts.columns:
            pts = pts.drop(g)
    coords = np.array([(p.x, p.y) for p in pts.geometry])
    geom = pd.DataFrame(coords, columns=geom_cols, index=pts.index)
    pts = pts.merge(geom, left_index=True, right_index=True)
    pts[raster.name[:8]] = __extract_raster_points(raster, coords)
    return pts


def __extract_raster_points(raster: str, coords_deduped: np.ndarray):
    with rasterio.open(Path(raster)) as src:
        print(f"---- intersecting {raster}--------")
        return [x[0] for x in src.sample(coords_deduped)]


if __name__ == '__main__':
    # local
    output_dir = Path('intersected')
    shapefile_location = Path("configs/data")
    downscale_factor = 1  # keep 1 point in a 2x2 cell

    geom_cols = ['POINT_X', 'POINT_Y']
    # check all required files are available on disc
    print('='*100)
    shp = shapefile_location.joinpath("geochem_sites.shp")
    raster = Path("configs/data/sirsam/LATITUDE_GRID1.tif")
    gdf = intersect_and_sample_shp(raster, shp)
    output_dir.mkdir(exist_ok=True, parents=True)
    gdf.to_file(output_dir.joinpath(shp.stem + '_intersected.shp'))

