import numpy as np
import geopandas as gpd
# from .intersect_rasters import geotifs

geotifs = {
    "3dem_mag0_finn.tif": "3dem_mag1",
    "3dem_mag1_fin.tif": "3dem_mag2",
    "3dem_mag2.tif": "3dem_mag3",
    "Clim_Prescott_LindaGregory.tif": "Clim_Pre1",
    "Dose_2016.tif": "Dose_2011",
    "Potassium_2016.tif": "Potassiu1",
    "Rad2016K_Th.tif": "Rad2016K1",
    "Rad2016K_UTH.tif": "Rad2016K2",
    "Rad2016Th_K.tif": "Rad2016T1",
    "Rad2016Th_UK.tif": "Rad2016T2",
    "Rad2016U_THK.tif": "Rad2016U1",
    "Rad2016U_Th.tif": "Rad2016U2",
    "SagaWET9cell_M.tif": "SagaWET91",
    "Thorium_2016.tif": "Thorium_1",
    "Uranium_2016.tif": "Uranium_1",
    "be-30y-85m-avg-CLAY-BLUE+SWIR1.filled.lzw.nodata.tif": "be-30y-81",
    "be-30y-85m-avg-CLAY-PC2.filled.lzw.nodata.tif": "be-30y-82",
    "be-30y-85m-avg-FERRIC-PC2.filled.lzw.nodata.tif": "be-30y-83",
    "be-30y-85m-avg-FERRIC-PC4.filled.lzw.nodata.tif": "be-30y-84",
    "be-30y-85m-avg-GREEN.filled.lzw.nodata.tif": "be-30y-85",
    "be-30y-85m-avg-RED.filled.lzw.nodata.tif": "be-30y-86",
    "be-30y-85m-avg-SWIR1.filled.lzw.nodata.tif": "be-30y-87",
    "be-30y-85m-avg-SWIR2.filled.lzw.nodata.tif": "be-30y-88",
    "be-30y-85m-avg_BLUE+SWIR2.tif": "be-30y-89",
    "ceno_euc_aust1.tif": "ceno_euc1",
    "dem_fill.tif": "dem_fill1",
    "national_Wii_RF_multirandomforest_prediction.tif": "national1",
    "relief_elev_focalrange1000m_3s.tif": "relief_e1",
    "relief_elev_focalrange300m_3s.tif": "relief_e2",
    "si_geol1.tif": "si_geol11",
    "slope_fill2.tif": "slope_fi1",
    "relief_mrvbf_3s_mosaic.tif": "relief_m1",
    "relief_roughness.tif": "relief_r1",
    "MvrtpLL_fin.tif": "MvrtpLL_1",
    "PM_Aster_regolithRatios_b_1.tif": "PM_Aster1",
    "PM_Aster_regolithRatios_b_2.tif": "PM_Aster2",
    "PM_Aster_regolithRatios_b_3.tif": "PM_Aster3",
    "modis1_te.tif": "modis1_t1",
    "modis2_te_fin.tif": "modis2_t1",
    "modis3_te.tif": "modis3_t1",
    "modis4_te.tif": "modis4_t1",
    "modis5_te_fin.tif": "modis5_t1",
    "modis6_te.tif": "modis6_t1",
}


df2 = gpd.GeoDataFrame.from_file('/g/data/ge3/john/MAJORS/Nat_Fe_albers.shp')
df3 = df2[list(geotifs.values())]
# df4 = df3.loc[df3.isna().sum(axis=1) ==0, :]
df4 = df2.loc[(df3.isna().sum(axis=1) == 0) & ((np.abs(df3) < 1e10).sum(axis=1) == 43), :]
df5 = df2.loc[~((df3.isna().sum(axis=1) == 0) & ((np.abs(df3) < 1e10).sum(axis=1) == 43)), :]

df4.to_file('cleaned.shp')
df5.to_file('cleaned_dropped.shp')
