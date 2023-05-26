import numpy as np
import rasterio

band_mapping = {
1:	"BCMAD_20m",
2:	"BI_10m",
3:	"BI2_10m",
4:	"CI_10m",
5:	"EMAD_20m",
6:	"EVI_10m",
7:	"FerricIron_10m",
8:	"FerricIron_20m",
9:	"FerricOxides_20m",
10:	"GNDVI_10m",
11:	"Gossan_20m",
12:	"GRVI_10m",
13:	"HydroxylBearing_20m",
14:	"IronOxides_10m",
15:	"LSWI_20m",
16:	"MSI_20m",
17:	"NDVI_10m",
18:	"PTP_BI_10m",
19:	"PTP_BI2_10m",
20:	"PTP_CI_10m",
21:	"PTP_EVI_10m",
22:	"PTP_GNDVI_10m",
23:	"PTP_GRVI_10m",
24:	"PTP_LSWI_20m",
25:	"PTP_MSI_20m",
26:	"PTP_NDVI_10m",
27:	"PTP_RI_10m",
28:	"PTP_SATVI_20m",
29:	"PTP_SAVI_10m",
30:	"PTP_TVI_10m",
31:	"PTP_V_10m",
32:	"PTP_WDVI_10m",
33:	"ReflectiveNonClays_20m",
34:	"RI_10m",
35:	"SATVI_20m",
36:	"SAVI_10m",
37:	"SMAD_10m",
38:	"SMAD_20m",
39:	"TVI_10m",
40:	"V_10m",
41:	"WDVI_10m",
}

reversed_band_mapping = {v:k for k, v in band_mapping.items()}

base_name = "dr_features_80m"
bands = ["PTP_SAVI_10m", "FerricIron_10m", "PTP_BI2_10m"]


band1 = rasterio.open(f"{base_name}.tif").read(reversed_band_mapping[bands[0]])
band2 = rasterio.open(f"{base_name}.tif").read(reversed_band_mapping[bands[1]])
band3 = rasterio.open(f"{base_name}.tif").read(reversed_band_mapping[bands[2]])

# normalized
band1 = (band1 - band1.min()) / (band1.max() - band1.min()) * 255
band2 = (band2 - band2.min()) / (band2.max() - band2.min()) * 255
band3 = (band3 - band3.min()) / (band3.max() - band3.min()) * 255


# Combine the bands into a single NumPy array.
image = np.dstack([band1, band2, band3])


albers_crs="EPSG:3577"   # Australian Albers
# write out
dst = rasterio.open(f'ternary_image_{bands[0]}_{bands[1]}_{bands[2]}.tif', 'w', driver='GTiff', width=image.shape[1], height=image.shape[0], count=3,
                    dtype=image.dtype, crs=albers_crs)
for i, b in enumerate([band1, band2, band3]):
    dst.write_band(b, i + 1)
