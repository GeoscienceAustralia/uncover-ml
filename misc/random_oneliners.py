import geopandas as gpd
shpf = '3D_AEM_model.shp'
df = gpd.read_file(shpf)
 
df_ml = df[df['class'] != 0]
import scipy
from scipy.stats.stats import pearsonr
corr = pearsonr(df_ml['ref_0001'], df_ml['ref_0100vs'])
corr = pearsonr(np.cumsum(df_ml['thick'][0:31]), df_ml['Z'][0:31])
corr = pearsonr(df_ml['ref_0001'], df_ml['ref_0001vs'])


gdf[(gdf.ref_0001 == gdf_ml_filt_shuffled.iloc[0]['ref_0001']) & (gdf.ref_0100 == gdf_ml_filt_shuffled.iloc[0]['ref_0100']) & (gdf.ref_0001vs == gdf_ml_filt_shuffled.iloc[0]['ref_0001vs']) & (gdf.wii_clip_1 == gdf_ml_filt_shuffled.iloc[0]['wii_clip_1']) & (gdf.ceno_clip_ == gdf_ml_filt_shuffled.iloc[0]['ceno_clip_']) & (gdf.Z == gdf_ml_filt_shuffled.iloc[0]['Z'])]


gdf_ml_out = gdf[gdf['class'] == 0]

gdf_ml_out_filt = gdf_ml_out[(gdf_ml_out.ref_0001 >= 0) & (gdf_ml_out.ref_0001vs >= 0) & (gdf_ml_out.ref_0100 >= 0)

X_out = gdf_ml_out_filt[['ref_0001', 'ref_0001vs', 'ref_0100', 'wii_clip_1', 'ceno_clip_', 'Z', 'age']]

predictions = mlp.predict(X_out)

X_out_unscaled = scaler.inverse_transform(X_out)

predictions_m = predictions.reshape(predictions.shape[0], 1)

ndarray_out = np.concatenate([X_out_unscaled, predictions_m], axis=1)

df_out = pd.DataFrame(data=ndarray_out, columns=['ref_0001', 'ref_0001vs', 'ref_0100', 'wii_clip_1', 'ceno_clip_', 'Z', 'age', 'class'])

for i, row in df_out.iterrows():

    gdf.loc[(gdf['ref_0001'] == row['ref_0001']) & (gdf['ref_0100'] == row['ref_0100']) & (gdf['ref_0001vs'] == row['ref_0001vs']) & (gdf['wii_clip_1'] == row['wii_clip_1']) & (gdf['ceno_clip_'] == row['ceno_clip_']) & (gdf['Z'] == row['Z']), 'class'] = row['class']

outp = pd.concat([gdf_ml_filt['X'].reset_index(), gdf_ml_filt['Y'].reset_index(), gdf_ml_filt['Z'].reset_index(), df_out['class']], axis=1).reset_index()
outp[[col for col in outp.columns if col in ['X', 'Y', 'Z', 'class']]]
outp_final.to_csv('3D_AEM_Model_out_V3.csv', index=False)


