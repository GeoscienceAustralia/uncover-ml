import pandas as pd
import dask.dataframe as dd

df = dd.read_csv('3D_AEM_model_V3_base.csv')
#df = pd.read_csv('3D_AEM_model_V3_sample.csv')
df_011 = pd.read_csv('fromYusen/test.011/lines/1090001.csv', sep=' ')
df_022 = pd.read_csv('fromYusen/test.022/lines/1090001.csv', sep=' ')
df_055 = pd.read_csv('fromYusen/test.055/lines/1090001.csv', sep=' ')
df_099 = pd.read_csv('fromYusen/test.099/lines/1090001.csv', sep=' ')

def tmpFunc(group):
    if len(group) > 31:
        return pd.DataFrame.from_records([])
    records_list = []
    inv_row_011 = df_011[((df_011['easting'] - group.iloc[0]['X']).abs() < 0.00001) & ((df_011['northing'] - group.iloc[0]['Y']).abs() < 0.00001)]
    inv_row_022 = df_022[((df_022['easting'] - group.iloc[0]['X']).abs() < 0.00001) & ((df_022['northing'] - group.iloc[0]['Y']).abs() < 0.00001)]
    inv_row_055 = df_055[((df_055['easting'] - group.iloc[0]['X']).abs() < 0.00001) & ((df_055['northing'] - group.iloc[0]['Y']).abs() < 0.00001)]
    inv_row_099 = df_099[((df_099['easting'] - group.iloc[0]['X']).abs() < 0.00001) & ((df_099['northing'] - group.iloc[0]['Y']).abs() < 0.00001)]
    if len(inv_row_011) == 0 or len(inv_row_022) == 0 or len(inv_row_055) == 0 or len(inv_row_099) == 0:
        return pd.DataFrame.from_records([]) 
    for index, row in group.iterrows():
        if index % 31 == 0:
            continue
        record_list = []
        record_list.append(inv_row_011.iloc[0]['line'])
        record_list.append(inv_row_011.iloc[0]['fiducial'])
        record_list.append(inv_row_055.iloc[0]['elevation'])
        record_list.append(row['X'])
        record_list.append(row['Y'])
        record_list.append(row['Z'])
        record_list.append(inv_row_055.iloc[0]['elevation'] - row['Z'])
        record_list.append(row['wii_clip_1'])
        record_list.append(row['ceno_clip_'])
        record_list.append(row['age'])
        record_list.append(inv_row_011.iloc[0]['cond011_'+str(index % 31)])
        record_list.append(inv_row_022.iloc[0]['cond022_'+str(index % 31)])
        record_list.append(inv_row_055.iloc[0]['cond055_'+str(index % 31)])
        record_list.append(inv_row_099.iloc[0]['cond099_'+str(index % 31)])
        record_list.append(row['class'])
        records_list.append(record_list)
    return pd.DataFrame.from_records(records_list)
        
meta = [ ('line#', 'float64'),
         ('fiducial', 'float64'),
         ('Elevation', 'float64'),
         ('X', 'float64'),
         ('Y', 'float64'),
         ('Z', 'float64'),
         ('depth', 'float64'),
         ('wii_clip_1', 'float64'),
         ('ceno_clip_', 'float64'),
         ('age', 'int64'),
         ('cond011', 'float64'),
         ('cond022', 'float64'),
         ('cond055', 'float64'),
         ('cond099', 'float64'),
         ('class', 'int64')]
out_df = df.groupby(['X', 'Y']).apply(tmpFunc, meta=meta).compute()
#out_df = df.groupby(['X', 'Y']).apply(tmpFunc)
out_df.to_csv('3D_AEM_model_V3_input.csv', index=False)
