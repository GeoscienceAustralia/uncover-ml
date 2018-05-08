import pandas as pd
import dask.dataframe as dd

df = dd.read_csv('3D_AEM_Model_out_V3_1.csv')
inv_df = pd.read_csv('inversions.csv')

def tmpFunc(group):
    if len(group) > 30:
        return pd.DataFrame.from_records([])
    record_list = []
    inv_row = inv_df[((inv_df['easting'] - group.iloc[0]['X']).abs() < 0.00001) & ((inv_df['northing'] - group.iloc[0]['Y']).abs() < 0.00001)]
    record_list.append(109001)
    record_list.append(inv_row.iloc[0]['fiducial'])
    record_list.append(group.iloc[0]['X'])
    record_list.append(group.iloc[0]['Y'])
    record_list.append(inv_row.iloc[0]['elevation'])
    class_list = []
    depth_list = []
    for index, row in group.iterrows():
        class_list.append(row['class'])
        depth_list.append(inv_row.iloc[0]['elevation'] - row['Z'])
    if len(group) < 30:
        for i in range(30 - len(group)):
            class_list.append(0)
            depth_list.append(0)
    record_list.extend(class_list)
    record_list.extend(depth_list)
    if record_list[35] != 2.0:
        print record_list
    return pd.DataFrame.from_records([record_list])

meta = [ ('line#', 'int64'),
         ('fiducial', 'float64'),
         ('X', 'float64'),
         ('Y', 'float64'),
         ('Elevation', 'float64'),
         ('class_1', 'float64'),
         ('class_2', 'float64'),
         ('class_3', 'float64'),
         ('class_4', 'float64'),
         ('class_5', 'float64'),
         ('class_6', 'float64'),
         ('class_7', 'float64'),
         ('class_8', 'float64'),
         ('class_9', 'float64'),
         ('class_10', 'float64'),
         ('class_11', 'float64'),
         ('class_12', 'float64'),
         ('class_13', 'float64'),
         ('class_14', 'float64'),
         ('class_15', 'float64'),
         ('class_16', 'float64'),
         ('class_17', 'float64'),
         ('class_18', 'float64'),
         ('class_19', 'float64'),
         ('class_20', 'float64'),
         ('class_21', 'float64'),
         ('class_22', 'float64'),
         ('class_23', 'float64'),
         ('class_24', 'float64'),
         ('class_25', 'float64'),
         ('class_26', 'float64'),
         ('class_27', 'float64'),
         ('class_28', 'float64'),
         ('class_29', 'float64'),
         ('class_30', 'float64'),
         ('depth_1', 'float64'),
         ('depth_2', 'float64'),
         ('depth_3', 'float64'),
         ('depth_4', 'float64'),
         ('depth_5', 'float64'),
         ('depth_6', 'float64'),
         ('depth_7', 'float64'),
         ('depth_8', 'float64'),
         ('depth_9', 'float64'),
         ('depth_10', 'float64'),
         ('depth_11', 'float64'),
         ('depth_12', 'float64'),
         ('depth_13', 'float64'),
         ('depth_14', 'float64'),
         ('depth_15', 'float64'),
         ('depth_16', 'float64'),
         ('depth_17', 'float64'),
         ('depth_18', 'float64'),
         ('depth_19', 'float64'),
         ('depth_20', 'float64'),
         ('depth_21', 'float64'),
         ('depth_22', 'float64'),
         ('depth_23', 'float64'),
         ('depth_24', 'float64'),
         ('depth_25', 'float64'),
         ('depth_26', 'float64'),
         ('depth_27', 'float64'),
         ('depth_28', 'float64'),
         ('depth_29', 'float64'),
         ('depth_30', 'float64')]
out_df = df.groupby(['X', 'Y']).apply(tmpFunc, meta=meta).compute()
out_df.to_csv('output_to_yusen_2.csv', index=False)
#out_df.to_csv('output_sample.csv', index=False)
