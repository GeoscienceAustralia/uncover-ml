import pandas as pd
from multiprocessing import Pool
# assume there are 32 cores on this machine

df = pd.read_csv('3D_AEM_Model_out_V3.csv')

def processChunk(frame_list):
    df_out = pd.DataFrame(columns=['X', 'Y', 'Z', 'class'])
    for frame in frame_list:
        z_prev = 0.0
        class_prev = 0.0
        for index, row in frame.iterrows():
            if index == 0:
                df_out = df_out.append({'X': row['X'], 'Y': row['Y'], 'Z': row['Z'], 'class': row['class']}, ignore_index=True)
                z_prev = row['Z']
                class_prev = row['class']
                continue
            if z_prev > 0 and row['Z'] >= z_prev:
                print 'The Z value for this row ' + str(row) + ' is greater than previous row\'s Z ' + str(z_prev)
                break
            if (z_prev - row['Z']) / 5 > 2:
                total_count = ((z_prev - row['Z']) // 5) - 1
                half_count = (((z_prev - row['Z']) // 5) - 1)//2
                count = 0
                for i in range(int(((z_prev - row['Z']) // 5) - 1)):
                    if row['class'] != class_prev and count < half_count:
                        df_out = df_out.append({'X': row['X'], 'Y': row['Y'], 'Z': z_prev - ((i+1)*5), 'class': class_prev}, ignore_index=True)
                    else:
                        df_out = df_out.append({'X': row['X'], 'Y': row['Y'], 'Z': z_prev - ((i+1)*5), 'class': row['class']}, ignore_index=True)
                    count = count + 1
            df_out = df_out.append({'X': row['X'], 'Y': row['Y'], 'Z': row['Z'], 'class': row['class']}, ignore_index=True)
            z_prev = row['Z']
            class_prev = row['class']
    return df_out
count = 0
mp_data_list = []
df_list = []
for xy, z_class in df.groupby(['X', 'Y']):
    if count >= 1320:
        mp_data_list.append(df_list)
        df_list = []
        count = 0
    df_list.append(z_class)
    count = count + 1
mp_data_list.append(df_list)

pool = Pool(processes=32)
results = pool.map(processChunk, tuple(mp_data_list))
result_df = pd.concat(results)
result_df.to_csv('3D_AEM_Model_out_expanded_V3.csv')
