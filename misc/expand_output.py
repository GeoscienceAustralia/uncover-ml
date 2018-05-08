import pandas as pd

df = pd.read_csv('3D_AEM_Model_out_V3.csv')
df_out = pd.DataFrame(columns=['X', 'Y', 'Z', 'class'])
for xy, z_class in df.groupby(['X', 'Y']):
    z_prev = 0.0
    class_prev = 0.0
    for index, row in z_class.iterrows():
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
df_out.to_csv('3D_AEM_Model_out_expanded_V3.csv')
