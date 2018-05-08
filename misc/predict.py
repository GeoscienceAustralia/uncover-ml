import pandas as pd
gdf = pd.read_csv('3D_AEM_model.csv')
gdf_ml = gdf[gdf['class'] != 0]
gdf_ml_filt = gdf_ml[(gdf_ml.ref_0001 >= 0) & (gdf_ml.ref_0001vs >= 0) & (gdf_ml.ref_0100 >= 0)]
gdf_ml_filt_shuffled = gdf_ml_filt.sample(frac=1)
gdf_ml_filt_shuffled.set_index(gdf_ml_filt.index, inplace=True)

X = gdf_ml_filt_shuffled[['ref_0001', 'ref_0001vs', 'ref_0100', 'wii_clip_1', 'ceno_clip_', 'Z', 'age']]
y = gdf_ml_filt_shuffled[['class']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(26,13,13),learning_rate='adaptive',max_iter=10000,solver='adam',tol=0.0,verbose=True)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

