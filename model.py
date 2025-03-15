import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("audi.csv")

df['model'] = df['model'].str.strip()
df['fuelType'] = df['fuelType'].str.strip() 

X = df.iloc[:, [0, 1, 3, 4, 5, 6, 7, 8]].values
Y = df.iloc[:, [2]].values

le1 = LabelEncoder()
X[:, 0] = le1.fit_transform(X[:, 0])

le2 = LabelEncoder()
X[:, -4] = le2.fit_transform(X[:, -4])

print(f"Before One-Hot Encoding: X shape = {X.shape}")

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), [2])], remainder='passthrough')
X = ct.fit_transform(X) 

print(f"After One-Hot Encoding: X shape = {X.shape}")

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train.ravel())


pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le1, open('le1.pkl', 'wb'))
pickle.dump(le2, open('le2.pkl', 'wb'))
pickle.dump(ct, open('ct.pkl', 'wb'))
pickle.dump(sc, open('sc.pkl', 'wb'))

print(" Model training complete! Saved as model.pkl")
