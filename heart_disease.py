import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import pickle

# Load dataset
df = pd.read_csv("/Users/sumondebnath/Documents/heart_disease_app/heart_disease_uci.csv")
df.rename(columns={'num': 'target'}, inplace=True)
df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)

# One-hot encode
df = pd.get_dummies(df, drop_first=True)

# Split
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# Save model, scaler, and columns
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X_train.columns.to_list(), f)

print("✅ Model, scaler, and column list saved!")

# Save model, scaler, and column list
import joblib
import pickle

# Model save
joblib.dump(model, "logistic_model.pkl")

# Scaler save
joblib.dump(scaler, "scaler.pkl")

# Column names save
columns = X_train.columns.to_list()
with open("model_columns.pkl", "wb") as f:
    pickle.dump(columns, f)

print("✅ Model, scaler, and column list saved!")
import pickle

columns = X_train.columns.to_list()
with open("model_columns.pkl", "wb") as f:
    pickle.dump(columns, f)

print("✅ Column list saved!")