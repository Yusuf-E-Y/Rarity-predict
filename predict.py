import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Data
data = pd.read_csv("Consoles.csv")

df = data.iloc[:, 1:]
# X and y
X = df[['amount', 'age', 'user score']]
y = df['rarity']

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Predict
user_input = [[1000000, 20, 8.0]]
user_scaled = scaler.transform(user_input)
prediction = model.predict(user_scaled)

print("Rarity predict", prediction[0])
