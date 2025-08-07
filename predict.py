import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Data
data = pd.read_csv("Consoles.csv")

df = data.iloc[:, 1:]
# X and y
X = df[['amount', 'age', 'user score']]
y = df['rarity']

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans cluster
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# Model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict
user_input = [[10000, 10, 7.0]]
user_scaled = scaler.transform(user_input)
prediction = model.predict(user_scaled)

user_cluster = kmeans.predict(user_scaled)[0]

print("Rarity predict:", prediction[0])
print("User cluster:", user_cluster)
