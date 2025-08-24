import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Datas/Consoles.csv")
df = data.iloc[:, 1:]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['cluster'] = clusters  
print(df.head(30)["cluster"])  <-- data set update with kmeans 
"""

# Data
data = pd.read_csv("Datas/Consoles.csv")

df = data.iloc[:, 1:]
# X and y
X = df[['amount', 'age', 'user score']]
y = df[['rarity','cluster']]

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict
user_input = [[10000, 10, 7.0]]
user_scaled = scaler.transform(user_input)
prediction = model.predict(user_scaled)
prediction = str(prediction).split(" ")

#Lambda func
Class = lambda x: 1 if x > str(0.5) else 0

print("Rarity predict:", prediction,Class(prediction[2]))
