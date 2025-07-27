import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import random

#Function class
def get_rarity_class(rarity):
    if rarity < 30:
        return "Widely"
    elif rarity < 70:
        return "Mid"
    else:
        return "Rare"
    
# Data
data = pd.read_csv("Consoles.csv")

df = data.iloc[:, 1:]
# X and y
X = df[['amount', 'age', 'user score']]
y = df['rarity']

# Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
num = 20

#Kmeans
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X_scaled)
df['cluster'] = clusters

# Model
model = LinearRegression()
model.fit(X_scaled, y)

# Predict
user_input = [[1000000, 30, 6.9]]
predictions = list()

for i in range(num):
    
    blurry_logic = random.randint(0,1)
    logic = random.randint(1,10)

    first = int(user_input[0][0]) + logic
    second = int(user_input[0][1]) + blurry_logic
    third = float(user_input[0][2]) + blurry_logic

    changed = [[first,second,third]]
    changed_scaled = scaler.transform(changed)

    prediction = model.predict(changed_scaled) #prediction = [model_predict] predict list
    predictions.append(prediction)
    average = sum(predictions) / len(predictions)

#Find class
rarity_class = get_rarity_class(prediction[0])

print(predictions)
print("Rarity predict", average)
print("Class: ",rarity_class)
