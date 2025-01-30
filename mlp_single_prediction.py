import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

df = pd.read_csv("Climate Dataset/DailyDelhiClimate.csv")
df = df.drop("date", axis=1)

#Correlation Matrix 
corr_matrix = df.corr()
print(corr_matrix)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='Blues')
plt.title("Correlation Matrix")
plt.show()

x = df.drop("meantemp", axis=1) 
y = df["meantemp"]

scaler = StandardScaler()
x_cleaned = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_cleaned, y, test_size=0.2, random_state=42)

mlp = MLPRegressor(
    hidden_layer_sizes=(64),
    activation='logistic',
    solver='sgd',
    learning_rate='adaptive',
    max_iter=2000,
    random_state=42,
    learning_rate_init=0.05,
    verbose=True
)

mlp.fit(x_train, y_train)

y_pred = mlp.predict(x_test)

#Training performance
y_train_pred = mlp.predict(x_train)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2_score = r2_score(y_train, y_train_pred)
print(f"Training Mean Squared Error: {train_mse:.4f}")
print(f"Training R2 Score: {train_r2_score:.4f}")

#Testing performance
y_test_pred = mlp.predict(x_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2_score = r2_score(y_test, y_test_pred)
print(f"Testing Mean Squared Error: {test_mse:.4f}")
print(f"Testing R2 Score: {test_r2_score:.4f}")


#Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title("Actual vs Predicted Mean Temperature (in °C)")
plt.xlabel("Actual Mean Temperature (in °C)")
plt.ylabel("Predicted Mean Temperature (in °C)")
plt.grid(True)
plt.show()

# Line plot with sampled data points
sample_step = 10
y_test_sampled = y_test[::sample_step]
y_pred_sampled = y_pred[::sample_step]

plt.figure(figsize=(10, 6))
plt.plot(y_test_sampled.values, label='Actual Mean Temperature', color='b')
plt.plot(y_pred_sampled, label='Predicted Mean Temperature', color='r')
plt.xlabel("Actual Mean Temperature (in °C)")
plt.ylabel("Predicted Mean Temperature (in °C)")
plt.title("Actual vs. Predicted Mean Temperature (in °C)")
plt.legend()
plt.show()

print("\nEnter the following features to predict the mean temperature:")
input_features = {}

for feature in x.columns:
    input_features[feature] = float(input(f"{feature.replace('_', ' ').capitalize()}: "))

user_input = np.array([list(input_features.values())])
user_input_scaled = scaler.transform(user_input)

predicted_temp = mlp.predict(user_input_scaled)[0]
print(f"\nPredicted Mean Temperature: {predicted_temp:.2f}°C")
