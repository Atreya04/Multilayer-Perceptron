import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")

df = pd.read_csv("Climate Dataset/DailyDelhiClimateTrain.csv")

if "date" in df.columns:
    df = df.drop("date", axis=1)
if "country" in df.columns:
    df = df.drop("country", axis=1)

corr_matrix = df.corr()
print(corr_matrix)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
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
    verbose = True
)

mlp.fit(x_train, y_train)

y_pred = mlp.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

print("\nEnter the following features to predict the mean temperature:")
input_features = {}

for feature in x.columns:
    input_features[feature] = float(input(f"{feature.replace('_', ' ').capitalize()}: "))

user_input = np.array([list(input_features.values())])
user_input_scaled = scaler.transform(user_input)

predicted_temp = mlp.predict(user_input_scaled)[0]
print(f"\nPredicted Mean Temperature: {predicted_temp:.2f}°C")
