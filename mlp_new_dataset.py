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

# Load the dataset
df = pd.read_csv("Multilayer-Perceptron/Climate Dataset/DailyDelhiClimateTrain.csv")

# Drop unnecessary columns
if "date" in df.columns:
    df = df.drop("date", axis=1)
if "country" in df.columns:
    df = df.drop("country", axis=1)

# Display correlation matrix
corr_matrix = df.corr()
print(corr_matrix)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Features (X) and target variable (y)
x = df.drop("meantemp", axis=1)  # Drop target variable
y = df["meantemp"]

# Scale features
scaler = StandardScaler()
x_cleaned = scaler.fit_transform(x)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x_cleaned, y, test_size=0.2, random_state=42)

# Initialize the MLP Regressor
mlp = MLPRegressor(
    hidden_layer_sizes=(64),
    activation='logistic',
    solver='sgd',
    learning_rate='adaptive',
    max_iter=2000,
    random_state=42,
    learning_rate_init=0.05,
)

# Train the model
mlp.fit(x_train, y_train)

# Evaluate the model
y_pred = mlp.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Prompt user input for prediction
print("\nEnter the following features to predict the mean temperature:")
input_features = {}

for feature in x.columns:
    input_features[feature] = float(input(f"{feature.replace('_', ' ').capitalize()}: "))

# Convert user input to numpy array and scale
user_input = np.array([list(input_features.values())])
user_input_scaled = scaler.transform(user_input)

# Predict mean temperature
predicted_temp = mlp.predict(user_input_scaled)[0]
print(f"\nPredicted Mean Temperature: {predicted_temp:.2f}°C")
