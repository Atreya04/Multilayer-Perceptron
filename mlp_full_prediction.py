import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings


warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")


df = pd.read_csv("Multilayer Perceptron/Climate Dataset/DailyDelhiClimateTrain.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df = df.dropna()

lags = 5
for i in range(1, lags + 1):
    df[f"meantemp_lag_{i}"] = df["meantemp"].shift(i)
    df[f"humidity_lag_{i}"] = df["humidity"].shift(i)
    df[f"wind_speed_lag_{i}"] = df["wind_speed"].shift(i)
    df[f"meanpressure_lag_{i}"] = df["meanpressure"].shift(i)

df = df.dropna()


X = df[[col for col in df.columns if "lag" in col]]
y = df["meantemp"]
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPRegressor(
    hidden_layer_sizes=(64),
    activation='logistic',
    solver='sgd',
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42,
    learning_rate_init=0.05,
)

mlp.fit(X_train_scaled, y_train)


y_pred = mlp.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")


last_known = df.iloc[-lags:][["meantemp", "humidity", "wind_speed", "meanpressure"]]
future_predictions = []

for _ in range(7):
    lags_input = pd.DataFrame([last_known.values.flatten()], columns=X.columns)
    lags_input_scaled = scaler.transform(lags_input)
    next_temp = mlp.predict(lags_input_scaled)[0]
    future_predictions.append(next_temp)

    next_row = np.append(last_known.values[1:], [[next_temp, 0, 0, 0]], axis=0)
    last_known = pd.DataFrame(next_row, columns=["meantemp", "humidity", "wind_speed", "meanpressure"])


print("Future Temperature Predictions for Next 7 Days:")
for i, temp in enumerate(future_predictions, 1):
    print(f"Day {i}: {temp:.2f}°C")

