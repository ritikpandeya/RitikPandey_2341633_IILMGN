# data_train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import dump

# ---- Simple sample training data ----
data = {
    'room': [101, 102, 103, 104, 105, 201, 202, 203, 204, 205],
    'area': [300, 250, 280, 320, 350, 400, 380, 360, 420, 410],
    'appliances': [3, 2, 3, 4, 5, 6, 5, 4, 7, 6],
    'temperature': [24, 26, 25, 23, 27, 25, 26, 27, 24, 23],
    'time_of_day': [10, 14, 18, 21, 9, 12, 15, 18, 20, 22],
    # target: electricity usage (kWh)
    'electricity_usage': [8.5, 7.1, 7.8, 9.2, 10.3, 11.5, 10.7, 9.8, 12.2, 11.6]
}

df = pd.DataFrame(data)

# Features and target
X = df[['room', 'area', 'appliances', 'temperature', 'time_of_day']]
y = df['electricity_usage']

# Train a simple model
model = LinearRegression()
model.fit(X, y)

# Save the trained model
dump(model, 'model.pkl')

print("Training completed. Model saved as model.pkl")
