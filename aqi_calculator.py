import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ==============================
# 1. LOAD DATA
# ==============================

df = pd.read_csv("Bangalore_AQI_Dataset.csv")

# Select correct columns
df = df[['Date', 'PM2.5']]

# Rename for easier handling
df.columns = ['date', 'pm2_5']

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# ==============================
# 2. CLEAN DATA
# ==============================

# Remove invalid values
df['pm2_5'] = df['pm2_5'].mask(df['pm2_5'] < 0)
df['pm2_5'] = df['pm2_5'].mask(df['pm2_5'] > 1000)

# Drop missing
df = df.dropna()

# ==============================
# 3. AQI CALCULATION (PM2.5 - Indian Standard)
# ==============================

def calculate_aqi_pm25(pm):
    breakpoints = [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 1000, 401, 500)
    ]

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (pm - c_low) + i_low

    return None

df['AQI'] = df['pm2_5'].apply(calculate_aqi_pm25)

# ==============================
# 4. FEATURE ENGINEERING
# ==============================

# Lag features
df['lag1'] = df['pm2_5'].shift(1)
df['lag2'] = df['pm2_5'].shift(2)
df['lag3'] = df['pm2_5'].shift(3)

# Rolling mean
df['rolling_mean_3'] = df['pm2_5'].rolling(3).mean()

# Time features
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

df = df.dropna()

# ==============================
# 5. MODEL TRAINING
# ==============================

features = ['lag1', 'lag2', 'lag3', 'rolling_mean_3', 'day_of_week', 'month']
X = df[features]
y = df['pm2_5']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred_test))

# ==============================
# 6. FUTURE PREDICTION (NEXT 7 DAYS)
# ==============================

future_days = 7
last_data = df.iloc[-1:].copy()
future_predictions = []

for i in range(future_days):
    input_features = last_data[features]
    next_pm = model.predict(input_features)[0]
    
    future_predictions.append(next_pm)

    # Update lag values
    new_row = last_data.copy()
    new_row['lag3'] = new_row['lag2']
    new_row['lag2'] = new_row['lag1']
    new_row['lag1'] = next_pm
    new_row['rolling_mean_3'] = np.mean([new_row['lag1'], new_row['lag2'], new_row['lag3']])
    new_row['day_of_week'] = (new_row['day_of_week'] + 1) % 7
    
    last_data = new_row

# ==============================
# 7. CONVERT FUTURE PM TO AQI
# ==============================

future_aqi = [calculate_aqi_pm25(pm) for pm in future_predictions]

aqi_min = round(min(future_aqi), 2)
aqi_max = round(max(future_aqi), 2)

print("\nPredicted AQI Range for Next 7 Days:")
print(f"{aqi_min} – {aqi_max}")