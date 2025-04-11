import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#Part a)
df = pd.read_csv("sp500.csv", parse_dates=["Date"], index_col="Date")
#Time series
ts = df["Adj Close"]

#Plot
plt.figure(figsize=(12,6))
plt.plot(ts, label="S&P 500 Adjusted Price", color="blue")
plt.xlabel("Year")
plt.ylabel("Adjusted Price")
plt.title("S&P 500 Adjusted Price Over Time")
plt.legend()
plt.grid()
plt.show()

#Part b)

#log returns
df["Log Return"] = np.log(df["Adj Close"]) - np.log(df["Adj Close"].shift(1))
df = df.dropna()

# Plot
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Log Return"], label="Monthly Log Return", color="red")
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)  # Reference line at 0
plt.xlabel("Year")
plt.ylabel("Log Return")
plt.title("S&P 500 Monthly Log Returns")
plt.legend()
plt.grid()
plt.show()

#Part c) From the plots below, I believe the ARMA (1,1) model would be a solid option as the ACF points to MA(1) and the PACT points to favoring AR(1). I chose for it to be a (1,1) model as the first lag is significant.

#Plot
fig, axes = plt.subplots(1, 2, figsize=(12,5))

plot_acf(df["Log Return"], ax=axes[0], lags=20)  # ACF plot
axes[0].set_title("Autocorrelation Function (ACF)")

plot_pacf(df["Log Return"], ax=axes[1], lags=20, method='ywm')  # PACF plot
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.show()

#part d) From the plot below, it seems log returns and white noise model are both a horizontal line. The AR model seems to show a stock-like pattern which I believe means it is outperforming the white noise model.

df = df.asfreq('B')

#training data
train_data = df.loc['2007-01-01':'2010-12-31', 'Log Return']
test_data = df.loc['2011-01-01':'2015-11-30', 'Log Return']
if train_data.isnull().any():
    print("Training data contains missing values. Dropping missing values...")
    train_data = train_data.dropna()

#Fitting and initializing
ar_model = ARIMA(train_data, order=(1, 0, 0))
ar_model_fit = ar_model.fit()
white_noise_model = sm.OLS(train_data, sm.add_constant(np.ones(len(train_data)))).fit()
ar_forecasts = []
wn_forecasts = []
window_size = 252

for i in range(len(test_data)):
    #Checking for out of bounds
    if i < window_size:
        
        train_data_ar = df.loc['2007-01-01':test_data.index[i], 'Log Return']
    else:
        
        train_data_ar = test_data.iloc[i-window_size:i]

    #Fit AR(1)
    ar_model = ARIMA(train_data_ar, order=(1, 0, 0))
    ar_model_fit = ar_model.fit()
    ar_forecast = ar_model_fit.forecast(steps=1)[0]  # Use the first forecast value from the result
    ar_forecasts.append(ar_forecast)

    #White noise forecast 
    wn_forecast = white_noise_model.predict(exog=np.ones(1))[0]  # Use a constant value (1) for prediction
    wn_forecasts.append(wn_forecast)

#Convert forecasts to pandas series
ar_forecasts_series = pd.Series(ar_forecasts, index=test_data.index)
wn_forecasts_series = pd.Series(wn_forecasts, index=test_data.index)

#Plot
plt.figure(figsize=(12,6))
plt.plot(test_data, label="Actual Log Returns", color="black", alpha=0.7)
plt.plot(ar_forecasts_series, label="AR(1) Forecast", color="blue", linestyle='--')
plt.plot(wn_forecasts_series, label="White Noise Forecast", color="red", linestyle='--')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.title("AR(1) vs White Noise Forecasting")
plt.legend()
plt.show()

#Part e) - The AR(1) Model is superior as it has a lower RMSE than the White Noise Model. Because the AR(1) Model is superior, it is fair to say that the market is not fully efficient as it should follow a random walk if it was.
#AR(1) Model RMSE: 0.031353464260646315
#White Noise Model RMSE: 0.03530113081496851

#RMSE Calculations
ar_rmse = np.sqrt(((test_data - ar_forecasts_series) ** 2).mean())
wn_rmse = np.sqrt(((test_data - wn_forecasts_series) ** 2).mean())
#RMSE Prints
print(f"AR(1) Model RMSE: {ar_rmse}")
print(f"White Noise Model RMSE: {wn_rmse}")

#Part f) - The White Noise now has a lower RMSE which means the market could be considered efficient which contradicts Part e). Comparing this forecast to the other one, it seems the AR(1) Model had more variance as the sharps seemed to last longer and had more magnitude
#AR(1) Rolling Forecast RMSE: 0.03931474359637712
#White Noise Rolling Forecast RMSE: 0.03263652117889297

df = df.asfreq('B')

#Training data
train_data = df.loc['2007-01-01':'2011-12-31', 'Log Return']
test_data = df.loc['2012-01-01':'2015-11-30', 'Log Return']
if train_data.isnull().any():
    print("Training data contains missing values. Dropping missing values...")
    train_data = train_data.dropna()

#Initialize Model
ar_model = ARIMA(train_data, order=(1, 0, 0))
ar_model_fit = ar_model.fit()
white_noise_model = sm.OLS(train_data, sm.add_constant(np.ones(len(train_data)))).fit()
ar_forecasts = []
wn_forecasts = []
rolling_window_size = 60

#Rolling forecast
for i in range(len(test_data)):
    if i < rolling_window_size:
        train_data_ar = df.loc['2007-01-01':test_data.index[i], 'Log Return']
    else:
        train_data_ar = test_data.iloc[i-rolling_window_size:i]  # Rolling window for AR(1)

    #Fit AR(1) Model
    ar_model = ARIMA(train_data_ar, order=(1, 0, 0))
    ar_model_fit = ar_model.fit()
    ar_forecast = ar_model_fit.forecast(steps=1)[0]  # Forecast next value
    ar_forecasts.append(ar_forecast)

    #White Noise Forecast
    wn_forecast = white_noise_model.predict(exog=np.ones(1))[0]  # Constant (white noise) prediction
    wn_forecasts.append(wn_forecast)

#Convert forecasts to series
ar_forecasts_series = pd.Series(ar_forecasts, index=test_data.index)
wn_forecasts_series = pd.Series(wn_forecasts, index=test_data.index)

#Plot
plt.figure(figsize=(12, 6))
plt.plot(test_data, label="Actual Log Returns", color="black", alpha=0.7)
plt.plot(ar_forecasts_series, label="AR(1) Forecast", color="blue", linestyle='--')
plt.plot(wn_forecasts_series, label="White Noise Forecast", color="red", linestyle='--')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.title("Rolling Forecast: AR(1) vs White Noise Models")
plt.legend()
plt.grid(True)
plt.show()

#RMSE Calc 
ar_rmse = np.sqrt(np.mean((ar_forecasts_series - test_data) ** 2))
wn_rmse = np.sqrt(np.mean((wn_forecasts_series - test_data) ** 2))

print(f"AR(1) Rolling Forecast RMSE: {ar_rmse}")
print(f"White Noise Rolling Forecast RMSE: {wn_rmse}")

#Part g) - both the AR(1) and White Noise model are both flat lines and did not do a great job compared to the true stock return values. 

df = df.asfreq('B')

#Train and Test data
train_data = df.loc['2007-01-01':'2014-12-31', 'Log Return']
test_data = df.loc['2015-01-01':'2015-11-30', 'Log Return']

if train_data.isnull().any():
    print("Warning: There are NaN values in the training data. These will be dropped.")
    train_data = train_data.dropna()

if test_data.isnull().any():
    print("Warning: There are NaN values in the test data. These will be dropped.")
    test_data = test_data.dropna()

#White Noise and AR(1)
last_train_value = train_data.iloc[-1]
white_noise_forecast = np.full(11, last_train_value)  # Forecast repeated 11 times
ar_model = ARIMA(train_data, order=(1, 0, 0))
ar_model_fit = ar_model.fit()

#AR(1) 11 steps ahead
ar_forecasts = ar_model_fit.forecast(steps=11)
if np.isnan(ar_forecasts).any():
    print("Warning: AR(1) forecast contains NaN values.")
else:
    print("AR(1) forecast generated successfully.")

#Plot
plt.figure(figsize=(12, 6))
plt.plot(test_data, label="Actual Log Returns", color="black", alpha=0.7)
plt.plot(test_data.index[:11], white_noise_forecast, label="White Noise Forecast", color="green", linestyle='--')
plt.plot(test_data.index[:11], ar_forecasts, label="AR(1) Forecast", color="blue", linestyle='--')
plt.axhline(y=0, color='black', linestyle='--', alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Log Return")
plt.title("White Noise vs AR(1) Forecasts vs Actual Log Returns (11 Steps Ahead)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

