import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from statsmodels.tsa.stattools import adfuller

#load dataset
df = pd.read_csv('TCS_stock_history.csv', parse_dates=['Date'], index_col='Date')
print(df.info())
df['5_day_MA'] = df['Close'].rolling(window=5).mean()
df['30_day_MA'] = df['Close'].rolling(window=30).mean()

#plot the closing price with moving average
plt.figure(figsize=(11,5))
plt.plot(df['Close'], label='TCS Close Price')
plt.plot(df['5_day_MA'], label='5 Day Moving Average', linestyle='--')
plt.plot(df['30_day_MA'], label= '30 Day Moving Average', linestyle='--')
plt.title('TCS Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

#heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of TCS Stock Data')
plt.show()

#Linear Regression 
X = df[['5_day_MA','30_day_MA']].dropna()
y = df['Close'].loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, shuffle= False)
lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error : {mse}')
print(f'R^2 Score: {r2}')

#lstm model for stock price prediction
data = df[['Close']].values
scaler = MinMaxScaler(feature_range= (0,1))
scaled_data = scaler.fit_transform(data)
def create_dataset(data, time_step= 60) : 
    X,y = [], []
    for i in range(len(data) - time_step - 1) :
        X.append(data[i : (i + time_step) , 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


time_step = 60
X, y = create_dataset(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[ :train_size] , X[train_size:]
y_train, y_test = y[ :train_size] , y[train_size:]

model = Sequential()
model.add(LSTM(units= 50, return_sequences= True, input_shape= (X_train.shape[1], 1)))
model.add(LSTM(units= 50, return_sequences= False))
model.add(Dense(units= 1))
model.compile(optimizer= 'adam', loss= 'mean_squared_error')
model.fit(X_train, y_train, epochs= 10, batch_size= 32)
lstm_predictions = model.predict(X_test)

lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

plt.figure(figsize= (12,6))
plt.plot(y_test_actual, label= 'Actual TCS Stock Price', color= 'blue')
plt.plot(lstm_predictions, label= 'Predicted TCS Stock Price (LSTM)', color= 'red')
plt.title('TCS Stock Price Prediction With LSTM')
plt.xlabel('Data')
plt.ylabel('Price(INR)')
plt.legend()
plt.show()


 





