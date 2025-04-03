import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

model = load_model('Stock_Predictions_Model.keras')

st.header('📈 Stock Market Predictor')

stock = st.text_input('Enter Stock Symbol (e.g., AAPL, TSLA, GOOG)', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, start, end)

st.subheader('📊 Stock Data')
st.write(data)

data_train = data.Close[:int(len(data) * 0.80)]
data_test = data.Close[int(len(data) * 0.80):]

scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test.values.reshape(-1, 1))

st.subheader('📈 Price vs Moving Averages')

ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(data.index, data.Close, label='Stock Price', color='green')
ax.plot(data.index, ma_50, label='50-Day MA', color='red')
ax.plot(data.index, ma_100, label='100-Day MA', color='blue')
ax.plot(data.index, ma_200, label='200-Day MA', color='orange')
ax.legend()
st.pyplot(fig)

x_test, y_test = [], []
for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

predictions = model.predict(x_test)

scale_factor = 1 / scaler.scale_[0]
predictions = predictions * scale_factor
y_test = y_test * scale_factor

st.subheader('📉 Original Price vs Predicted Price')
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(y_test, label='Actual Price', color='blue')
ax2.plot(predictions, label='Predicted Price', color='red')
ax2.set_xlabel('Time')
ax2.set_ylabel('Stock Price')
ax2.legend()
st.pyplot(fig2)
