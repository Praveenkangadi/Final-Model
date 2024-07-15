import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Title of the Streamlit app
st.title("Stock Price Predictor App")

# Input section for stock ticker symbol
stock = st.text_input("Enter the Stock Ticker Symbol (e.g., GOOG, AAPL, RELIANCE.BO):", "GOOG")

# Dropdown menu to select the market
market = st.selectbox("Select Market:", ["US Market", "Indian Market"])

# Date pickers for selecting the date range
start_date = st.date_input("Start Date", datetime.now().date() - pd.DateOffset(years=20))
end_date = st.date_input("End Date", datetime.now().date())

# Convert date inputs to datetime objects
start = datetime.combine(start_date, datetime.min.time())
end = datetime.combine(end_date, datetime.min.time())

# Download stock data using yfinance
if stock:
    if market == "US Market":
        google_data = yf.download(stock, start, end)
    elif market == "Indian Market":
        stock = stock + ".BO"  # Add .BO for Bombay Stock Exchange
        google_data = yf.download(stock, start, end)

    # Check if data is fetched
    if not google_data.empty:
        # Load the pre-trained model
        model = load_model("Latest_stock_price_model.keras")

        # Display the stock data
        st.subheader(f"Stock Data for {stock}")
        st.write(google_data)

        # Split the data into training and testing sets
        splitting_len = int(len(google_data) * 0.7)
        x_test = pd.DataFrame(google_data.Close[splitting_len:])

        # Function to plot the graphs
        def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
            fig = plt.figure(figsize=figsize)
            plt.plot(values, 'Orange', label='Moving Average')
            plt.plot(full_data.Close, 'b', label='Close Price')
            if extra_data:
                plt.plot(extra_dataset, label='Extra Data')
            plt.legend()
            return fig

        # Plot moving averages
        st.subheader('Original Close Price and Moving Averages')
        ma_days = [100, 200, 250]
        for ma in ma_days:
            ma_col = f'MA_for_{ma}_days'
            google_data[ma_col] = google_data.Close.rolling(ma).mean()
            st.pyplot(plot_graph((15, 6), google_data[ma_col], google_data))
        
        st.subheader('Original Close Price with Multiple Moving Averages')
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(google_data.Close, label='Close Price')
        for ma in ma_days:
            ma_col = f'MA_for_{ma}_days'
            ax.plot(google_data[ma_col], label=f'MA for {ma} days')
        ax.legend()
        st.pyplot(fig)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test[['Close']])

        # Prepare the data for the model
        x_data = []
        y_data = []

        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        # Make predictions using the model
        predictions = model.predict(x_data)

        # Inverse transform the predictions
        inv_pre = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        # Prepare the plotting data
        plotting_data = pd.DataFrame({
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        }, index=google_data.index[splitting_len+100:])

        # Display the original vs predicted values
        st.subheader("Original values vs Predicted values")
        st.write(plotting_data)

        # Plot the original vs predicted close prices
        st.subheader('Original Close Price vs Predicted Close price')
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(pd.concat([google_data.Close[:splitting_len+100], plotting_data], axis=0))
        ax.legend(["Data not used", "Original Test data", "Predicted Test data"])
        st.pyplot(fig)

        # Generate buy/sell signals based on 100-day MA crossing 250-day MA
        google_data['MA_for_100_days'] = google_data.Close.rolling(window=100).mean()
        google_data['MA_for_250_days'] = google_data.Close.rolling(window=250).mean()
        google_data.dropna(subset=['MA_for_100_days', 'MA_for_250_days'], inplace=True)  # Drop rows with NaN values

        def generate_signals(data):
            buy_signals = [np.nan] * len(data)
            sell_signals = [np.nan] * len(data)
            for i in range(1, len(data)):
                if data['MA_for_100_days'].iloc[i] > data['MA_for_250_days'].iloc[i] and data['MA_for_100_days'].iloc[i-1] <= data['MA_for_250_days'].iloc[i-1]:
                    buy_signals[i] = data['Close'].iloc[i]
                elif data['MA_for_100_days'].iloc[i] < data['MA_for_250_days'].iloc[i] and data['MA_for_100_days'].iloc[i-1] >= data['MA_for_250_days'].iloc[i-1]:
                    sell_signals[i] = data['Close'].iloc[i]
            return buy_signals, sell_signals

        google_data['Buy Signal'], google_data['Sell Signal'] = generate_signals(google_data)

        # Plot buy/sell signals
        st.subheader('Close Price with Buy and Sell Signals Based on 100-day and 250-day Moving Averages')
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(google_data.Close, label='Close Price')
        ax.plot(google_data['MA_for_100_days'], label='100-day MA', linestyle='--')
        ax.plot(google_data['MA_for_250_days'], label='250-day MA', linestyle='--')
        ax.plot(google_data['Buy Signal'], marker='^', color='g', linestyle='None', label='Buy Signal')
        ax.plot(google_data['Sell Signal'], marker='v', color='r', linestyle='None', label='Sell Signal')
        ax.legend()
        st.pyplot(fig)

        # Real-time prediction for the next day
        latest_data = google_data.Close.values[-100:].reshape(-1, 1)
        latest_data_scaled = scaler.transform(latest_data)
        latest_data_scaled = latest_data_scaled.reshape(1, -1, 1)
        next_day_prediction = model.predict(latest_data_scaled)
        next_day_price = scaler.inverse_transform(next_day_prediction)
        
        # Adjust the next day price based on the latest signals
        latest_buy_signal = google_data['Buy Signal'].iloc[-1]
        latest_sell_signal = google_data['Sell Signal'].iloc[-1]
        
        if not np.isnan(latest_buy_signal):
            next_day_price = next_day_price * 1.01  # assuming 1% increase after a buy signal
        elif not np.isnan(latest_sell_signal):
            next_day_price = next_day_price * 0.99  # assuming 1% decrease after a sell signal

        # Convert the price prediction to Indian Rupees if the market is Indian
        if market == "Indian Market":
            usd_inr = yf.Ticker("USDINR=X")
            usd_inr_rate = usd_inr.history(period="1d")['Close'][0]
            next_day_price_inr = next_day_price[0][0] * usd_inr_rate
            st.subheader(f"Predicted Close Price for the next trading day: ₹{next_day_price_inr:.2f}")
        else:
            st.subheader(f"Predicted Close Price for the next trading day: ${next_day_price[0][0]:.2f}")

    else:
        st.error(f"Failed to fetch data for {stock}. Please check the ticker symbol and try again.")
else:
    st.info("Please enter a stock ticker symbol to fetch data.")
