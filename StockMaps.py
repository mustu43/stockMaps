import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def plot_graph(figsize, values, full_data, extra_data=None, extra_dataset=None, title="Stock Data"):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(values, 'orange', label='MA')
    ax.plot(full_data['Close'], 'b', label='Close Price')
    if extra_data and extra_dataset is not None:
        ax.plot(extra_dataset, label='Additional MA')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    return fig

# Initialize session state for recent stocks
if 'recent_stocks' not in st.session_state:
    st.session_state.recent_stocks = []
if 'search_triggered' not in st.session_state:
    st.session_state.search_triggered = False

# Main App Page
def main_page():
    st.set_page_config(page_title="Stock Maps", layout="wide")
    st.title("Stock Maps")

    # Sidebar for user input
    st.sidebar.header("User Input")

    # Text input for stock ID
    stock_input = st.sidebar.text_input("Enter the Stock ID", "GOOG", key="stock_input")

    # Button to trigger search
    search_button = st.sidebar.button("Search")

    # Date range selection
    default_start_date = datetime(2022, 3, 2)
    end = datetime.now()
    start = st.sidebar.date_input("Start Date", default_start_date)
    end = st.sidebar.date_input("End Date", end)

    if start > end:
        st.sidebar.error("Start date must be before end date.")
        st.stop()

    # Trigger search when the button is clicked or when Enter is pressed
    if search_button or st.session_state.get('search_triggered', False):
        selected_stock = stock_input

        # Fetch stock data
        try:
            google_data = yf.download(selected_stock, start, end)
            if google_data.empty:
                st.error("No data found for the stock ID. Please try a different one.")
                st.stop()
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            st.stop()

        # Load the model
        try:
            model = load_model("Latest_stock_price_model.keras")
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            st.stop()

        # Display stock data
        st.subheader("Stock Data")
        st.write(google_data)

        # Add the current stock to the recent stocks list
        if selected_stock not in st.session_state.recent_stocks:
            st.session_state.recent_stocks.insert(0, selected_stock)
            if len(st.session_state.recent_stocks) > 5:
                st.session_state.recent_stocks.pop()

        # Display recent stocks
        st.sidebar.subheader("Recently Viewed Stocks")
        for recent_stock in st.session_state.recent_stocks:
            if st.sidebar.button(f"Load {recent_stock}"):
                st.session_state.search_triggered = True
                st.session_state.search_triggered_stock = recent_stock

        # Sidebar for moving average selection
        ma_options = st.sidebar.multiselect(
            "Select Moving Averages",
            ["100 days", "200 days", "250 days"],
            default=["100 days", "200 days", "250 days"]
        )

        # Plot moving averages based on user selection
        for days in ["100 days", "200 days", "250 days"]:
            if days in ma_options:
                days_int = int(days.split()[0])
                ma_col = f'MA_for_{days_int}_days'
                google_data[ma_col] = google_data['Close'].rolling(days_int).mean()
                st.subheader(f'Original Close Price and MA for {days}')
                st.pyplot(plot_graph((15, 6), google_data[ma_col], google_data, title=f"MA for {days}"))

        # Plot combined moving averages
        if "100 days" in ma_options and "250 days" in ma_options:
            st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
            st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days'], title="MA for 100 and 250 days"))

        # Prepare data for predictions
        splitting_len = int(len(google_data) * 0.7)
        x_test = pd.DataFrame(google_data['Close'][splitting_len:])

        # Normalize data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test[['Close']])

        x_data = []
        y_data = []

        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])
            y_data.append(scaled_data[i])

        x_data, y_data = np.array(x_data), np.array(y_data)

        # Predict
        try:
            predictions = model.predict(x_data)
            inv_pre = scaler.inverse_transform(predictions)
            inv_y_test = scaler.inverse_transform(y_data)
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            st.stop()

        # Plot predictions
        plotting_data = pd.DataFrame({
            'original_test_data': inv_y_test.reshape(-1),
            'predictions': inv_pre.reshape(-1)
        }, index=google_data.index[splitting_len+100:])

        st.subheader("Original values vs Predicted values")
        st.write(plotting_data)

        # Plot original vs predicted close price
        fig = plt.figure(figsize=(15, 6))
        plt.plot(pd.concat([google_data['Close'][:splitting_len+100], plotting_data], axis=0))
        plt.legend(["Data - not used", "Original Test data", "Predicted Test data"])
        plt.title("Original vs Predicted Close Prices")
        plt.xlabel('Date')
        plt.ylabel('Price')
        st.pyplot(fig)

        # Reset search trigger
        st.session_state.search_triggered = False

# Main logic to display content
def main():
    main_page()

if __name__ == "__main__":
    main()
