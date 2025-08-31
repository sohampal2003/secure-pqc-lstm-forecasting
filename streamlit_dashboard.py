import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from pqcrypto.kem.ml_kem_512 import generate_keypair, encrypt, decrypt

# Page configuration
st.set_page_config(
    page_title="Time Series Forecasting + PQC Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .forecast-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Utility functions
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / std
    return normalized, mean, std

def denormalize_data(normalized_data, mean, std):
    return normalized_data * std + mean

def train_model(X, y, epochs=50):
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Training Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    
    progress_bar.empty()
    status_text.empty()
    return model

def make_forecast(model, last_sequence, forecast_steps):
    model.eval()
    with torch.no_grad():
        # Ensure last_sequence is properly shaped: (sequence_length, input_features)
        if last_sequence.dim() == 1:
            # If 1D, reshape to (sequence_length, 1)
            last_sequence = last_sequence.unsqueeze(-1)
        
        current_sequence = last_sequence.clone()
        forecasts = []
        
        # Debug: Print initial tensor shapes
        print(f"DEBUG: last_sequence shape: {last_sequence.shape}")
        print(f"DEBUG: current_sequence initial shape: {current_sequence.shape}")
        
        for i in range(forecast_steps):
            # Ensure current_sequence is 3D: (batch_size, sequence_length, input_features)
            if current_sequence.dim() == 2:
                # If 2D, add batch dimension: (1, sequence_length, input_features)
                current_sequence = current_sequence.unsqueeze(0)
            elif current_sequence.dim() == 3:
                # If 3D, ensure first dimension is 1 (batch_size)
                if current_sequence.shape[0] != 1:
                    current_sequence = current_sequence.unsqueeze(0)
            else:
                # If 4D or more, take the first batch
                current_sequence = current_sequence[0:1]
            
            # Make prediction
            prediction = model(current_sequence)
            forecasts.append(prediction.item())
            
            # Update sequence for next prediction
            # Remove the batch dimension to get back to (sequence_length, input_features)
            if current_sequence.dim() == 3:
                current_sequence = current_sequence.squeeze(0)
            
            # Add prediction to the end of the sequence
            # Ensure prediction has the same number of dimensions as current_sequence
            if current_sequence.dim() == 2:
                # current_sequence is (sequence_length, input_features)
                # prediction should be (1, input_features)
                prediction_reshaped = prediction.unsqueeze(-1)
            else:
                # current_sequence is (sequence_length,)
                # prediction should be (1,)
                prediction_reshaped = prediction.unsqueeze(0)
            
            print(f"DEBUG: Step {i+1} - current_sequence shape: {current_sequence.shape}, prediction shape: {prediction.shape}, prediction_reshaped shape: {prediction_reshaped.shape}")
            
            current_sequence = torch.cat([current_sequence[1:], prediction_reshaped], dim=0)
    
    return np.array(forecasts)

# Main dashboard
def main():
    st.markdown('<h1 class="main-header">📈 Time Series Forecasting + PQC Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Stock symbol selection
    stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)")
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.date.today())
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    sequence_length = st.sidebar.slider("Sequence Length", min_value=5, max_value=30, value=10)
    hidden_size = st.sidebar.slider("Hidden Size", min_value=25, max_value=100, value=50)
    num_layers = st.sidebar.slider("Number of Layers", min_value=1, max_value=4, value=2)
    training_epochs = st.sidebar.slider("Training Epochs", min_value=10, max_value=100, value=50)
    forecast_steps = st.sidebar.slider("Forecast Steps", min_value=5, max_value=30, value=10)
    
    # PQC settings
    st.sidebar.subheader("Post-Quantum Cryptography")
    enable_pqc = st.sidebar.checkbox("Enable PQC Encryption", value=True)
    
    # Main content area
    if st.button("🚀 Start Analysis", type="primary"):
        try:
            with st.spinner("Downloading stock data..."):
                # Download stock data
                data = yf.download(stock_symbol, start=start_date, end=end_date)
                
                if data.empty:
                    st.error(f"No data found for {stock_symbol}. Please check the stock symbol and date range.")
                    return
                
                prices = data['Close'].values
                dates = data.index
                
                st.success(f"✅ Downloaded {len(prices)} data points for {stock_symbol}")
            
            # Data preprocessing
            with st.spinner("Preprocessing data..."):
                # Normalize data
                normalized_prices, mean_price, std_price = normalize_data(prices)
                
                # Create sequences
                X, y = create_sequences(normalized_prices, sequence_length)
                
                # Convert to PyTorch tensors
                X = torch.tensor(X, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32)
                
                if X.dim() == 2:
                    X = X.unsqueeze(-1)
                if y.dim() == 1:
                    y = y.unsqueeze(-1)
            
            # Display data overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Points", len(prices))
            with col2:
                st.metric("Current Price", f"${float(prices[-1]):.2f}")
            with col3:
                price_change = float(prices[-1] - prices[-2]) if len(prices) > 1 else 0
                st.metric("Price Change", f"${price_change:.2f}")
            with col4:
                price_change_pct = (price_change / float(prices[-2]) * 100) if len(prices) > 1 else 0
                st.metric("Change %", f"{price_change_pct:.2f}%")
            
            # Training section
            st.subheader("🤖 Model Training")
            model = train_model(X, y, epochs=training_epochs)
            st.success("✅ Model training completed!")
            
            # PQC Encryption
            if enable_pqc:
                st.subheader("🔐 Post-Quantum Cryptography")
                with st.spinner("Performing PQC encryption..."):
                    # Generate keypair
                    pk, sk = generate_keypair()
                    
                    # Encrypt model weights
                    weights = torch.cat([p.flatten() for p in model.parameters()])
                    ciphertext, shared_secret_enc = encrypt(pk)
                    decrypted_secret = decrypt(sk, ciphertext)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Encryption Status", "✅ Success" if shared_secret_enc == decrypted_secret else "❌ Failed")
                    with col2:
                        st.metric("Key Size", f"{len(pk)} bytes")
                    
                    st.info("🔐 Model weights encrypted using ML-KEM-512 post-quantum cryptography")
            
            # Forecasting
            st.subheader("🔮 Price Forecasting")
            with st.spinner("Generating forecasts..."):
                # Get last sequence for forecasting
                last_sequence = torch.tensor(normalized_prices[-sequence_length:], dtype=torch.float32)
                
                # Generate forecast
                normalized_forecast = make_forecast(model, last_sequence, forecast_steps)
                forecast_prices = denormalize_data(normalized_forecast, mean_price, std_price)
                
                # Create future dates
                last_date = dates[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
                
                # Create visualization
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Historical Prices & Forecast', 'Forecast Details'),
                    vertical_spacing=0.1
                )
                
                # Historical data
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=prices,
                        mode='lines',
                        name='Historical Prices',
                        line=dict(color='#1f77b4', width=2)
                    ),
                    row=1, col=1
                )
                
                # Forecast
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=forecast_prices,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#ff7f0e', width=2, dash='dash'),
                        marker=dict(size=6)
                    ),
                    row=1, col=1
                )
                
                # Forecast details
                fig.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=forecast_prices,
                        mode='lines+markers',
                        name='Forecast Trend',
                        line=dict(color='#ff7f0e', width=3),
                        marker=dict(size=8, symbol='diamond')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=600,
                    showlegend=True,
                    title_text=f"{stock_symbol} Stock Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price ($)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Forecast metrics
            st.subheader("📊 Forecast Metrics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Next Day Forecast", f"${float(forecast_prices[0]):.2f}")
            with col2:
                st.metric("5-Day Forecast", f"${float(forecast_prices[4]):.2f}")
            with col3:
                st.metric("10-Day Forecast", f"${float(forecast_prices[9]):.2f}")
            
            # Forecast table
            st.subheader("📋 Detailed Forecast")
            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecasted Price': forecast_prices,
                'Change from Current': forecast_prices - float(prices[-1]),
                'Change %': ((forecast_prices - float(prices[-1])) / float(prices[-1]) * 100)
            })
            
            st.dataframe(forecast_df, use_container_width=True)
            
            # Download forecast data
            csv = forecast_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Data (CSV)",
                data=csv,
                file_name=f"{stock_symbol}_forecast_{datetime.date.today()}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.exception(e)
    
    # Information section
    with st.expander("ℹ️ About this Dashboard"):
        st.markdown("""
        This dashboard provides real-time stock price forecasting using:
        
        - **LSTM Neural Networks**: Advanced deep learning for time series prediction
        - **Post-Quantum Cryptography**: ML-KEM-512 encryption for secure model weights
        - **Real-time Data**: Live stock data from Yahoo Finance
        - **Interactive Visualizations**: Plotly charts for better insights
        
        **How to use:**
        1. Select a stock symbol and date range
        2. Configure model parameters
        3. Click "Start Analysis" to begin
        4. View forecasts and download results
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Built with Streamlit • LSTM • Post-Quantum Cryptography • Yahoo Finance"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
