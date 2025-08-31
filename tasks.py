"""
Celery tasks for background processing of time series forecasting
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import celery
from celery import Celery
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yfinance as yf
from sqlalchemy import create_engine, text
from pqcrypto.kem.ml_kem_512 import generate_keypair, encrypt, decrypt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery app configuration
app = Celery('forecasting_tasks')
app.config_from_object('celeryconfig')

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://forecasting_user:forecasting_password@localhost:5432/forecasting_db')
engine = create_engine(DATABASE_URL)

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

def create_sequences(data: np.ndarray, seq_length: int) -> tuple:
    """Create sequences for LSTM training"""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

def normalize_data(data: np.ndarray) -> tuple:
    """Normalize data using z-score normalization"""
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / std
    return normalized, mean, std

def denormalize_data(normalized_data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Denormalize data"""
    return normalized_data * std + mean

@app.task(bind=True, name='forecast_stock')
def forecast_stock(self, stock_symbol: str, forecast_days: int = 10, 
                   sequence_length: int = 10, hidden_size: int = 50, 
                   num_layers: int = 2, training_epochs: int = 50) -> Dict:
    """
    Background task to forecast stock prices
    
    Args:
        stock_symbol: Stock symbol to forecast
        forecast_days: Number of days to forecast
        sequence_length: LSTM sequence length
        hidden_size: LSTM hidden layer size
        num_layers: Number of LSTM layers
        training_epochs: Number of training epochs
    
    Returns:
        Dictionary containing forecast results and metadata
    """
    try:
        # Update task state
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Starting forecast...'}
        )
        
        logger.info(f"Starting forecast for {stock_symbol}")
        
        # Download stock data
        self.update_state(
            state='PROGRESS',
            meta={'current': 10, 'total': 100, 'status': 'Downloading stock data...'}
        )
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year of data
        
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {stock_symbol}")
        
        prices = data['Close'].values
        dates = data.index
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 20, 'total': 100, 'status': 'Preprocessing data...'}
        )
        
        # Data preprocessing
        normalized_prices, mean_price, std_price = normalize_data(prices)
        X, y = create_sequences(normalized_prices, sequence_length)
        
        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        if X.dim() == 2:
            X = X.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 30, 'total': 100, 'status': 'Training LSTM model...'}
        )
        
        # Train model
        model = LSTMModel(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(training_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            # Update progress
            progress = 30 + (epoch / training_epochs) * 40
            self.update_state(
                state='PROGRESS',
                meta={'current': int(progress), 'total': 100, 'status': f'Training epoch {epoch+1}/{training_epochs}'}
            )
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 70, 'total': 100, 'status': 'Generating forecasts...'}
        )
        
        # Generate forecasts
        model.eval()
        with torch.no_grad():
            last_sequence = torch.tensor(normalized_prices[-sequence_length:], dtype=torch.float32)
            forecasts = []
            
            for i in range(forecast_days):
                if last_sequence.dim() == 2:
                    last_sequence = last_sequence.unsqueeze(-1)
                
                prediction = model(last_sequence.unsqueeze(0))
                forecasts.append(prediction.item())
                
                # Update sequence for next prediction
                last_sequence = torch.cat([last_sequence[1:], prediction.unsqueeze(0)], dim=0)
                
                # Update progress
                progress = 70 + (i / forecast_days) * 20
                self.update_state(
                    state='PROGRESS',
                    meta={'current': int(progress), 'total': 100, 'status': f'Generating forecast {i+1}/{forecast_days}'}
                )
        
        # Denormalize forecasts
        forecast_prices = denormalize_data(np.array(forecasts), mean_price, std_price)
        
        # Create future dates
        last_date = dates[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        
        self.update_state(
            state='PROGRESS',
            meta={'current': 90, 'total': 100, 'status': 'Saving results...'}
        )
        
        # Save results to database
        save_forecast_results(stock_symbol, future_dates, forecast_prices, prices[-1])
        
        # Prepare result
        result = {
            'stock_symbol': stock_symbol,
            'forecast_dates': [d.strftime('%Y-%m-%d') for d in future_dates],
            'forecast_prices': forecast_prices.tolist(),
            'current_price': float(prices[-1]),
            'model_parameters': {
                'sequence_length': sequence_length,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'training_epochs': training_epochs
            },
            'metadata': {
                'data_points': len(prices),
                'training_date': datetime.now().isoformat(),
                'forecast_generated': datetime.now().isoformat()
            }
        }
        
        self.update_state(
            state='SUCCESS',
            meta={'current': 100, 'total': 100, 'status': 'Forecast completed successfully'}
        )
        
        logger.info(f"Forecast completed for {stock_symbol}")
        return result
        
    except Exception as e:
        logger.error(f"Error in forecast_stock task: {str(e)}")
        self.update_state(
            state='FAILURE',
            meta={'current': 0, 'total': 100, 'status': f'Error: {str(e)}'}
        )
        raise

@app.task(name='batch_forecast')
def batch_forecast(stock_symbols: List[str], forecast_days: int = 10) -> Dict:
    """
    Batch forecast multiple stocks
    
    Args:
        stock_symbols: List of stock symbols to forecast
        forecast_days: Number of days to forecast
    
    Returns:
        Dictionary containing results for all stocks
    """
    results = {}
    
    for symbol in stock_symbols:
        try:
            # Submit individual forecast task
            task = forecast_stock.delay(symbol, forecast_days)
            results[symbol] = {
                'task_id': task.id,
                'status': 'submitted'
            }
        except Exception as e:
            results[symbol] = {
                'error': str(e),
                'status': 'failed'
            }
    
    return results

@app.task(name='update_historical_data')
def update_historical_data(stock_symbol: str) -> Dict:
    """
    Update historical data for a stock
    
    Args:
        stock_symbol: Stock symbol to update
    
    Returns:
        Dictionary containing update results
    """
    try:
        # Download latest data
        data = yf.download(stock_symbol, start=datetime.now() - timedelta(days=30), end=datetime.now())
        
        if not data.empty:
            # Save to database
            save_historical_data(stock_symbol, data)
            
            return {
                'stock_symbol': stock_symbol,
                'status': 'success',
                'data_points': len(data),
                'last_updated': datetime.now().isoformat()
            }
        else:
            return {
                'stock_symbol': stock_symbol,
                'status': 'no_data',
                'message': 'No new data available'
            }
            
    except Exception as e:
        logger.error(f"Error updating historical data for {stock_symbol}: {str(e)}")
        return {
            'stock_symbol': stock_symbol,
            'status': 'error',
            'error': str(e)
        }

@app.task(name='cleanup_old_forecasts')
def cleanup_old_forecasts(days_to_keep: int = 30) -> Dict:
    """
    Clean up old forecast data
    
    Args:
        days_to_keep: Number of days of forecasts to keep
    
    Returns:
        Dictionary containing cleanup results
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with engine.connect() as conn:
            # Delete old forecasts
            result = conn.execute(text("""
                DELETE FROM forecasts 
                WHERE forecast_date < :cutoff_date
            """), {'cutoff_date': cutoff_date})
            
            deleted_count = result.rowcount
            
        return {
            'status': 'success',
            'deleted_forecasts': deleted_count,
            'cutoff_date': cutoff_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in cleanup_old_forecasts: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

def save_forecast_results(stock_symbol: str, forecast_dates: pd.DatetimeIndex, 
                         forecast_prices: np.ndarray, current_price: float) -> None:
    """Save forecast results to database"""
    try:
        with engine.connect() as conn:
            for date, price in zip(forecast_dates, forecast_prices):
                conn.execute(text("""
                    INSERT INTO forecasts (stock_symbol, forecast_date, forecast_price, current_price, created_at)
                    VALUES (:symbol, :date, :price, :current_price, :created_at)
                    ON CONFLICT (stock_symbol, forecast_date) 
                    DO UPDATE SET 
                        forecast_price = :price,
                        current_price = :current_price,
                        updated_at = :created_at
                """), {
                    'symbol': stock_symbol,
                    'date': date.date(),
                    'price': float(price),
                    'current_price': float(current_price),
                    'created_at': datetime.now()
                })
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error saving forecast results: {str(e)}")
        raise

def save_historical_data(stock_symbol: str, data: pd.DataFrame) -> None:
    """Save historical data to database"""
    try:
        with engine.connect() as conn:
            for date, row in data.iterrows():
                conn.execute(text("""
                    INSERT INTO historical_data (stock_symbol, date, open, high, low, close, volume, created_at)
                    VALUES (:symbol, :date, :open, :high, :low, :close, :volume, :created_at)
                    ON CONFLICT (stock_symbol, date) 
                    DO UPDATE SET 
                        open = :open,
                        high = :high,
                        low = :low,
                        close = :close,
                        volume = :volume,
                        updated_at = :created_at
                """), {
                    'symbol': stock_symbol,
                    'date': date.date(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume']),
                    'created_at': datetime.now()
                })
            conn.commit()
            
    except Exception as e:
        logger.error(f"Error saving historical data: {str(e)}")
        raise

if __name__ == '__main__':
    app.start()
