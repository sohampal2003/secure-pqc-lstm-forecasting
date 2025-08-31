-- Database initialization script for Time Series Forecasting + PQC
-- This script creates the necessary tables for storing historical data and forecasts

-- Create database if it doesn't exist
-- Note: This needs to be run manually as CREATE DATABASE cannot be run in a transaction
-- CREATE DATABASE forecasting_db;

-- Connect to the database
-- \c forecasting_db;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create historical_data table
CREATE TABLE IF NOT EXISTS historical_data (
    id SERIAL PRIMARY KEY,
    stock_symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 4) NOT NULL,
    high DECIMAL(10, 4) NOT NULL,
    low DECIMAL(10, 4) NOT NULL,
    close DECIMAL(10, 4) NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_price CHECK (open > 0 AND high > 0 AND low > 0 AND close > 0),
    CONSTRAINT valid_volume CHECK (volume >= 0),
    CONSTRAINT valid_date CHECK (date <= CURRENT_DATE),
    
    -- Indexes
    UNIQUE(stock_symbol, date)
);

-- Create forecasts table
CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    stock_symbol VARCHAR(10) NOT NULL,
    forecast_date DATE NOT NULL,
    forecast_price DECIMAL(10, 4) NOT NULL,
    current_price DECIMAL(10, 4) NOT NULL,
    confidence_score DECIMAL(5, 4) DEFAULT 0.0,
    model_version VARCHAR(50),
    model_parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_forecast_price CHECK (forecast_price > 0),
    CONSTRAINT valid_current_price CHECK (current_price > 0),
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CONSTRAINT valid_forecast_date CHECK (forecast_date > CURRENT_DATE),
    
    -- Indexes
    UNIQUE(stock_symbol, forecast_date)
);

-- Create model_performance table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    stock_symbol VARCHAR(10) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    training_date TIMESTAMP WITH TIME ZONE NOT NULL,
    test_date TIMESTAMP WITH TIME ZONE NOT NULL,
    mse DECIMAL(10, 6) NOT NULL,
    mae DECIMAL(10, 6) NOT NULL,
    rmse DECIMAL(10, 6) NOT NULL,
    r2_score DECIMAL(5, 4),
    model_parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_metrics CHECK (mse >= 0 AND mae >= 0 AND rmse >= 0),
    CONSTRAINT valid_r2 CHECK (r2_score >= -1 AND r2_score <= 1)
);

-- Create task_queue table for tracking background tasks
CREATE TABLE IF NOT EXISTS task_queue (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(255) UNIQUE NOT NULL,
    task_name VARCHAR(100) NOT NULL,
    stock_symbol VARCHAR(10),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    priority INTEGER DEFAULT 5,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    result JSONB,
    error_message TEXT,
    
    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
    CONSTRAINT valid_priority CHECK (priority >= 1 AND priority <= 10)
);

-- Create user_preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) UNIQUE NOT NULL,
    default_stock_symbol VARCHAR(10) DEFAULT 'AAPL',
    default_forecast_days INTEGER DEFAULT 10,
    default_sequence_length INTEGER DEFAULT 10,
    default_hidden_size INTEGER DEFAULT 50,
    default_num_layers INTEGER DEFAULT 2,
    default_training_epochs INTEGER DEFAULT 50,
    enable_pqc BOOLEAN DEFAULT TRUE,
    notification_preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_forecast_days CHECK (default_forecast_days >= 1 AND default_forecast_days <= 365),
    CONSTRAINT valid_sequence_length CHECK (default_sequence_length >= 5 AND default_sequence_length <= 100),
    CONSTRAINT valid_hidden_size CHECK (default_hidden_size >= 10 AND default_hidden_size <= 500),
    CONSTRAINT valid_num_layers CHECK (default_num_layers >= 1 AND default_num_layers <= 10),
    CONSTRAINT valid_training_epochs CHECK (default_training_epochs >= 10 AND default_training_epochs <= 1000)
);

-- Create audit_log table
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(100),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_date ON historical_data(stock_symbol, date);
CREATE INDEX IF NOT EXISTS idx_historical_data_date ON historical_data(date);
CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_date ON forecasts(stock_symbol, forecast_date);
CREATE INDEX IF NOT EXISTS idx_forecasts_date ON forecasts(forecast_date);
CREATE INDEX IF NOT EXISTS idx_model_performance_symbol ON model_performance(stock_symbol);
CREATE INDEX IF NOT EXISTS idx_task_queue_status ON task_queue(status);
CREATE INDEX IF NOT EXISTS idx_task_queue_priority ON task_queue(priority);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_action ON audit_log(user_id, action);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);

-- Create full-text search indexes
CREATE INDEX IF NOT EXISTS idx_historical_data_symbol_trgm ON historical_data USING gin(stock_symbol gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_trgm ON forecasts USING gin(stock_symbol gin_trgm_ops);

-- Create views for common queries
CREATE OR REPLACE VIEW recent_forecasts AS
SELECT 
    f.stock_symbol,
    f.forecast_date,
    f.forecast_price,
    f.current_price,
    f.confidence_score,
    f.created_at,
    ROUND(((f.forecast_price - f.current_price) / f.current_price * 100), 2) as price_change_percent
FROM forecasts f
WHERE f.forecast_date >= CURRENT_DATE
ORDER BY f.created_at DESC;

CREATE OR REPLACE VIEW stock_summary AS
SELECT 
    h.stock_symbol,
    COUNT(h.id) as data_points,
    MIN(h.date) as first_date,
    MAX(h.date) as last_date,
    AVG(h.close) as avg_price,
    STDDEV(h.close) as price_volatility,
    COUNT(f.id) as active_forecasts
FROM historical_data h
LEFT JOIN forecasts f ON h.stock_symbol = f.stock_symbol AND f.forecast_date > CURRENT_DATE
GROUP BY h.stock_symbol
ORDER BY h.stock_symbol;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_historical_data_updated_at 
    BEFORE UPDATE ON historical_data 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_forecasts_updated_at 
    BEFORE UPDATE ON forecasts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at 
    BEFORE UPDATE ON user_preferences 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create function to calculate forecast accuracy
CREATE OR REPLACE FUNCTION calculate_forecast_accuracy(
    p_stock_symbol VARCHAR(10),
    p_days_back INTEGER DEFAULT 30
)
RETURNS TABLE(
    forecast_date DATE,
    forecast_price DECIMAL(10, 4),
    actual_price DECIMAL(10, 4),
    absolute_error DECIMAL(10, 4),
    percentage_error DECIMAL(5, 2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        f.forecast_date,
        f.forecast_price,
        h.close as actual_price,
        ABS(f.forecast_price - h.close) as absolute_error,
        ROUND(ABS((f.forecast_price - h.close) / h.close * 100), 2) as percentage_error
    FROM forecasts f
    JOIN historical_data h ON f.stock_symbol = h.stock_symbol AND f.forecast_date = h.date
    WHERE f.stock_symbol = p_stock_symbol
    AND f.forecast_date >= CURRENT_DATE - p_days_back
    ORDER BY f.forecast_date;
END;
$$ LANGUAGE plpgsql;

-- Insert sample data for testing
INSERT INTO user_preferences (user_id, default_stock_symbol, default_forecast_days)
VALUES ('default_user', 'AAPL', 10)
ON CONFLICT (user_id) DO NOTHING;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO forecasting_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO forecasting_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO forecasting_user;

-- Create comments for documentation
COMMENT ON TABLE historical_data IS 'Stores historical stock price data';
COMMENT ON TABLE forecasts IS 'Stores generated price forecasts';
COMMENT ON TABLE model_performance IS 'Tracks model performance metrics';
COMMENT ON TABLE task_queue IS 'Manages background task execution';
COMMENT ON TABLE user_preferences IS 'Stores user-specific configuration';
COMMENT ON TABLE audit_log IS 'Tracks user actions for security and compliance';

COMMENT ON FUNCTION calculate_forecast_accuracy IS 'Calculates forecast accuracy metrics for a given stock';
COMMENT ON VIEW recent_forecasts IS 'Shows recent forecasts with price change calculations';
COMMENT ON VIEW stock_summary IS 'Provides summary statistics for each stock';
