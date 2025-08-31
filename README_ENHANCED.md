# Time Series Forecasting + PQC - Enhanced Version

A comprehensive time series forecasting system with Post-Quantum Cryptography (PQC) protection, featuring a real-time Streamlit dashboard and automated CI/CD pipeline.

## 🚀 New Features

### 1. **Streamlit Dashboard for Real-time Forecasts**
- Interactive web interface for stock price forecasting
- Real-time data from Yahoo Finance
- Configurable LSTM model parameters
- Beautiful Plotly visualizations
- Post-quantum cryptography integration
- Download forecasts as CSV

### 2. **CI/CD Pipeline for Auto-deployment**
- GitHub Actions workflow
- Automated testing and security scanning
- Docker image building and testing
- Multi-environment deployment (staging/production)
- Performance testing and monitoring
- Automated documentation generation

### 3. **Background Task Processing**
- Celery-based asynchronous forecasting
- Redis for task queuing and caching
- PostgreSQL for data persistence
- Scheduled data updates and cleanup

### 4. **Monitoring and Observability**
- Prometheus metrics collection
- Grafana dashboards
- Health checks and logging
- Performance monitoring

### 5. **Production-Ready Infrastructure**
- Docker Compose orchestration
- Nginx reverse proxy with SSL
- Load balancing and rate limiting
- Security headers and SSL termination

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Celery        │    │   PostgreSQL    │
│   Dashboard     │◄──►│   Worker        │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │      Redis      │    │   Prometheus    │
│   Reverse Proxy │    │   Task Queue    │    │   Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Grafana     │    │      Flower     │    │   CI/CD         │
│   Dashboards    │    │   Monitoring    │    │   Pipeline      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Git
- OpenSSL (for SSL certificates)

## 🚀 Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd secure_pqc_lstm_yfinance_project

# Make deployment script executable
chmod +x deploy.sh

# Deploy everything
./deploy.sh deploy
```

### Option 2: Manual Setup

```bash
# Create necessary directories
mkdir -p data models logs ssl grafana/provisioning/{datasources,dashboards}

# Generate SSL certificates
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Start services
docker-compose up -d
```

## 🌐 Access Points

After deployment, access the services at:

- **Streamlit Dashboard**: https://localhost
- **Grafana**: https://localhost/grafana/ (admin/admin)
- **Prometheus**: https://localhost/prometheus/
- **Flower (Celery)**: https://localhost/flower/

## 📊 Using the Streamlit Dashboard

### 1. **Basic Forecasting**
1. Open https://localhost in your browser
2. Select a stock symbol (e.g., AAPL, GOOGL, MSFT)
3. Choose date range and model parameters
4. Click "Start Analysis"
5. View forecasts and download results

### 2. **Advanced Configuration**
- **Sequence Length**: Number of historical data points for LSTM input
- **Hidden Size**: LSTM hidden layer dimensions
- **Number of Layers**: LSTM stack depth
- **Training Epochs**: Number of training iterations
- **Forecast Steps**: Days to predict into the future

### 3. **Post-Quantum Cryptography**
- Enable/disable PQC encryption
- ML-KEM-512 key generation
- Secure model weight storage

## 🔧 CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline includes:

1. **Code Quality**
   - Linting with flake8
   - Code formatting with black
   - Import sorting with isort

2. **Testing**
   - Unit tests with pytest
   - Coverage reporting
   - Multi-Python version testing

3. **Security**
   - Bandit security scanning
   - Safety dependency checks
   - Vulnerability assessment

4. **Build & Deploy**
   - Docker image building
   - Multi-environment deployment
   - Automated releases

### Setup CI/CD

1. **Fork/Clone Repository**
2. **Set GitHub Secrets**:
   ```bash
   DOCKER_USERNAME=your_dockerhub_username
   DOCKER_PASSWORD=your_dockerhub_password
   ```
3. **Push to Main Branch**: Triggers production deployment
4. **Push to Develop Branch**: Triggers staging deployment

## 🗄️ Database Schema

### Core Tables

- **`historical_data`**: Stock price history
- **`forecasts`**: Generated predictions
- **`model_performance`**: Model accuracy metrics
- **`task_queue`**: Background task tracking
- **`user_preferences`**: User configurations
- **`audit_log`**: Security audit trail

### Views

- **`recent_forecasts`**: Latest predictions
- **`stock_summary`**: Stock statistics

## 🔄 Background Tasks

### Celery Tasks

- **`forecast_stock`**: Generate stock price forecasts
- **`batch_forecast`**: Process multiple stocks
- **`update_historical_data`**: Refresh stock data
- **`cleanup_old_forecasts`**: Remove expired predictions

### Task Scheduling

- **Daily**: Update historical data
- **Weekly**: Clean up old forecasts
- **On-demand**: User-initiated forecasting

## 📈 Monitoring

### Prometheus Metrics

- Application performance
- Task execution times
- Error rates
- Resource utilization

### Grafana Dashboards

- Forecasting accuracy
- System health
- Task queue status
- Performance trends

## 🛡️ Security Features

- **Post-Quantum Cryptography**: ML-KEM-512 encryption
- **SSL/TLS**: Secure communication
- **Rate Limiting**: API protection
- **Security Headers**: XSS, CSRF protection
- **Audit Logging**: User action tracking

## 🐳 Docker Services

### Core Services

- **streamlit-dashboard**: Main application
- **postgres**: Database
- **redis**: Cache and task queue
- **nginx**: Reverse proxy

### Monitoring Services

- **prometheus**: Metrics collection
- **grafana**: Visualization
- **flower**: Celery monitoring

## 📝 Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# Celery
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Streamlit
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Docker Compose

```bash
# Start specific services
docker-compose up streamlit-dashboard postgres

# Scale workers
docker-compose up --scale celery-worker=3

# View logs
docker-compose logs -f streamlit-dashboard
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=./ --cov-report=html

# Run specific test file
pytest test_streamlit_dashboard.py
```

### Test Coverage

- Unit tests for core functions
- Integration tests for API endpoints
- End-to-end tests for workflows
- Performance benchmarks

## 📚 API Documentation

### Streamlit Dashboard

The dashboard provides a RESTful interface through Streamlit's components:

- **Stock Selection**: Choose from any valid stock symbol
- **Parameter Configuration**: Adjust model hyperparameters
- **Real-time Processing**: Live training and forecasting
- **Data Export**: Download results in multiple formats

### Background Tasks API

```python
from tasks import forecast_stock, batch_forecast

# Single stock forecast
task = forecast_stock.delay('AAPL', forecast_days=10)

# Batch forecast
task = batch_forecast.delay(['AAPL', 'GOOGL', 'MSFT'])

# Check status
print(task.status)
print(task.result)
```

## 🚀 Deployment Options

### 1. **Local Development**
```bash
./deploy.sh deploy
```

### 2. **Staging Environment**
```bash
git push origin develop  # Triggers staging deployment
```

### 3. **Production Environment**
```bash
git push origin main     # Triggers production deployment
```

### 4. **Custom Deployment**
```bash
# Build and push Docker image
docker build -t your-registry/time-series-forecasting:latest .
docker push your-registry/time-series-forecasting:latest

# Deploy to Kubernetes
kubectl apply -f k8s/
```

## 🔍 Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   lsof -i :8501
   
   # Change ports in docker-compose.yml
   ```

2. **Database Connection**
   ```bash
   # Check PostgreSQL logs
   docker-compose logs postgres
   
   # Test connection
   docker-compose exec postgres psql -U forecasting_user -d forecasting_db
   ```

3. **Celery Issues**
   ```bash
   # Check worker status
   docker-compose exec flower celery -A tasks inspect active
   
   # Restart workers
   docker-compose restart celery-worker
   ```

### Logs and Debugging

```bash
# View all logs
./deploy.sh logs

# View specific service logs
docker-compose logs -f streamlit-dashboard

# Access container shell
docker-compose exec streamlit-dashboard bash
```

## 📊 Performance Optimization

### LSTM Model Tuning

- **Sequence Length**: 10-30 for daily data
- **Hidden Size**: 50-100 for good accuracy
- **Layers**: 2-4 for complex patterns
- **Epochs**: 50-200 based on convergence

### System Resources

- **Memory**: 4GB+ for model training
- **CPU**: Multi-core for parallel processing
- **Storage**: SSD for database performance
- **Network**: Stable internet for data updates

## 🔮 Future Enhancements

### Planned Features

1. **Advanced Models**
   - Transformer-based forecasting
   - Ensemble methods
   - Multi-variate analysis

2. **Real-time Streaming**
   - WebSocket connections
   - Live price updates
   - Instant notifications

3. **Machine Learning Pipeline**
   - AutoML for model selection
   - Hyperparameter optimization
   - A/B testing framework

4. **Enterprise Features**
   - Multi-tenant architecture
   - Role-based access control
   - Advanced analytics

## 🤝 Contributing

### Development Setup

1. **Fork Repository**
2. **Create Feature Branch**
3. **Make Changes**
4. **Run Tests**
5. **Submit Pull Request**

### Code Standards

- Follow PEP 8 style guide
- Add type hints
- Write comprehensive tests
- Update documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit**: Interactive web applications
- **PyTorch**: Deep learning framework
- **Post-Quantum Cryptography**: Future-proof security
- **Yahoo Finance**: Financial data provider
- **Open Source Community**: Tools and libraries

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: [Wiki](wiki-link)
- **Email**: support@example.com

---

**Built with ❤️ for the future of secure, intelligent financial forecasting**
