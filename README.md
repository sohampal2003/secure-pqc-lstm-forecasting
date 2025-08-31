# Secure Financial Forecasting with LSTM and ML-KEM-512

This project demonstrates:
- Sequential prediction of Apple stock prices using an LSTM network.
- Post-quantum encryption of model weights using ML-KEM-512.

## Steps to Run

### Quick Start (Shell Script)
1. Make the script executable and run:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

### Local
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the training and encryption:
   ```bash
   python main.py
   ```

### Docker
1. Build Docker image:
   ```bash
   docker build -t secure-pqc-lstm .
   ```

2. Run container:
   ```bash
   docker run --rm secure-pqc-lstm
   ```
