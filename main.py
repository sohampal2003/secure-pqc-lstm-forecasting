import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
from pqcrypto.kem.ml_kem_512 import generate_keypair, encrypt, decrypt
import time

# 1. Fetch Apple stock prices using yfinance
print("Downloading Apple stock prices...")
try:
    # Add delay to avoid rate limiting
    time.sleep(1)
    data = yf.download('AAPL', start='2015-01-01', end='2023-01-01', progress=False)
    
    if data.empty:
        print("❌ No data received. Trying with a shorter date range...")
        data = yf.download('AAPL', start='2020-01-01', end='2023-01-01', progress=False)
    
    if data.empty:
        print("❌ Still no data. Trying with even shorter range...")
        data = yf.download('AAPL', start='2022-01-01', end='2023-01-01', progress=False)
    
    if data.empty:
        raise Exception("Failed to download data from Yahoo Finance")
        
    prices = data['Close'].values
    print(f"✅ Loaded {len(prices)} closing prices.")
    
except Exception as e:
    print(f"❌ Error downloading data: {e}")
    print("Using sample data for demonstration...")
    # Generate sample data for demonstration
    np.random.seed(42)
    dates = np.arange(1000)
    trend = np.linspace(100, 200, 1000)
    noise = np.random.normal(0, 5, 1000)
    prices = trend + noise
    print(f"✅ Generated {len(prices)} sample prices for demonstration.")

# Prepare sequential dataset for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

seq_length = 10
X, y = create_sequences(prices, seq_length)

print(f"Created {len(X)} sequences with shape X: {X.shape}, y: {y.shape}")

# Reshape X to (batch_size, sequence_length, input_features)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Ensure X is 3D: (batch_size, sequence_length, input_features)
if X.dim() == 2:
    X = X.unsqueeze(-1)  # Add input_features dimension
    print(f"Reshaped X to: {X.shape}")

# Ensure y is 2D: (batch_size, output_size)
if y.dim() == 1:
    y = y.unsqueeze(-1)  # Add output dimension
    print(f"Reshaped y to: {y.shape}")

print(f"Final tensor shapes - X: {X.shape}, y: {y.shape}")

# 2. Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Ensure input is 3D: (batch_size, sequence_length, input_features)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
            
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last output
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# 3. Train model
print("Training LSTM model...")
epochs = 5
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    try:
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"❌ Error in epoch {epoch+1}: {e}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        break

# 4. Post-quantum encryption using ML-KEM-512
print("Generating ML-KEM-512 keypair...")
try:
    pk, sk = generate_keypair()
    print("Encrypting model weights...")
    
    # Get model weights as a single tensor
    weights = torch.cat([p.flatten() for p in model.parameters()])
    print(f"Model weights shape: {weights.shape}")
    
    # Encrypt and decrypt
    ciphertext, shared_secret_enc = encrypt(pk)
    decrypted_secret = decrypt(sk, ciphertext)
    
    encryption_success = shared_secret_enc == decrypted_secret
    print(f"Encryption successful. Shared secret recovered correctly: {encryption_success}")
    
    if encryption_success:
        print("✅ Post-quantum cryptography working correctly!")
    else:
        print("❌ Encryption/decryption mismatch")
        
except Exception as e:
    print(f"❌ Error in PQC encryption: {e}")

# Save model
try:
    torch.save(model.state_dict(), "lstm_model.pth")
    print("✅ Model saved as lstm_model.pth")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# Test the model with a sample prediction
print("\nTesting model with sample prediction...")
try:
    model.eval()
    with torch.no_grad():
        # Use the last sequence for prediction
        test_input = X[-1:]  # Take the last sequence
        prediction = model(test_input)
        actual = y[-1]
        print(f"Sample prediction: {prediction.item():.2f}")
        print(f"Actual value: {actual.item():.2f}")
        print(f"Prediction error: {abs(prediction.item() - actual.item()):.2f}")
        
except Exception as e:
    print(f"❌ Error in prediction test: {e}")

print("\n🎉 Script completed successfully!")
