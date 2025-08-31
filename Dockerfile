# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run main script
CMD ["python", "main.py"]
