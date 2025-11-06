FROM python:3.12-slim

# Set working directory
WORKDIR /ML

# Copy all files
COPY . /ML

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
