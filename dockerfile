# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy bot directory and all its files
COPY Bot/ ./bot/

# Install requirements if requirements.txt exists
RUN if [ -f bot/requirements.txt ]; then pip install --no-cache-dir -r bot/requirements.txt; fi

# Set entrypoint to run query_pot.py
CMD ["python", "bot/query_bot.py"]