# Base image with python

FROM python:3.10-slim

# Set the working directory in the container

WORKDIR /app

# Copy the requirements file to the working directory

COPY requirements.txt .

# Install the dependencies with extended timeout and retries

RUN pip install --no-cache-dir --timeout=300 --retries=5 -r requirements.txt

# Copy the rest of the application code to the working directory

COPY . .

# Expose the port that Streamlit will run on

EXPOSE 8501

# Command to run the Streamlit app

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]