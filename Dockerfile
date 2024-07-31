# Use an official Python runtime as a parent image
FROM python:3.11-slim

COPY . /app

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--port", "8000"]
