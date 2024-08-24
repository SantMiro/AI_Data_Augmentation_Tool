# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements_docker.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements_docker.txt

# Copy the rest of your application code into the container
COPY src/ /app/

# Set the Python path to include the src directory
ENV PYTHONPATH=/app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for Flask
ENV FLASK_APP=src/app.py

# Run your Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
