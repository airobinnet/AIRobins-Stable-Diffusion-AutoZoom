# syntax=docker/dockerfile:1

# Use an official Python runtime as a parent image
FROM nvidia/cuda:11.7.1-base-ubuntu22.04

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
# ADD . /app
COPY main_local.py /app
COPY requirements.txt /app
COPY cache /app/cache
COPY static /app/static
COPY templates /app/templates

# Install system dependencies
RUN apt-get update && apt-get install -y \
  python3-pip \
  python3.11 \
  libgl1-mesa-glx \
  libglib2.0-0

# Upgrade pip
RUN pip3 install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5002 available to the world outside this container
EXPOSE 5002

# Run main_local.py when the container launches
CMD ["python3", "main_local.py"]