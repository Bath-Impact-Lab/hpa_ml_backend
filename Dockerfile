# Base image with Miniconda
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy your environment.yml file into the container
COPY environment.yml /app/environment.yml

# Install the conda environment
RUN conda env create -f /app/environment.yml

# Activate the environment and install additional dependencies
RUN echo "conda activate hpa_ml_worker" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Copy the rest of the project files into the container
COPY . /app

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]