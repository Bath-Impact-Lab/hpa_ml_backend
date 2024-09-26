
# Project Setup Guide


## Prerequisites

- Python 3.x
- Celery
- Uvicorn
- RabbitMQ (or another broker)
- Docker (optional, if using Docker for RabbitMQ)

Required dependencies are all bundled in the `environment.yml` file. You can use the following command to create a conda environment : 

```bash
conda env create -f /src/environment.yml
conda activate celeryworkertemplate
```

## 1. Setup Celery Workers

To start Celery workers, navigate to the `src/worker_init` directory and run the following Celery command:

```bash
cd src/worker_init
celery -A worker_init worker --loglevel=info
```

This will initialize the Celery workers in the `src/worker_init` module. Adjust the log level (`--loglevel=info`) as needed.

To enable multi-threaded workers run
```bash
cd src/worker_init
celery -A worker_init worker -P threads --loglevel=info
```

## 2. Start Uvicorn Server

To start the Uvicorn server, navigate to the `src/main` directory and run the following command:

```bash
cd src/main
uvicorn main:app --reload
```

This will start the Uvicorn server with automatic reloading enabled. If you want to run it in production mode, use the following command instead (adjust the host and port as needed):

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 3. Running with Docker (Optional)

If you're running RabbitMQ in Docker, ensure that the appropriate containers are running before starting the Celery workers or Uvicorn server. For example, to start RabbitMQ using Docker, you can use the following `docker-compose.yml` file:

```yaml
version: '3'
services:
  rabbitmq:
    image: rabbitmq:management
    ports:
      - "5672:5672"
      - "15672:15672"
```

Run the following command to start the services:

```bash
docker-compose up -d
```

Make sure to update the Celery broker URL and backend settings accordingly.

## 4. Environment Variables

Ensure that the following environment variables are correctly set up for both Celery and Uvicorn:

```bash
export CELERY_BROKER_URL='your-broker-url'
export CELERY_RESULT_BACKEND='your-result-backend'
```

For Uvicorn, set any additional environment variables needed by your app in the `src/main` directory.


