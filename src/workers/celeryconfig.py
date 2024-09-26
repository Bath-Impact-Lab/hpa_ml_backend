import os

broker_url = os.getenv('CELERY_BROKER_URL', 'amqp://guest:guest@rabbitmq:5672//')

result_backend = 'rpc://'


# Optional: concurrency and other settings
worker_concurrency = 4
result_expires = 3600
enable_utc = True

