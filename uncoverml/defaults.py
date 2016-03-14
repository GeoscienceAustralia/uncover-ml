
redis_address = 'localhost'
redis_port = 6379
redis_db = 0

#CELERY CONFIGS
celery_redis = 'redis://{}:{}/{}'.format(redis_address, redis_port, redis_db)
BROKER_URL = celery_redis
CELERY_RESULT_BACKEND = celery_redis
CELERY_TASK_SERIALIZER = 'pickle'
CELERY_ACCEPT_CONTENT = ['pickle']
