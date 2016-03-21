
from celery import Celery
import uncoverml.defaults

celery = Celery('uncoverml.tasks')
celery.config_from_object(uncoverml.defaults)
    
def configure(host, port, db):
    celery_redis='redis://{}:{}/{}'.format(host, port, db)
    celery.conf.BROKER_URL = celery_redis
    celery.conf.CELERY_RESULT_BACKEND = celery_redis


