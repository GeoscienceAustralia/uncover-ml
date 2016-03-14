
from celery import Celery
import uncoverml.defaults

celery = Celery('uncoverml.tasks')
celery.config_from_object(uncoverml.defaults)

