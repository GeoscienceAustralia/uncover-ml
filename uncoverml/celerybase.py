import time
import click
from celery import Celery
import uncoverml.defaults

celery = Celery('uncoverml.tasks')
celery.config_from_object(uncoverml.defaults)
    
def configure(host, port, db, standalone=False):
    if not standalone:
        celery_redis='redis://{}:{}/{}'.format(host, port, db)
        celery.conf.BROKER_URL = celery_redis
        celery.conf.CELERY_RESULT_BACKEND = celery_redis

def print_celery_progress(async_results, title):
    total_jobs = len(async_results)
    with click.progressbar(length=total_jobs) as bar:
        last_jobs_done = 0
        jobs_done = 0
        while jobs_done < total_jobs:
            jobs_done = 0
            for r in async_results:
                jobs_done += int(r.ready())
            if jobs_done > last_jobs_done:
                bar.update(jobs_done - last_jobs_done)
                last_jobs_done = jobs_done
            time.sleep(0.1)


def map_over(f, iterable, standalone=False, **kwargs):
    # Send off the jobs
    progress_title = "Processing Chunks"
    if not standalone:
        async_results = [f.delay(i, **kwargs) for i in iterable]
        print_celery_progress(async_results, progress_title)
    else:
        with click.progressbar(length=len(iterable)) as bar: 
            for i in iterable:
                r = f(i, **kwargs)
                bar.update(1)
