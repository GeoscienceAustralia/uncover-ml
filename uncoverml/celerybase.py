import time
import pyprind

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
    bar = pyprind.ProgBar(total_jobs, width=60, title=title)
    last_jobs_done = 0
    jobs_done = 0
    while jobs_done < total_jobs:
        jobs_done = 0
        for r in async_results:
            jobs_done += int(r.ready())
        if jobs_done > last_jobs_done:
            bar.update(jobs_done - last_jobs_done, force_flush=True)
            last_jobs_done = jobs_done
        time.sleep(0.1)


def apply_over_chunks(f, data, **kwargs):

    # Build the chunk indices for creating jobs
    # chunk_indices = [(x, y) for x in range(splits) for y in range(splits)]
    # chunk_indices = range(nchunks)

    # Send off the jobs
    progress_title = "Processing Image Chunks"
    if not standalone:
        async_results = []
        for i, d in in enumerate(data):
            r = f.delay(i, d, **kwargs)
            async_results.append(r)
        print_celery_progress(async_results, progress_title)
        results = [r.result for r in async_results]
    else:
        bar = pyprind.ProgBar(len(chunk_indices), width=60,
                              title=progress_title)
        results = []
        for i, d in enumerate(data):
            results.append(f(i, d, **kwargs))
            bar.update(force_flush=True)

    return results
