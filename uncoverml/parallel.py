import time
import click
import uncoverml.defaults
import ipyparallel as ipp

def task_view(profile):
    c = ipp.Client(profile=profile) if profile is not None else ipp.Client()
    return c.load_balanced_view()

def direct_view(profile):
    c = ipp.Client(profile=profile) if profile is not None else ipp.Client()
    return c[:]

def print_async_progress(async_result, title):
    total_jobs = len(async_result)
    with click.progressbar(length=total_jobs, label=title) as bar:
        last_jobs_done = 0
        jobs_done = 0
        while jobs_done < total_jobs:
            jobs_done = async_result.progress
            if jobs_done > last_jobs_done:
                bar.update(jobs_done - last_jobs_done)
                last_jobs_done = jobs_done
            time.sleep(0.1)

def map(f, iterable, cluster_view=None):
    # Send off the jobs
    progress_title = "Processing Image Chunks"
    if cluster_view is not None:
        async_result = cluster_view.map(f, iterable, block=False)
        print_async_progress(async_result, progress_title)
        results = async_result.get()
    else:
        results = []
        with click.progressbar(length=len(iterable),
                               label=progress_title) as bar: 
            for i in iterable:
                r = f(i)
                results.append(r)
                bar.update(1)
    return results
