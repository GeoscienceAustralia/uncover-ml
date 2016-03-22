"""
Start a Celery image worker.

.. program-output:: uncoverml-worker --help
"""
import logging
import click as cl
from uncoverml import celerybase
import uncoverml.tasks

log = logging.getLogger(__name__)

@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output", default=False)
@cl.option('--redisdb', type=int, default=0)
@cl.option('--redishost', type=str, default='localhost')
@cl.option('--redisport', type=int, default=6379)
def main(quiet, redisdb, redishost, redisport): 
    

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
        # initialise celery
    celerybase.configure(redishost, redisport, redisdb, standalone=False)

    
    # run the worker
    celerybase.celery.worker_main()
