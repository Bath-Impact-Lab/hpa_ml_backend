from .workers.CeleryManager import CeleryManager
from .data_processing.pipeline_base import PipelineBase
from celery import Celery

# Create celery app 
app=Celery("tasks")


# Create instances of Pipelines 
task_pipeline = PipelineBase(task_name='example')


# Create celery manager instance
celeryManager = CeleryManager(task_pipelines=[task_pipeline], app=app)


