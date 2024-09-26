from .data_processing.add_data_pipeline import add_data_pipeline
from .workers.CeleryManager import CeleryManager
from .data_processing.pipeline_base import PipelineBase
from celery import Celery

# Create celery app 
app=Celery("tasks")


# Create instances of Pipelines 
task_pipeline = add_data_pipeline()


# Create celery manager instance
celeryManager = CeleryManager(task_pipelines=[task_pipeline], app=app)


