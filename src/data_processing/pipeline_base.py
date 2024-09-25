import json

class PipelineBase:
    """
    Base class to act as template for celery task execution.
    """

    def __init__(self):
        self.task_name = 'Your task name'
        self.startup_message = 'STARTED'
        self.success_message = 'SUCCESS'

    def execute(self):
        """
        Implement execution logic here.
        """
        message = "Hello World"
        
        return message
