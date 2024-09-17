import json

class PipelineBase:
    """
    Base class to act as template for celery task execution.
    """

    def __init__(self, task_name, startup_message="STARTED", success_message="SUCCESS"):
        self.task_name = task_name
        self.startup_message = startup_message
        self.success_message = success_message

    def execute(self):
        """
        Implement execution logic here.
        """
        message = "Hello World"
        
        return message
