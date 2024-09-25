import aio_pika
from celery import Celery
from ..data_processing.pipeline_base import PipelineBase
import json
import asyncio



class CeleryManager:
    """
    CeleryManager is responsible for managing and queuing tasks in a Celery-based
    environment. It allows dynamic task creation from processing pipelines and ensures 
    communication of task statuses via RabbitMQ. Users can queue tasks for execution 
    and track their progress through status updates.

    Attributes:
        task_map (dict): A dictionary mapping task names to their respective Celery task functions.
        app (Celery): The Celery application instance used to configure and manage the tasks.

    Methods:
        add_task_to_queue(task_name: str, **kwargs):
            Queues a task for execution based on the provided task name and input arguments.
            The task must be pre-defined in the task_map. Raises a ValueError if the task 
            name is not found.

    Usage Example:
        task_builder = CeleryTaskBuilder(pipelines)
        task_builder.add_task_to_queue('task_name', arg1=value1, arg2=value2)
        
    """

    def __init__(self, task_pipelines: list[PipelineBase], app, broker_url = 'amqp://guest:guest@localhost:5672//'):
                
        # Initialize Celery app with the name 'tasks'
        self.app = app
        self.app.config_from_object('src.workers.celeryconfig')
        self.broker_url = broker_url
        # Map task names to their respective Celery tasks created from pipelines
        self.task_map = {
            pipeline.task_name: self._create_celery_task(pipeline) for pipeline in task_pipelines
        }

    @staticmethod
    async def _send_status_update(task_id: str, status: str, broker_url: str):
        """
        Sends a task status update message to a RabbitMQ exchange.
        :param task_id: ID of the task
        :param status: Status of the task (e.g., 'started', 'success', 'failed')
        """
        # Create the message body as a dictionary
        message_body = {
            "task_id": task_id,
            "status": status
        }

        # Convert the dictionary to a JSON string
        json_message = json.dumps(message_body)

        # Set up a robust connection to RabbitMQ
        connection = await aio_pika.connect_robust(broker_url)
        async with connection:
            # Open a channel in the connection
            channel = await connection.channel()

            # Declare an exchange (type FANOUT) to broadcast the status message to all consumers
            exchange = await channel.declare_exchange("task_updates", aio_pika.ExchangeType.FANOUT, durable=True)

            # Publish the status update to the RabbitMQ exchange
            message = aio_pika.Message(json_message.encode())
            await exchange.publish(message, routing_key="update_message")

            # Close the RabbitMQ connection after the message is sent
            await connection.close()

    def _create_celery_task(self, pipeline: PipelineBase):
        """
        Creates a Celery task for a given pipeline.
        :param pipeline: A pipeline object responsible for processing the task
        :return: The Celery task function
        """
        broker_url_ = self.broker_url
        @self.app.task(bind=True)
        def task_wrapper(self, **kwargs):
            try:
                # Send a 'started' status update to RabbitMQ
                asyncio.run(CeleryManager._send_status_update(self.request.id, pipeline.startup_message, broker_url_))
                
                # Execute the task using the pipeline
                result = pipeline.execute(**kwargs)

                # Send a 'success' status update to RabbitMQ after the task completes
                asyncio.run(CeleryManager._send_status_update(self.request.id, pipeline.success_message, broker_url_))

                # Return the successful result as a JSON string
                return json.dumps(result)

            except Exception as e:
                # Handle any exception gracefully and log the error
                error_message = f"Task {self.request.id} failed with error: {str(e)}"
                print(error_message)  # Ideally, you'd use a proper logging system

                # Send an 'error' status update to RabbitMQ
                asyncio.run(CeleryManager._send_status_update(self.request.id, f"error: {str(e)}", broker_url_))

                # Return an error response
                return json.dumps({
                    "status": "failure",
                    "error": str(e)
                })

        return task_wrapper

    def add_task_to_queue(self, task_name: str, **kwargs):
        """
        Adds a task to the task queue to be executed
        :param task_name: The name of the task to be executed
        :param kwargs: Input arguments required for the task execution
        """
        if task_name in self.task_map:
            # Trigger the task using the mapped Celery task
            return self.task_map[task_name].delay(**kwargs)
        else:
            raise ValueError(f"Task '{task_name}' not found in the task map.")

    def get_task_result(self, task_name: str, task_id: str):
        """
        Gets the result/state of a task
        :param task_name: The name of the task thats been executed
        :param task_id: The task id to query for
        """
        if task_name in self.task_map:
            # Trigger the task using the mapped Celery task
            result = self.task_map[task_name].AsyncResult(task_id)
            if result.state == 'SUCCESS':
                return result.result
            else:
                return f'{task_id}: {result.state}'
        else:
            raise ValueError(f"Task '{task_name}' not found in the task map.")