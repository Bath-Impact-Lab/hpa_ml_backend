# workitem.py

import asyncio
import json
from enum import Enum
from aio_pika import connect_robust, Message, ExchangeType
import numpy as np

class PipelineLevel(Enum):
    SEGMENTATION = 'Segmentation'
    FEATURE_EXTRACTION = 'FeatureExtraction'
    DIMENSIONALITY_REDUCTION = 'DimensionalityReduction'
    CLUSTERING = 'Clustering'

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder that converts numpy ndarrays to lists."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if len(obj) > 100:
                return 'list too large'
            return obj.tolist()
        return super().default(obj)


class WorkItem:
    """
    WorkItem tracks the progress of an object moving through pipelines.

    Attributes:
    ----------
    status : str or None
        Current level of the work item in the pipeline.
    body : dict
        Holds key-value pairs containing information about the work item.
    levels : set
        A set of valid levels in the pipeline.
    """

    # Shared RabbitMQ connection and exchange
    _rabbitmq_url = "amqp://guest:guest@rabbitmq:5672/"
    _exchange_name = "task_updates"
    _exchange = None
    _connection = None
    _lock = asyncio.Lock()

    def __init__(self, status=None, body=None):
        self.body = body if body is not None else {}
        self._status = status

        self.levels = {level.value for level in PipelineLevel}

    @property
    def status(self):
        """Get the current status."""
        return self._status

    async def _get_exchange(self):
        """Initialize RabbitMQ connection and declare exchange if not already done."""
        async with self._lock:
            if self._exchange is None or self._connection.is_closed:
                print(self._exchange)
                self._connection = await connect_robust(self._rabbitmq_url)
                channel = await self._connection.channel()
                self._exchange = await channel.declare_exchange(
                    self._exchange_name, ExchangeType.FANOUT, durable=True
                )
        return self._exchange

    async def _publish_update(self, message: dict):
        """Publish a message to the RabbitMQ exchange."""
        exchange = await self._get_exchange()
        message_body = json.dumps(message, cls=NumpyEncoder).encode()
        try:
            await exchange.publish(Message(message_body), routing_key="")
        except Exception as e:
            print(f'exchange failed to publish: {e}')

    @status.setter
    def status(self, level):
        """Set the status if the level is valid, else raise ValueError."""
        if level not in self.levels:
            raise ValueError(f'Invalid level: {level}. Available levels are: {self.levels}')
        self._status = level
        # Publish the update
        asyncio.run(
            self._publish_update({
                "event": "status_update",
                "status": self._status,
                "body": self.body
            })
        )

    def update_status(self, level):
        """Update the status to the specified level."""
        asyncio.run(self._set_status_async(level))

    async def _set_status_async(self, level):
        self.status = level

    def set_attribute(self, key, value):
        """Set a key-value pair in the body dictionary."""
        if not isinstance(key, str):
            raise TypeError("Key must be a string")
        self.body[key] = value
        print(self.body)
        # Publish the update
        # asyncio.run(
        #     self._publish_update({
        #         "event": "attribute_update",
        #         "key": key,
        #         "value": value,
        #         "body": self.body
        #     })
        # )

    def get_attribute(self, key, default=None):
        """
        Get the value for the specified key from the body dictionary.
        If the key is not present, return the default value.
        """
        return self.body.get(key, default)

    def __repr__(self):
        return f"<WorkItem(status={self.status}, body={self.body})>"

    async def close(self):
        """Close the RabbitMQ connection."""
        if self._connection:
            await self._connection.close()
