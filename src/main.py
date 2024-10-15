# main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import aio_pika
import json

from .workers.WorkItems import WorkItem  # Adjust the import based on your project structure
from .worker_init import celeryManager  # Assuming this is set up correctly
from pydantic import BaseModel
app = FastAPI()

class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcast a message to all active connections."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Handle or log exception
                pass

manager = ConnectionManager()


class Image_Request(BaseModel):
    image_url: str
    protein: str
    tissue_type: str


@app.post("/add_data")
async def add_data(req: Image_Request):
    result = celeryManager.add_task_to_queue(task_name='add_data', image_url=req.image_url, protein=req.protein, tissue_type=req.tissue_type)
    return {"task_id": result.id}

@app.get("/add_data/{task_id}")
async def example_result(task_id: str):
    result = celeryManager.get_task_result(task_name='add_data', task_id=task_id)
    return result

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Connect to RabbitMQ
        connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq:5672/")
        channel = await connection.channel()
        exchange = await channel.declare_exchange("task_updates", aio_pika.ExchangeType.FANOUT, durable=True)
        queue = await channel.declare_queue("", exclusive=True)
        await queue.bind(exchange)

        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    decoded_message = message.body.decode()
                    try:
                        parsed_message = json.loads(decoded_message)
                        await manager.broadcast(parsed_message)
                    except json.JSONDecodeError:
                        await manager.broadcast({"message": decoded_message})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)
    finally:
        await channel.close()
        await connection.close()
