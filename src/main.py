from fastapi import FastAPI
from .worker_init import celeryManager

'''
Example fastAPI APP to handle adding requests to celery workers

'''

app = FastAPI()

@app.post("/example")
async def example():
    result = celeryManager.add_task_to_queue(task_name='example')
    return {"task_id": result.id}

@app.get("/example/{task_id}")
async def example_result(task_id:str):
    result = celeryManager.get_task_result(task_name='example', task_id=task_id)
    return result






# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     # Set up asynchronous RabbitMQ connection
#     connection = await aio_pika.connect_robust("amqp://guest:guest@rabbitmq:5672/")
#     channel = await connection.channel()

#     # Declare the fanout exchange
#     exchange = await channel.declare_exchange("task_updates", aio_pika.ExchangeType.FANOUT, durable=True)

#     # Declare a unique, temporary, auto-deleted queue for this WebSocket connection
#     queue = await channel.declare_queue("status_message", exclusive=False, auto_delete=True)

#     # Bind the queue to the exchange
#     await queue.bind(exchange)

#     # Create an asynchronous consumer to receive messages
#     async with queue.iterator() as queue_iter:
#         async for message in queue_iter:
#             async with message.process():
#                 # Try to decode and parse the message as JSON
#                 try:
#                     decoded_message = message.body.decode()
#                     parsed_message = json.loads(decoded_message)
#                     await websocket.send_json(parsed_message)
#                 except json.JSONDecodeError:
#                     # If the message is not valid JSON, send it as plain text
#                     await websocket.send_text(decoded_message)

#     # Close the WebSocket connection
#     await channel.close()
#     await connection.close()
#     await websocket.close()
