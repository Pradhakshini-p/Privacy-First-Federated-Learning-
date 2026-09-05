"""Simple WebSocket client simulator for the Privacy-First Federated Learning Platform.
Connects to the server's /ws endpoint and prints incoming updates.

Run:
    python client_simulator.py --clients 3

This will start several simulated clients that connect and log updates.
"""
import asyncio
import websockets
import json
import argparse
import random
import time

WS_URL = "ws://localhost:8000/ws"

async def simulated_client(client_id: int):
    try:
        async with websockets.connect(WS_URL) as ws:
            print(f"Client {client_id} connected to {WS_URL}")
            # send initial hello
            await ws.send(json.dumps({"client_id": client_id, "type": "hello"}))
            while True:
                try:
                    data = await ws.recv()
                    payload = json.loads(data)
                    if payload.get("type") == "update":
                        print(f"[Client {client_id}] Round {payload['round']}: acc={payload['accuracy']} loss={payload['loss']} clients={payload['active_clients']} privacy={payload['privacy_budget']}")
                    elif payload.get('type') == 'echo':
                        print(f"[Client {client_id}] Echo: {payload.get('message')}")
                except websockets.exceptions.ConnectionClosed:
                    print(f"Client {client_id} connection closed")
                    break
    except Exception as e:
        print(f"Client {client_id} failed to connect: {e}")

async def main(num_clients: int):
    tasks = []
    for i in range(num_clients):
        tasks.append(asyncio.create_task(simulated_client(i+1)))
        await asyncio.sleep(0.2 + random.random() * 0.5)
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clients', type=int, default=2, help='Number of simulated clients')
    args = parser.parse_args()
    try:
        asyncio.run(main(args.clients))
    except KeyboardInterrupt:
        print('Stopped by user')
