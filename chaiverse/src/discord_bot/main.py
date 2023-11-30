import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
from multiprocessing import Process

import discord
from discord_bot import config, discord_bot_client
import uvicorn


# Dummy HTTP server so it can be deployed via CI/CD to Cloud run
# with other services (else painful to deploy)
def run_http_server():
    uvicorn.run("discord_bot_server:app", port=8000, log_level="info")


def run_http_server():
    server = HTTPServer(('', 8000), BaseHTTPRequestHandler)
    server.serve_forever()


async def run_discord_bot():
    async with discord_bot_client() as bot_client:
        discord.utils.setup_logging(level=logging.INFO)
        await bot_client.start(config.BOT_TOKEN, reconnect=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    server_proc = Process(target=run_http_server, daemon=True)
    server_proc.start()
    asyncio.run(run_discord_bot())
