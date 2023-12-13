import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
from multiprocessing import Process

from chaiverse import metrics as chai_metrics
import discord
from discord_bot import config, discord_bot_client


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
