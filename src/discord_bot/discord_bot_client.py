from contextlib import asynccontextmanager

import discord
from discord_bot import config
from discord_bot.cogs import ChatCog, DeveloperKeyCog, InfoCog


@asynccontextmanager
async def discord_bot_client():
    intents = discord.Intents.all()

    async with discord.ext.commands.Bot(
        command_prefix='/',
        intents=intents,
        aplication_id=config.APPLICATION_ID
    ) as bot_client:
        await bot_client.add_cog(ChatCog())
        await bot_client.add_cog(DeveloperKeyCog())
        await bot_client.add_cog(InfoCog())
        yield bot_client
