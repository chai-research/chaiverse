from contextlib import asynccontextmanager
import logging

import discord
from discord.ext import commands
from discord_bot import config
from discord_bot.cogs import ChatCog, DeveloperKeyCog, InfoCog
from guanaco_database import auth


@asynccontextmanager
async def discord_bot_client():
    intents = discord.Intents.default()
    intents.guild_messages = True
    intents.members = True

    async with discord.ext.commands.Bot(
        command_prefix='/',
        intents=intents,
        aplication_id=config.APPLICATION_ID
    ) as bot_client:
        await bot_client.add_cog(ChatCog())
        await bot_client.add_cog(DeveloperKeyCog())
        await bot_client.add_cog(InfoCog())
        yield bot_client
