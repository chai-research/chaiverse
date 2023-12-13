from contextlib import asynccontextmanager

import discord
from discord.ext import commands
from discord_bot import config
from discord_bot.cogs import ChatCog, DeveloperKeyCog, InfoCog


class DiscordBotClient(commands.Bot):
    async def setup_hook(self) -> None:
        await self.add_cog(ChatCog(self))
        await self.add_cog(DeveloperKeyCog(self))
        await self.add_cog(InfoCog(self))
        guild = discord.Object(config.GUILD_ID)
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)


@asynccontextmanager
async def discord_bot_client():
    intents = discord.Intents.all()
    async with DiscordBotClient(
        command_prefix='/',
        intents=intents,
        aplication_id=config.APPLICATION_ID
    ) as bot_client:
        yield bot_client
