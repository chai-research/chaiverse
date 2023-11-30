import logging

from discord.ext import commands


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InfoCog(commands.Cog):

    @commands.command(description="Information about the bot and how to use it.")
    async def info(self, ctx):
        await ctx.reply(
            "Info2: You can chat with any currently deployed model and a bot from the list in this channel.\n"
            "Here are some commands you need to know:\n"
            "1. `/chat` — Starts conversation with the bot, served by deployed submission.\n"
            "2. `/info` — Prints this message.\n"
            "Enjoy ❤️",
            mention_author=False
        )
