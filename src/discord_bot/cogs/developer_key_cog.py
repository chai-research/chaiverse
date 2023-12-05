import logging

import discord
from discord.ext import commands
from guanaco_database import auth


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DeveloperKeyCog(commands.Cog):

    @commands.Cog.listener()
    async def on_member_join(self, member: discord.Member):
        developer_key = auth.get_authenticator().obtain_developer_key(member.name)
        logger.info('member_joined discord id={0} name={1} global_name={2}'.format(member.id, member.name, member.global_name))
        welcome_text = _get_welcome_text(member.name, developer_key)
        await member.send(welcome_text)


def _get_welcome_text(member_name, developer_key):
    welcome_text = f"Welcome to the Chai LLM Competition, {member_name} 🔥💰🚀\n\n" \
           "To submit solutions you need a developer key, here is yours:\n" \
           f"🔑 `{developer_key}` 🔑\n\n" \
           f"[Here is a Python Package](https://pypi.org/project/chai-guanaco/) that will help you deploy your " \
           f"models to real users for evaluations.\n\n " \
           f"P.S. Ask Tom (Team Chai) in Discord anything, he is kinda like a robot so never sleeps 🤖\n\n" \
           f"Meanwhile, [introduce yourself](https://discord.com/channels/1104020730678612001/1104028332753957025), " \
           f"checkout our [leaderboard](https://discord.com/channels/1104020730678612001/1134163974296961195) " \
           f"and [write-up](https://discord.com/channels/1104020730678612001/1112835255439740939) channels  🏆"
    return welcome_text

