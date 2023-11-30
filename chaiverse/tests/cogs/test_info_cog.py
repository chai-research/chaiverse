import pytest
from unittest.mock import AsyncMock

from discord_bot.cogs import InfoCog


@pytest.mark.asyncio
async def test_slash_info_will_return_info():

    ctx = AsyncMock()

    cog = InfoCog()
    await cog.info(cog, ctx)

    assert 'commands you need to know' in ctx.reply.call_args_list[0].args[0]
