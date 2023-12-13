import pytest
from unittest.mock import AsyncMock

from discord_bot.cogs import InfoCog


@pytest.mark.asyncio
async def test_slash_info_will_return_info():
    interaction = AsyncMock()

    cog = InfoCog()
    await cog.info.callback(cog, interaction)

    assert 'commands you need to know' in interaction.response.send_message.call_args_list[0].args[0]
