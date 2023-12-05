import pytest
from unittest.mock import Mock, AsyncMock, patch

from discord_bot.cogs import DeveloperKeyCog


@pytest.mark.asyncio
@patch('discord_bot.cogs.developer_key_cog.auth')
async def test_on_member_join_will_obtain_key_and_send_to_user(auth):
    member = AsyncMock()
    member.name = 'mock-name'
    cog = DeveloperKeyCog()
    authenticator = Mock()
    auth.get_authenticator.return_value = authenticator
    authenticator.obtain_developer_key.return_value = 'mock-key'
    await cog.on_member_join(member)
    authenticator.obtain_developer_key.assert_called_once_with('mock-name')
    assert 'mock-key' in member.send.await_args_list[0].args[0]
