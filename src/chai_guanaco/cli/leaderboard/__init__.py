from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from chai_guanaco.cli.application import Application


@click.command(short_help='Display the leaderboard')
@click.pass_obj
def leaderboard(app: Application):
    from chai_guanaco import display_leaderboard

    display_leaderboard(developer_key=app.api_key)
