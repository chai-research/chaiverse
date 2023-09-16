from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from chai_guanaco.cli.application import Application


@click.command(short_help='Update credentials for accessing the platform')
@click.pass_obj
def login(app: Application):
    import json

    api_key = click.prompt('Please enter your API key', hide_input=True)

    app.credentials_file.parent.mkdir(parents=True, exist_ok=True)
    app.credentials_file.write_text(json.dumps({'api_key': api_key}), encoding='utf-8')
