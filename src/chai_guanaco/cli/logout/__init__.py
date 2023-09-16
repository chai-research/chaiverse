from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from chai_guanaco.cli.application import Application


@click.command(short_help='Remove credentials')
@click.pass_obj
def logout(app: Application):
    if app.credentials_file.is_file():
        app.credentials_file.unlink()
