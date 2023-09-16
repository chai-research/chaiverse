from __future__ import annotations

import click

from chai_guanaco.cli.leaderboard import leaderboard
from chai_guanaco.cli.login import login
from chai_guanaco.cli.logout import logout


@click.group(
    context_settings={'help_option_names': ['-h', '--help'], 'max_content_width': 120}, invoke_without_command=True
)
@click.pass_context
def chai_guanaco(ctx: click.Context):
    """
    \b
      ____ _           _
     / ___| |__   __ _(_)
    | |   | '_ \\ / _` | |
    | |___| | | | (_| | |
     \\____|_| |_|\\__,_|_|
    """
    import json
    from pathlib import Path

    from chai_guanaco.cli.application import Application

    app = Application(ctx.exit)

    if app.credentials_file.is_file():
        credentials = json.loads(app.credentials_file.read_text(encoding='utf-8'))
        app.set_api_key(credentials['api_key'])
    else:
        app.credentials_file.parent.mkdir(parents=True, exist_ok=True)
        if (legacy_developer_key_file := Path.home() / '.chai-guanaco' / 'developer_key.json').is_file():
            api_key = legacy_developer_key_file.read_text(encoding='utf-8')

            app.credentials_file.write_text(json.dumps({'api_key': api_key}), encoding='utf-8')
            app.set_api_key(api_key)
            legacy_developer_key_file.unlink()
        else:
            api_key = click.prompt(
                f"""Welcome to Chai Guanaco 🚀!
By logging in, we will create a file under {app.data_dir}.
Please enter your API key""",
                hide_input=True,
            )

            app.credentials_file.write_text(json.dumps({'api_key': api_key}), encoding='utf-8')
            app.set_api_key(api_key)

    if not ctx.invoked_subcommand:
        print(ctx.get_help())
        return

    # Persist app data for sub-commands
    ctx.obj = app


chai_guanaco.add_command(leaderboard)
chai_guanaco.add_command(login)
chai_guanaco.add_command(logout)


def main():  # no cov
    try:
        return chai_guanaco(prog_name='chai_guanaco', windows_expand_args=False)
    except Exception:
        from rich.console import Console

        console = Console()
        console.print_exception(suppress=[click])
        return 1
