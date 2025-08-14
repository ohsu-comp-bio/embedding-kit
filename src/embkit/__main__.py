import click
from .commands import model

# ---- Top-level CLI ----
@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Embedding Kit CLI.

    Use 'embkit help [COMMAND]' for details on a specific command.
    """
    # If no subcommand provided, show the top-level help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register our commands here
cli.add_command(model)

# ---- 'help' command ----
@cli.command("help", context_settings=dict(ignore_unknown_options=True))
@click.argument("path", nargs=-1)
@click.pass_context
def help_cmd(ctx, path):
    """
    Show help for the CLI or a specific command.

    Examples:
      embkit help
      embkit help model
      embkit help model train
    """
    # No args -> show top-level help plus a neat command list summary
    if not path:
        click.echo(ctx.parent.get_help())
        click.echo("\nCommands:")
        # Sorted, with 1-line summaries
        for name in sorted(cli.commands):
            cmd = cli.commands[name]
            summary = (cmd.help or cmd.short_help or "").strip().splitlines()[0]
            click.echo(f"  {name:15s} {summary}")
        return

    # Resolve dotted/space-separated path to a nested command
    cmd = cli
    info_name = []
    parent = ctx.parent  # start from top-level context
    for part in path:
        if not hasattr(cmd, "get_command"):
            raise click.UsageError(f"'{ ' '.join(info_name) }' has no subcommands.")
        nxt_cmd = cmd.get_command(parent, part)
        if nxt_cmd is None:
            raise click.UsageError(f"Unknown command: {' '.join(path)}")
        info_name.append(part)
        cmd = nxt_cmd
        parent = click.Context(cmd, info_name=part, parent=parent)

    # Print help for the resolved command
    click.echo(cmd.get_help(parent))


if __name__ == "__main__":
    cli()