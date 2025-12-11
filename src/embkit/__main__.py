import click
import logging

from .commands import model, matrix, cbio_cmd, protein, datasets, align

@click.group(invoke_without_command=True)
@click.option('--verbose', is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, verbose):
    """Embedding Kit CLI.

    Use 'embkit help [COMMAND]' for details on a specific command.
    """
    # Set up logging based on the verbose flag
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    ctx.ensure_object(dict)
    ctx.obj["VERBOSE"] = verbose

    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Register our commands here
cli.add_command(model)
cli.add_command(matrix)
cli.add_command(cbio_cmd)
cli.add_command(protein)
cli.add_command(datasets)
cli.add_command(align)


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
            click.echo(cli.get_help(ctx)) # pragma: no cover
        nxt_cmd = cmd.get_command(parent, part)
        if nxt_cmd is None:
            raise click.UsageError(f"Unknown command: {' '.join(path)}")
        info_name.append(part)
        cmd = nxt_cmd
        parent = click.Context(cmd, info_name=part, parent=parent)

    # Print help for the resolved command
    click.echo(cmd.get_help(parent))


cli_main = cli

if __name__ == "__main__": # pragma: no cover
    cli()