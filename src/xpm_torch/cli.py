"""xpm-torch CLI tools."""

import click
from pathlib import Path


@click.group()
def main():
    """xpm-torch CLI tools"""


@main.command()
@click.option(
    "--workspace",
    type=str,
    help="Experimaestro workspace ID (interactive if not given)",
)
@click.option(
    "--workdir",
    type=click.Path(exists=True),
    help="Direct workspace path (overrides --workspace)",
)
@click.option(
    "--experiment",
    type=str,
    help="Experiment name (interactive if not given)",
)
@click.option(
    "--model-key",
    type=str,
    help="Select specific model by key (interactive if not given)",
)
@click.option(
    "--repo-id",
    type=str,
    help="HuggingFace Hub repo ID (interactive if not given)",
)
@click.option(
    "--save-dir",
    type=click.Path(),
    help="Save to local directory instead of/in addition to HF Hub",
)
@click.option("--private", is_flag=True, help="Make the HF Hub repo private")
@click.option(
    "--key-override", type=str, help="Override the model key name"
)
def upload_hfhub(
    workspace, workdir, experiment, model_key, repo_id, save_dir, private, key_override
):
    """Export a trained model, optionally uploading to HuggingFace Hub.

    Reads TrainingResults from the experiment's saved data.
    Interactive prompts fill in any missing options.
    """
    from experimaestro.huggingface import ExperimaestroHFHub
    from experimaestro.settings import find_workspace, get_settings

    # Workspace selection
    if workdir:
        workspace_path = Path(workdir)
    elif workspace:
        ws = find_workspace(workspace=workspace)
        workspace_path = ws.path
    else:
        settings = get_settings()
        if not settings.workspaces:
            raise click.ClickException(
                "No workspaces configured in experimaestro settings"
            )
        click.echo("Available workspaces:")
        for i, ws in enumerate(settings.workspaces):
            click.echo(f"  [{i}] {ws.id} ({ws.path})")
        choice = click.prompt("Select workspace", type=int)
        workspace_path = Path(settings.workspaces[choice].path)

    # Experiment selection
    if experiment is None:
        exp_dir = workspace_path / "experiments"
        if not exp_dir.exists():
            raise click.ClickException(f"No experiments found in {workspace_path}")
        experiments = sorted(
            [d.name for d in exp_dir.iterdir() if d.is_dir()],
        )
        if not experiments:
            raise click.ClickException(f"No experiments found in {exp_dir}")
        click.echo("Available experiments:")
        for i, exp in enumerate(experiments):
            click.echo(f"  [{i}] {exp}")
        choice = click.prompt("Select experiment", type=int)
        experiment = experiments[choice]

    # Load results
    results_path = (
        workspace_path
        / "experiments"
        / experiment
        / "current"
        / "data"
        / "xpm-torch-models"
    )
    if not results_path.exists():
        raise click.ClickException(
            f"No saved results found at {results_path}. "
            "Make sure the experiment saved TrainingResults."
        )

    from experimaestro.core.objects import ConfigInformation

    results = ConfigInformation.deserialize(
        lambda p: results_path / p, as_instance=True
    )

    # Model selection
    if model_key is None:
        keys = list(results.models.keys())
        if not keys:
            raise click.ClickException("No models found in results")
        click.echo("Available models:")
        for i, key in enumerate(keys):
            click.echo(f"  [{i}] {key}")
        choice = click.prompt("Select model", type=int)
        model_key = keys[choice]

    model = results.models[model_key]
    key = key_override or model_key

    # Save to local directory
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"Saving {key} to {save_dir}")
        ExperimaestroHFHub(model).save_pretrained(save_dir)

    # Upload to HF Hub
    if repo_id is None and not save_dir:
        if click.confirm("Publish to HuggingFace Hub?"):
            repo_id = click.prompt("HF Hub repo ID (e.g. user/model-name)")
            private = private or click.confirm("Make repo private?", default=False)

    if repo_id:
        click.echo(f"Uploading {key} to {repo_id} (private={private})")
        ExperimaestroHFHub(model).push_to_hub(repo_id=repo_id, private=private)
    elif not save_dir:
        click.echo("Nothing to do (no --save-dir or HF Hub upload).")
