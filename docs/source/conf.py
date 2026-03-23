import sys
from pathlib import Path

# Add source directory so Sphinx can import xpm_torch without installing
# the full project (which would pull in torch, lightning, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

project = "xpm-torch"
copyright = "2024, Benjamin Piwowarski"
author = "Benjamin Piwowarski"

extensions = [
    # Experimaestro extension for documenting Config/Param classes
    "experimaestro.sphinx",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    # Link to other documentations
    "sphinx.ext.intersphinx",
    # Auto-link identifiers in code blocks to API docs
    "sphinx_codeautolink",
]

intersphinx_mapping = {
    "experimaestro": (
        "https://experimaestro-python.readthedocs.io/en/latest/",
        None,
    ),
}

# Mock heavy dependencies that are not needed to build the docs.
# experimaestro is NOT mocked — it's installed in the docs group
# and needed for experimaestro.sphinx and Config/Param introspection.
autodoc_mock_imports = [
    "huggingface_hub",
    "lightning",
    "numpy",
    "safetensors",
    "tensorboard",
    "torch",
    "torchdata",
]

templates_path = ["_templates"]
exclude_patterns = []
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
source_suffix = ".rst"
