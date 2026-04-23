# Configuration file for the Sphinx documentation builder.
import importlib.metadata

project = "HiMGA"
author = "colehank"
copyright = "2026, colehank"
release = importlib.metadata.version("himga")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "tasklist",
]
myst_heading_anchors = 3

html_theme = "pydata_sphinx_theme"
html_title = "HiMGA"

html_theme_options = {
    "github_url": "https://github.com/colehank/HiMGA",
    "navbar_align": "left",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}

html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

html_static_path = ["_static"]
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
