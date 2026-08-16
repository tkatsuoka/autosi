# Sphinx configuration for the AutoSI documentation.
# Build locally with:  uv run --group docs sphinx-build docs docs/_build/html

import os
import sys

# Make the autosi package importable for autodoc
sys.path.insert(0, os.path.abspath(".."))

project = "AutoSI"
# Anonymized while the accompanying paper is under review
author = "AutoSI developers"
copyright = f"2026, {author}"
release = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",      # NumPy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# Generate stub pages for every entry listed in autosummary tables
autosummary_generate = True

# NumPy style only (Google style off) to keep parsing strict
napoleon_google_docstring = False
napoleon_numpy_docstring = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_static_path = ["_static"]
# Animated rational-function background (active only on the landing page,
# which carries the #autosi-bg marker)
html_css_files = ["landing-bg.css"]
html_js_files = ["landing-bg.js"]

# Footer: removed entirely (the theme's site-foot partial is overridden as
# empty in _templates/; leaving only the credit out kept a bare colored bar)
html_show_copyright = False
html_show_sphinx = False

html_theme = "shibuya"
html_title = "AutoSI"
html_theme_options = {
    "accent_color": "indigo",
    # Dark code blocks even in light mode
    "dark_code": True,
    # Header navigation
    "nav_links": [
        {"title": "Quickstart", "url": "quickstart"},
        {"title": "Examples", "url": "examples"},
        {"title": "API", "url": "api"},
    ],
    # No outbound AI-service links while the paper is under anonymous review
    "show_ai_links": False,
    # Repository/social links are added after the anonymous review period
}
