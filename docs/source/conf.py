# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath('../../src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Atlas'
copyright = f'{datetime.now().year}, Matthew Reda'
author = 'Matthew Reda'
release = '0.0.1'
version = '0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'myst_parser',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'nbsphinx',
    'sphinx_tabs.tabs',
    'sphinx_design', 
]

# Add support for Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'css/custom.css',
]

# The name of an image file (relative to this directory) to place at the top
# of the sidebar
html_logo = '_static/logo.png'

# The name of an image file (relative to this directory) to use as a favicon
html_favicon = '_static/favicon.ico'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{algorithm}
\usepackage{algorithmic}
''',
}

# Grouping the document tree into LaTeX files
latex_documents = [
    (master_doc, 'OptimizerFramework.tex', 'Atlas Documentation',
     'Atlas Team', 'manual'),
]

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'xarray': ('https://docs.xarray.dev/en/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# MyST parser configuration
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# nbsphinx configuration
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# Todo extension
todo_include_todos = True

# -- Custom setup ------------------------------------------------------------

def setup(app):
    app.add_css_file('css/custom.css')
    
    # Add custom directives or roles if needed
    from docutils.parsers.rst import directives
    
    # Example: Add a custom directive for optimization examples
    from docutils import nodes
    from docutils.parsers.rst import Directive
    
    class OptimizationExample(Directive):
        has_content = True
        required_arguments = 1
        optional_arguments = 0
        final_argument_whitespace = True
        option_spec = {
            'channels': directives.unchanged,
            'objective': directives.unchanged,
        }
        
        def run(self):
            title = self.arguments[0]
            self.options.get('channels', 'tv, digital, radio')
            self.options.get('objective', 'revenue')
            
            para = nodes.paragraph()
            para += nodes.strong(text=f"Optimization Example: {title}")
            
            content = nodes.container()
            content += para
            
            # Process the content
            self.state.nested_parse(self.content, self.content_offset, content)
            
            return [content]
    
    app.add_directive('optimization-example', OptimizationExample)