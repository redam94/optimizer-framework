# Atlas Documentation

This directory contains the documentation source files for Atlas.

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -e ".[docs]"
# or
pip install -r docs/requirements-docs.txt
```

### Building HTML Documentation

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`.

### Live Development Server

For development, use the auto-build server:

```bash
cd docs
make livehtml
```

This will start a server at http://localhost:8000 that auto-rebuilds when you change files.

### Building Other Formats

```bash
# PDF (requires LaTeX)
make latexpdf

# ePub
make epub

# Single page HTML
make singlehtml
```

### Cleaning Build Files

```bash
make clean
```

## Documentation Structure

```
docs/
├── source/
│   ├── _static/          # Static files (CSS, images, etc.)
│   ├── _templates/       # Custom templates
│   ├── api/              # API reference
│   ├── examples/         # Example notebooks and scripts
│   ├── guides/           # User guides
│   ├── conf.py           # Sphinx configuration
│   └── index.rst         # Main documentation index
├── build/                # Build output (git ignored)
├── Makefile              # Unix/Linux build commands
├── make.bat              # Windows build commands
└── requirements-docs.txt # Documentation dependencies
```

## Writing Documentation

### Adding New Pages

1. Create a new `.md` or `.rst` file in the appropriate directory
2. Add it to the relevant `toctree` in `index.rst` or section index
3. Follow the existing style and formatting

### Using Markdown

We support both reStructuredText and Markdown. For Markdown files:

```markdown
# Page Title

## Section

Content with **bold** and *italic* text.

```python
# Code blocks with syntax highlighting
def example():
    return "Hello, World!"
```

### API Documentation

API docs are auto-generated from docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description with more details about what
    the function does and how to use it.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is empty
        TypeError: When param2 is not an integer
        
    Example:
        >>> my_function("test", 42)
        True
    """
```

### Adding Examples

1. Create example scripts in `source/examples/`
2. Use notebook format (`.ipynb`) for interactive examples
3. Include in documentation with:

```rst
.. toctree::
   :maxdepth: 1
   
   examples/my_example
```

## Style Guide

- Use clear, concise language
- Include code examples for all features
- Add type hints to all code examples
- Use present tense
- Keep line length under 88 characters
- One sentence per line in source files

## Deployment

Documentation is automatically built and deployed to Read the Docs when pushing to main branch.

Manual deployment:

```bash
# Build for deployment
make clean html

# Check for broken links
make linkcheck

# Deploy (if not using Read the Docs)
rsync -av build/html/ user@server:/path/to/docs/
```
