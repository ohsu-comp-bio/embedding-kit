

## Doc gen

### Install dependencies
```
pip install mkdocs
pip install mkdocstrings mkdocstrings-python
pip install mkdocs-click
```

### View docs
```
mkdocs serve
```

### Build docs
```
mkdocs build
```

## Development

To install the library locally:

```bash
pip install -e .
```

### Running tests

```bash
coverage run --source=embkit -m unittest discover -s tests
```

### Coverage report

To generate an HTML coverage report:

```bash
coverage html
```

To open the report in a browser:

**macOS**

```bash
open htmlcov/index.html
```

**Linux**

```bash
xdg-open htmlcov/index.html
```

**Windows**

```bash
start htmlcov\index.html
```
