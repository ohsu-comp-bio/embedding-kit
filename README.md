
# Embedding Kit

Methods for data normalization, embedding, synthesis and transformation


## Installation
```
pip install embkit
```

# Training a model
```
embkit model train-vae ./experiments/tcga/tumor.normalized.tsv --epochs 120
```

## Development

To install the library locally use:  
```pip install -e .```  
```python setup.py build```  
```python setup.py install```  

### To run tests use:
```bash
coverage run --source=embkit -m unittest discover -s tests
```

To generate a coverage report use:
```bash
coverage html
```

To open the coverage report in a browser, run:
#### MacOS:
```bash
open htmlcov/index.html
```

#### Linux:
```bash
xdg-open htmlcov/index.html
```

#### Windows:
```bash
start htmlcov\index.html
```

