# SPEAR Documentation

We use [MkDocs](https://www.mkdocs.org/) to generate the documentation with the [MkDocs Material theme](https://squidfunk.github.io/mkdocs-material/).

## Development

From the root directory:

1. Install SPEAR locally

```bash
CCACHE_NOHASHDIR="true" uv pip install -e ".[dev,docs]"
```

note that `docs` is the extra requirement for the documentation.


2. To build the documentation, run:

```bash
mkdocs serve
```

3. (optional) Deploy to GitHub Pages

```bash
mkdocs gh-deploy
```

This will deploy the documentation to the `gh-pages` branch of the repository.


### Hooks

We are using the [hooks.py](hooks.py) for additional modifications. MkDocs for instance cannot detect files that are not in the same directory as an `__init__.py` (as described [here](https://stackoverflow.com/questions/75232397/mkdocs-unable-to-find-modules)) so we are automatically creating and deleting such files with our script

---

Note: adapted from [RL4CO](https://github.com/ai4co/rl4co)