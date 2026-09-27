# Development

## Set up

```console
uv sync --group dev
```

## Validate

```console
uv run ruff check .
uv run black --check .
uv run pyrefly check
uv run pytest
uv run mkdocs build --strict
uv build
```

Documentation is public-product documentation. Build records, source-tree
layouts, implementation sequencing, and other framework-maintainer notes are
intentionally not published in the MkDocs site.
