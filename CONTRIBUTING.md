# Contributing to Agloom

Thank you for helping improve Agloom. This guide covers the development setup,
the checks every change must pass, the documentation and diagrams, and the
release process for maintainers.

## Ground rules

- **One version, one API.** Agloom does not keep compatibility shims or
  parallel APIs. When behavior changes, change it everywhere: code, tests,
  documentation, examples, the Agent Skill, and the changelog.
- **Build on the ecosystem.** Prefer what LangChain, LangGraph, and the
  integrated packages already provide over new abstractions.
- **No secrets in the repository.** Credentials come only from environment
  variables; `.env` is git-ignored.
- **Small, focused pull requests** with tests and a changelog entry in an
  `Unreleased` section at the top of [CHANGELOG.md](CHANGELOG.md).

Report security issues privately as described in [SECURITY.md](SECURITY.md).

## Development setup

Agloom uses [uv](https://docs.astral.sh/uv/) and supports Python 3.12, 3.13,
and 3.14.

```bash
git clone https://github.com/smuniharish/agloom.git
cd agloom
uv sync            # installs the dev, docs, and examples dependency groups
```

If your network uses a custom certificate authority, add `--system-certs` to
uv commands. To test another Python version, run
`uv sync --python 3.14` (or `3.13`).

## Checks

Run these before opening a pull request.

| Check | Command |
|---|---|
| Lint | `uv run ruff check .` |
| Format | `uv run ruff format .` (add `--check` to verify without changing files) |
| Types | `uv run pyrefly check` |
| Tests | `uv run pytest` |
| Coverage | `uv run pytest --cov` (fails below 100% branch coverage) |
| Documentation | `uv run mkdocs build --strict` |
| Diagrams | `python scripts/render_diagrams.py --check` |

### Tests

- Unit tests use the scripted chat model in `tests/helpers.py`
  (`ScriptedChatModel`, `tool_call`, `analysis`) and never call a real model.
- Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/).
  Two settings profiles are registered: `dev` (the default, fast) and `ci`
  (more examples). Select one with `HYPOTHESIS_PROFILE`:

    ```bash
    HYPOTHESIS_PROFILE=ci uv run pytest
    ```

    In PowerShell: `$env:HYPOTHESIS_PROFILE = "ci"; uv run pytest`.

- Tests that need network access or a live model carry the `network` marker.
  Skip them with `uv run pytest -m "not network"`.
- Run tests with `LANGGRAPH_STRICT_MSGPACK=true` to catch checkpoint
  serialization problems early.

### Examples

The scripts in `examples/` run against a live, OpenAI-compatible model. Copy
`.env.example` to `.env`, set `EXPLABS_API_KEY`, `EXPLABS_BASE_URL`, and
`EXPLABS_MODEL`, then run an example with `uv run python examples/01_direct.py`.
Every example asserts facts about its result; keep it that way when you add
one. Each reference application under `examples/` has its own `pyproject.toml`,
lockfile, tests, and README.

## Documentation

The documentation is built with MkDocs and Material and published on
[Read the Docs](https://agloom.readthedocs.io/).

```bash
uv run mkdocs serve           # live preview at http://127.0.0.1:8000
uv run mkdocs build --strict  # the build CI and Read the Docs run; warnings fail it
```

- Pages live in `docs/`; the navigation is in `mkdocs.yml`.
- The API reference is generated from docstrings (Google style) of the
  public names exported by `agloom`.
- Prefer including tested example files with snippets
  (`--8<-- "examples/<file>.py"`) over long inline code, and run every inline
  snippet a reader might copy.
- Link to API objects with cross-references such as
  `` [`PolicyManager`][agloom.PolicyManager] ``.

## Diagrams

Diagram sources are Mermaid files in `diagrams/<name>.mmd`, rendered to
`docs/assets/diagrams/<name>-light.png` and `<name>-dark.png`. Pages embed both
variants:

```markdown
![Architecture](../assets/diagrams/architecture-light.png#only-light)
![Architecture](../assets/diagrams/architecture-dark.png#only-dark)
```

Rendering needs Node.js 22.13 or later (it runs the pinned
`@mermaid-js/mermaid-cli` through `npx`):

```bash
python scripts/render_diagrams.py                      # render every diagram
python scripts/render_diagrams.py --only architecture  # render selected diagrams
python scripts/render_diagrams.py --check              # verify renders are current
```

Commit the sources and the rendered PNGs together.

## Agent Skill

The Agent Skill in `agloom-skills/skills/agloom` teaches coding agents to use
Agloom. Update it in the same pull request whenever public behavior changes.

## Release process (maintainers)

Releases are immutable, follow Semantic Versioning, and are built from a
signed tag. No PyPI token is stored in GitHub: publishing uses OIDC trusted
publishing.

### One-time configuration

1. Create TestPyPI and PyPI trusted publishers for this repository and the
   `publish.yml` workflow, using the `testpypi` and `pypi` environments.
2. Create both GitHub environments with required reviewers.
3. Set the repository variable `PYPI_PUBLISH_ENABLED` to `true` only after
   trusted publishing and required reviewers are configured.
4. Enable private vulnerability reporting and artifact attestations.

### Publishing a release

1. Make sure CI, the dependency-compatibility checks, and the supply-chain
   workflow pass on the default branch.
2. Rename the `Unreleased` section of `CHANGELOG.md` to the new version and
   date, and set the same version in `pyproject.toml`.
3. Run `uv lock --check`, the full checks above, and `uv build`; inspect the
   wheel and the source archive.
4. Merge, then create and push a signed tag `vX.Y.Z` that matches the package
   version exactly. The supply-chain workflow audits the lockfiles, builds the
   distributions, generates a CycloneDX SBOM, and records provenance.
5. Run the **Publish release** workflow with the tag and `none` to validate the
   candidate, then with `testpypi`, and verify an installation from TestPyPI.
6. Run it with `pypi`; a different authorized reviewer approves the protected
   environment.
7. Verify the files and provenance on PyPI, then publish GitHub release notes
   from the changelog section, and check that Read the Docs built the release.

Never re-tag or replace published artifacts; fix a release with a new patch
version. To halt publication, set `PYPI_PUBLISH_ENABLED` to `false` before
rejecting pending deployments.
