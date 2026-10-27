# Diagrams

Mermaid sources for the diagrams in the documentation and the README. Each
`<name>.mmd` file is rendered twice into `docs/assets/diagrams/`:

- `<name>-light.png` on a white background, for Material's light scheme.
- `<name>-dark.png` on a transparent background, for Material's dark (slate) scheme.

Docs pages show both variants and let Material pick one:

```markdown
![Alt text](../assets/diagrams/<name>-light.png#only-light)
![Alt text](../assets/diagrams/<name>-dark.png#only-dark)
```

The README is rendered on PyPI and GitHub, so it links the light PNG with an
absolute URL:
`https://raw.githubusercontent.com/smuniharish/agloom/master/docs/assets/diagrams/<name>-light.png`.

## Render and check

You need Node.js 22.13 or later (for `npx`) and Python 3.12 or later. The script
uses only the standard library.

```bash
python scripts/render_diagrams.py                     # render every diagram
python scripts/render_diagrams.py --only topology-react architecture
python scripts/render_diagrams.py --check             # verify without rendering
```

Rendering runs the pinned `@mermaid-js/mermaid-cli@12.0.0` at scale 2. A full
render also removes files in `docs/assets/diagrams/` that no source produces.
Each run rewrites `manifest.json`, which records the SHA-256 of every source and
config. `--check` fails when:

- a source or config changed since the last render;
- a PNG is missing or is not a PNG;
- `docs/assets/diagrams/` contains a file that no source produces.

On first use, Puppeteer downloads its own browser. To use an installed Chrome
or Edge instead, set `PUPPETEER_EXECUTABLE_PATH` to the browser executable. If
the download fails behind a corporate TLS proxy, set `NODE_USE_SYSTEM_CA=1`.

## Style

`mermaid-light.json` and `mermaid-dark.json` hold the whole visual design: fonts,
spacing, the palette (Material's indigo, matching `mkdocs.yml`), and the CSS for
the semantic classes below. Sources assign classes only and never set colors, so
one source renders correctly in both themes.

| Concept | Shape | Class |
| --- | --- | --- |
| Agloom component or step | rounded box `id("...")` | none |
| Decision point | rhombus `id{"..."}` | none |
| Public API call or result | stadium `id(["..."])` | `api` |
| Model or sub-agent work | rounded box `id("...")` | `model` |
| Structured-output decision | hexagon `id{{"..."}}` | `model` |
| One node per worker, stage, or step | `id@{ shape: procs, label: "..." }` | `model` |
| Tools and capabilities | subroutine `id[["..."]]` | `ext` |
| Stores (checkpointer, agentbarrier, board) | cylinder `id[("...")]` | `ext` |
| Application, outside system, or integration | rounded box | `ext` |
| Waiting for a human | rounded box | `human` |
| Success outcome | rounded box | `ok` |
| Failure or error | rounded box | `bad` |
| Explanatory note | rectangle `id["..."]` | `note` |
| Left-aligned list | rectangle `id["..."]` | `list` |
| Graph start and end | `@{ shape: sm-circ }` and `@{ shape: fr-circ }` | none |

Guidelines:

- Show public API names and concepts only: never private modules, internal
  helpers, or private configuration keys.
- Use the real graph node names in topology diagrams, such as
  `worker_<i>_<name>`, so they match `interrupt_before` and `interrupt_after`.
- Keep labels short and use `<small>` for secondary text. Keep diagrams at most
  about 900 px wide (the script prints each size and flags wider ones), so text
  stays legible in the docs column.
- Prefer top-down layouts. Use left-to-right only for short chains.
- After editing, render, then look at both variants before you commit.
