# Mosaic

Paper website for **Mosaic: Multi-Session Agentic Reinforcement Learning**.

## Local preview

```sh
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://127.0.0.1:8000/.

## Editing

- `index.html`: article, diagrams, results and BibTeX.
- `static/css/style.css`: styles and layout; update its version query in `index.html` after changes.
- `static/js/`: animations and metadata loading.
- `static/data/paper.json`: authors, affiliations and paper/code links.

## Publishing

In GitHub **Settings → Pages**, select **GitHub Actions**. The included workflow deploys changes pushed to `main` or `master`; it can also be run manually.

Only `index.html`, `static/` and `.nojekyll` are packaged into `_site/` for deployment.

## Checks

Requires Node.js 20+, Python 3 and local Chrome (or a Playwright Chromium installation).

```sh
npm ci
npm test -- --workers=1
npm run package:pages
```
