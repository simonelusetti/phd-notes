# quartz-base

Personal Quartz engine template by [@simonelusetti](https://github.com/simonelusetti).

This repo contains the full [Quartz v4](https://quartz.jzhao.xyz) engine with no site-specific content.
It is used as the **build engine** for all my Quartz-powered sites.

## How it works

Content repos (e.g. `notes-quartz`, `rpg-quartz`) contain only:
- `content/` — the markdown notes
- `quartz.config.ts` — site-specific config (title, baseUrl, theme)
- `quartz.layout.ts` — site-specific layout (optional override)
- `.github/workflows/deploy.yml` — CI workflow that pulls this engine and builds

At deploy time, GitHub Actions checks out this repo as the engine, overlays the
content repo's files, runs `npm ci && npx quartz build`, and deploys to GitHub Pages.

## Creating a new site

1. Create a new repo on GitHub (use this repo as template if marked as one)
2. Add only: `content/index.md`, `quartz.config.ts`, `quartz.layout.ts`, `.github/workflows/deploy.yml`
3. In `quartz.config.ts`, set `pageTitle` and `baseUrl` (`YOUR_USERNAME.github.io/YOUR_REPO`)
4. Enable GitHub Pages (Settings → Pages → Source: GitHub Actions)
5. Add an alias to `~/.zshrc` following the pattern in the other sites

## Updating Quartz

To pull upstream Quartz updates into this engine:

```bash
git remote add upstream https://github.com/jackyzha0/quartz.git  # first time only
git fetch upstream
git merge upstream/v4
```

Then push — all content repos will pick up the update on their next deploy.
