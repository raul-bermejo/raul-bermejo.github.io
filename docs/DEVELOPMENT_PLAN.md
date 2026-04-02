# Development Plan: raul-bermejo.github.io

## 1. Project Overview

### What is this?

A personal website and portfolio for Raul Bermejo, built with [Jekyll](https://jekyllrb.com/) using the [Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy). The site hosts blog posts (primarily on ML engineering, MLOps and data science), an about page, a CV viewer, and standard category/tag navigation.

### Tech stack

| Component       | Current state                                                       |
| --------------- | ------------------------------------------------------------------- |
| Static site gen | Jekyll 4.4.1                                                       |
| Theme           | jekyll-theme-chirpy v5.6.1 (gem), local gemspec pinned at v5.2.1   |
| Ruby            | `.ruby-version` set to `system` (not pinned)                       |
| CI/CD           | GitHub Actions (`pages.yml`) deploying to GitHub Pages              |
| JS build        | gulp-based pipeline (`gulpfile.js/`, `package.json`)                |
| Hosting         | GitHub Pages (free, public repo) at `raul-bermejo.github.io`       |
| Custom domain   | None configured                                                     |

### Repository

- **Origin:** `https://github.com/raul-bermejo/raul-bermejo.github.io.git`
- **Upstream theme:** `https://github.com/cotes2020/jekyll-theme-chirpy`
- **Default branch:** `main`

---

## 2. Requirements and Assumptions

### Functional requirements

- FR1: Site builds and serves correctly on both macOS and Linux.
- FR2: CI/CD pipeline builds the site on every PR to `main`, blocking merge on failure.
- FR3: CI/CD pipeline deploys to GitHub Pages on push to `main`.
- FR4: Local development setup is straightforward (single command to install deps, single command to serve).
- FR5: Ability to sync with the upstream Chirpy theme when needed.
- FR6: Support for a custom domain (optional, explorable).
- FR7: Site can be hosted on a managed service with limited resources as an alternative to GitHub Pages.

### Non-functional requirements

- NFR1: Build time under 60 seconds for the current content volume.
- NFR2: Lighthouse performance score above 90.
- NFR3: No hardcoded secrets or credentials in the repository.
- NFR4: Dependencies are kept up to date with automated tooling.
- NFR5: The site works as a Progressive Web App (PWA is already enabled in config).

### Assumptions

- The repository remains public (GitHub Pages free tier requires this for user sites, unless on a paid GitHub plan).
- The primary audience is technical users comfortable with the command line.
- Content is authored in Markdown and committed via Git.
- Ruby and Bundler are available on the development machine (or can be installed via a version manager).
- No server-side logic is needed; the site is fully static.

---

## 3. Current State Audit

### What works

- The site builds and deploys successfully via `pages.yml` on push to `main`.
- PR builds run (build-only, no deploy) thanks to the `pull_request` trigger and conditional deploy steps in `pages.yml`.
- Content structure is clean: 12 blog posts, about page, CV page, categories and tags.
- PWA is enabled.
- Sass compilation with compressed output.
- HTML compression configured for production.

### What needs attention

| Issue | Detail | Severity |
| ----- | ------ | -------- |
| Stale workflow | `pages-deploy.yml` targets `master` with Ruby 2.7 and `actions/checkout@v2`. Dead code. | Low (no harm, but confusing) |
| Ruby version not pinned | `.ruby-version` contains `system`, making builds non-reproducible. | Medium |
| Outdated local gemspec | `jekyll-theme-chirpy.gemspec` says v5.2.1 but the gem resolves to v5.6.1. The gemspec is from the original fork and is not used by Bundler (the Gemfile controls this), but it is misleading. | Low |
| No Makefile | Local dev relies on `tools/run.sh`. No single entrypoint for lint, build, serve, test. | Medium |
| Missing .gitignore entries | `.DS_Store`, `node_modules/`, `Gemfile.lock` (optional), `*.gem` not all covered. `.DS_Store` files are committed. | Low |
| Upstream authors in `_data/authors.yml` | Still contains `cotes` and `sille_bille` example entries instead of Raul's info. | Low |
| No link checking | Broken links in posts or pages are not caught by CI. | Medium |
| Theme version gap | Current v5.x is several major versions behind upstream (likely v7.x+). Upgrade brings new features, bug fixes, and security patches. | Medium (deferred) |
| No custom domain | Site is at `raul-bermejo.github.io` with no CNAME. | N/A (by choice, for now) |
| No Docker setup | No containerised dev environment for consistent builds. | Low |
| README is upstream boilerplate | Does not describe Raul's site or how to develop it. | Low |
| `sandbox_editor.md` in `_posts/` | Dev artifact, gitignored but present on disk. | Trivial |

---

## 4. Phased Development Checklist

### Phase 0: Foundation (local dev and repo hygiene)

Goal: Make the repository clean, reproducible, and easy to work with locally.

- [ ] **Pin Ruby version** - set `.ruby-version` to a specific version (e.g. `3.3.x`) matching the CI workflow.
- [ ] **Create a Makefile** with targets:
  - `make install` - runs `bundle install`
  - `make serve` - runs `bundle exec jekyll serve -H 0.0.0.0 --livereload`
  - `make build` - runs `bundle exec jekyll build`
  - `make clean` - removes `_site/` and `.jekyll-cache/`
- [ ] **Update `.gitignore`** - add `.DS_Store`, `node_modules/`, `*.gem`, `.bundle/`, `vendor/`, `_site/`, `.sass-cache/`.
- [ ] **Remove committed `.DS_Store` files** from the repository.
- [ ] **Update `_data/authors.yml`** - replace upstream example authors with Raul's details.
- [ ] **Rewrite `README.md`** - describe the project, prerequisites, setup instructions (referencing the Makefile), and link to this development plan.
- [ ] **Clean up `jekyll-theme-chirpy.gemspec`** - either delete it (not needed when using the gem via Gemfile) or update the version to match reality.
- [ ] **Remove `tools/init.sh`** - this is an upstream initialisation script that has already been run and is no longer useful.
- [ ] **Review `tools/deploy.sh` and `tools/release.sh`** - likely dead code given the GitHub Actions workflow handles deployment. Remove if unused.

### Phase 1: CI/CD Hardening

Goal: Ensure every PR is validated before merge, and deployments are reliable.

- [ ] **Remove `.github/workflows/pages-deploy.yml`** - targets `master` branch with Ruby 2.7. Dead workflow.
- [ ] **Update `pages.yml` action versions** - verify `actions/checkout@v4`, `ruby/setup-ruby@v1`, `actions/configure-pages@v5`, `actions/upload-pages-artifact@v3`, `actions/deploy-pages@v4` are the latest. (Currently looks correct as of early 2025.)
- [ ] **Add HTML-Proofer to CI** - run `htmlproofer` on the built `_site/` directory to catch broken links, missing alt text, and invalid HTML. Add as a step after the Jekyll build in the PR-only path.
  ```yaml
  - name: Check links and HTML
    if: github.event_name == 'pull_request'
    run: bundle exec htmlproofer _site --disable-external --allow-hash-href
  ```
  Add `html-proofer` to the Gemfile.
- [ ] **Add build status badge** to README.
- [ ] **Consider branch protection rules** on `main` - require the build job to pass before merge.

### Phase 2: Theme Upgrade (Chirpy v5.x to latest)

Goal: Bring the theme up to date for new features, security patches, and upstream compatibility.

This phase is intentionally separate because Chirpy v6.x and v7.x introduced breaking changes (new directory structure, different JS bundling, updated config keys).

- [ ] **Research the current latest Chirpy version** and read the migration guides:
  - [v5 to v6 migration](https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Upgrade-Guide)
  - [v6 to v7 migration](https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Upgrade-Guide)
- [ ] **Create a feature branch** (`feat/chirpy-upgrade`) for the upgrade.
- [ ] **Update `Gemfile`** to target the new version.
- [ ] **Reconcile `_config.yml`** with the new theme's default config (new keys, renamed keys, removed keys).
- [ ] **Review `_includes/`, `_layouts/`, `_sass/`** for local overrides that may conflict with the new theme. The current repo has extensive local copies of theme files - many may be deletable if the gem provides them.
- [ ] **Update `_data/` files** to match the new theme's expected format.
- [ ] **Update or remove `gulpfile.js/` and `package.json`** - newer Chirpy versions may use a different JS build approach.
- [ ] **Test locally** - build, serve, check all pages, verify dark/light mode, check mobile layout.
- [ ] **Test in CI** - push to the feature branch and verify the build passes.
- [ ] **Add upstream remote** for future syncs:
  ```bash
  git remote add upstream https://github.com/cotes2020/jekyll-theme-chirpy.git
  ```

### Phase 3: Custom Domain and Hosting Options

Goal: Document and optionally implement custom domain support, with cost analysis for different hosting approaches.

#### Option A: GitHub Pages with custom domain (recommended starting point)

| Item | Cost | Notes |
| ---- | ---- | ----- |
| GitHub Pages hosting | Free | Requires public repo for free tier (or GitHub Pro at ~US$4/mo for private repos) |
| Domain registration | ~US$10-15/yr | Varies by registrar and TLD. Popular registrars: Cloudflare Registrar (at-cost pricing, typically cheapest), Namecheap, Porkbun, Google Domains (now Squarespace) |
| DNS management | Free | Most registrars include DNS. Cloudflare free tier is excellent |
| SSL/TLS certificate | Free | GitHub Pages provides automatic HTTPS via Let's Encrypt |
| **Total** | **~US$10-15/yr** | |

Steps:
- [ ] Purchase a domain (e.g. `raulbermejo.com`, `raulbermejo.dev`, `raul.engineer`).
- [ ] Configure DNS: add an `A` record pointing to GitHub Pages IPs and a `CNAME` for `www`.
- [ ] Add a `CNAME` file to the repo root containing the custom domain.
- [ ] Update `_config.yml` `url` field to `https://yourdomain.com`.
- [ ] Enable "Enforce HTTPS" in the GitHub Pages settings.
- [ ] Verify the site loads on the custom domain with HTTPS.

#### Option B: Cloudflare Pages (free tier)

| Item | Cost | Notes |
| ---- | ---- | ----- |
| Hosting | Free | 500 builds/month, unlimited bandwidth, unlimited sites |
| Domain (via Cloudflare Registrar) | ~US$9-12/yr | At-cost pricing, often the cheapest option |
| SSL/TLS | Free | Automatic via Cloudflare |
| **Total** | **~US$9-12/yr** | |

Advantages over GitHub Pages: faster global CDN, can use private repos on free tier, built-in analytics, Web Application Firewall.

#### Option C: Netlify (free tier)

| Item | Cost | Notes |
| ---- | ---- | ----- |
| Hosting | Free | 100 GB bandwidth/month, 300 build minutes/month |
| Domain | ~US$10-15/yr | Via any registrar |
| SSL/TLS | Free | Automatic via Let's Encrypt |
| **Total** | **~US$10-15/yr** | |

Advantages: deploy previews on PRs (very useful for reviewing visual changes), form handling, serverless functions if ever needed.

#### Option D: Vercel (free tier)

| Item | Cost | Notes |
| ---- | ---- | ----- |
| Hosting | Free | 100 GB bandwidth/month, hobby tier |
| Domain | ~US$10-15/yr | Via any registrar |
| SSL/TLS | Free | Automatic |
| **Total** | **~US$10-15/yr** | |

Similar to Netlify. Vercel's free tier is intended for personal, non-commercial projects (which this qualifies as).

#### Option E: Self-hosted (VPS)

| Item | Cost | Notes |
| ---- | ---- | ----- |
| VPS (e.g. Hetzner, DigitalOcean, Linode) | ~US$4-6/mo | Smallest tier, 1 vCPU, 1 GB RAM is sufficient for a static site behind nginx |
| Domain | ~US$10-15/yr | Via any registrar |
| SSL/TLS | Free | Let's Encrypt with certbot |
| **Total** | **~US$60-87/yr** | |

Only justified if you plan to run other services on the same VPS (e.g. analytics, API backends). Requires server maintenance.

#### Recommendation

Start with **Option A** (GitHub Pages + custom domain). It is the simplest, cheapest, and requires zero migration. If you later want deploy previews or a private repo on free hosting, move to **Option B** (Cloudflare Pages).

### Phase 4: Production Hardening

Goal: Ensure the site is performant, accessible, and properly configured for search engines.

- [ ] **Run Lighthouse audit** - target 90+ on Performance, Accessibility, Best Practices, SEO.
- [ ] **Configure Google Analytics** (optional) - add the GA4 measurement ID to `_config.yml` `google_analytics.id`. Or consider privacy-friendly alternatives like Plausible or Umami.
- [ ] **Verify sitemap** - `assets/robots.txt` and the Jekyll-generated `sitemap.xml` should be correct. Submit sitemap to Google Search Console.
- [ ] **Set up Google Search Console** - verify ownership via the `google_site_verification` field in `_config.yml`.
- [ ] **Review `assets/404.html`** - ensure the 404 page is helpful and styled consistently with the theme.
- [ ] **Validate PWA** - test the service worker registration and offline behaviour. The `pwa.enabled: true` config is set, but verify it works end-to-end.
- [ ] **Check meta tags and Open Graph** - verify social sharing previews render correctly (use Facebook's Sharing Debugger, Twitter Card Validator).
- [ ] **Add `description` to `_config.yml`** - the current value is empty after the `>-` block scalar. Fill it in for SEO.

### Phase 5: Content and Maintenance

Goal: Establish processes for ongoing content creation and dependency maintenance.

- [ ] **Set up Dependabot** for Ruby gem updates. Create `.github/dependabot.yml`:
  ```yaml
  version: 2
  updates:
    - package-ecosystem: "bundler"
      directory: "/"
      schedule:
        interval: "weekly"
  ```
- [ ] **Document the upstream sync process** in the README or this plan:
  1. Add the upstream remote (done in Phase 2).
  2. Fetch upstream: `git fetch upstream`.
  3. Create a branch: `git checkout -b sync/chirpy-vX.Y`.
  4. Merge or cherry-pick relevant changes.
  5. Resolve conflicts, test locally, push and create a PR.
- [ ] **Establish a content authoring workflow**:
  1. Create a new branch for each post or batch of content changes.
  2. Write the post in `_posts/` following the `YYYY-MM-DD-title.md` naming convention.
  3. Preview locally with `make serve`.
  4. Push and create a PR - CI validates the build, you can preview via deploy preview if using Netlify/Cloudflare Pages.
  5. Merge to `main` to publish.
- [ ] **Consider a Docker-based dev environment** (optional, lower priority):
  ```dockerfile
  FROM ruby:3.3-slim
  WORKDIR /site
  COPY Gemfile Gemfile.lock ./
  RUN bundle install
  COPY . .
  CMD ["bundle", "exec", "jekyll", "serve", "-H", "0.0.0.0", "--livereload"]
  ```
  This ensures identical builds across macOS and Linux without needing to install Ruby locally.

---

## 5. Priority Order

| Priority | Phase | Effort | Impact |
| -------- | ----- | ------ | ------ |
| 1 | Phase 0: Foundation | Low | High (unblocks everything else) |
| 2 | Phase 1: CI/CD Hardening | Low | High (catches issues before deploy) |
| 3 | Phase 4: Production Hardening | Low-Medium | Medium (SEO, performance) |
| 4 | Phase 3: Custom Domain | Low | Medium (professional URL) |
| 5 | Phase 5: Maintenance | Low | Medium (long-term health) |
| 6 | Phase 2: Theme Upgrade | High | Medium (new features, but risky) |

The theme upgrade (Phase 2) is listed last because it carries the most risk and effort. The site works fine on v5.x. The other phases deliver higher value for lower effort.
