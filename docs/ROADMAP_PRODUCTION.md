# Production Roadmap

Checklist for taking raul-bermejo.github.io from working to production-grade. Phases are ordered by priority (highest value, lowest effort first). The theme upgrade is last because it carries the most risk.

For hosting and custom domain options, see [ROADMAP_HOSTING.md](ROADMAP_HOSTING.md).

---

## Phase 0: Foundation (local dev and repo hygiene)

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
- [ ] **Rewrite `README.md`** - describe the project, prerequisites, setup instructions (referencing the Makefile), and link to these roadmaps.
- [ ] **Clean up `jekyll-theme-chirpy.gemspec`** - either delete it (not needed when using the gem via Gemfile) or update the version to match reality.
- [ ] **Remove `tools/init.sh`** - upstream initialisation script, already run, no longer useful.
- [ ] **Review `tools/deploy.sh` and `tools/release.sh`** - likely dead code given GitHub Actions handles deployment. Remove if unused.

---

## Phase 1: CI/CD Hardening

Goal: Every PR is validated before merge, deployments are reliable.

- [ ] **Remove `.github/workflows/pages-deploy.yml`** - targets `master` branch with Ruby 2.7 and `actions/checkout@v2`. Dead workflow.
- [ ] **Verify `pages.yml` action versions are current** - `actions/checkout@v4`, `ruby/setup-ruby@v1`, `actions/configure-pages@v5`, `actions/upload-pages-artifact@v3`, `actions/deploy-pages@v4`.
- [ ] **Add HTML-Proofer to CI** - run `htmlproofer` on the built `_site/` to catch broken links, missing alt text, and invalid HTML. Add as a step after Jekyll build in the PR path.
- [ ] **Add `html-proofer` gem to Gemfile**.
- [ ] **Add build status badge** to README.
- [ ] **Consider branch protection rules** on `main` - require the build job to pass before merge.

---

## Phase 2: Theme Upgrade (Chirpy v5.x to latest)

Goal: Bring the theme up to date for new features, security patches, and upstream compatibility.

Chirpy v6.x and v7.x introduced breaking changes (new directory structure, different JS bundling, updated config keys). This phase is intentionally separate and last in priority.

- [ ] **Research the current latest Chirpy version** and read the [upgrade guide](https://github.com/cotes2020/jekyll-theme-chirpy/wiki/Upgrade-Guide).
- [ ] **Create a feature branch** (`feat/chirpy-upgrade`).
- [ ] **Update `Gemfile`** to target the new version.
- [ ] **Reconcile `_config.yml`** with the new theme's default config (new keys, renamed keys, removed keys).
- [ ] **Review `_includes/`, `_layouts/`, `_sass/`** for local overrides that may conflict. The current repo has extensive local copies of theme files - many may be deletable if the gem provides them.
- [ ] **Update `_data/` files** to match the new theme's expected format.
- [ ] **Update or remove `gulpfile.js/` and `package.json`** - newer Chirpy versions may use a different JS build approach.
- [ ] **Test locally** - build, serve, check all pages, verify dark/light mode, check mobile layout.
- [ ] **Test in CI** - push to the feature branch and verify the build passes.
- [ ] **Add upstream remote** for future syncs:
  ```
  git remote add upstream https://github.com/cotes2020/jekyll-theme-chirpy.git
  ```

---

## Phase 4: Production Hardening

Goal: Performant, accessible, and properly configured for search engines.

- [ ] **Run Lighthouse audit** - target 90+ on Performance, Accessibility, Best Practices, SEO.
- [ ] **Fill in `_config.yml` `description`** - currently empty after the `>-` block scalar.
- [ ] **Configure analytics** (optional) - add a GA4 measurement ID to `_config.yml`, or consider privacy-friendly alternatives (Plausible, Umami).
- [ ] **Verify sitemap** - check `robots.txt` and the generated `sitemap.xml`. Submit to Google Search Console.
- [ ] **Set up Google Search Console** - verify ownership via the `google_site_verification` field in `_config.yml`.
- [ ] **Review `assets/404.html`** - ensure the 404 page is helpful and styled consistently.
- [ ] **Validate PWA** - test service worker registration and offline behaviour (`pwa.enabled: true` is set, verify it works end-to-end).
- [ ] **Check Open Graph / social meta tags** - verify sharing previews render correctly on LinkedIn, Twitter, etc.

---

## Phase 5: Content and Maintenance

Goal: Sustainable processes for ongoing content and dependency health.

- [ ] **Set up Dependabot** for Ruby gem updates - create `.github/dependabot.yml`.
- [ ] **Document the upstream sync process** in README:
  1. `git fetch upstream`
  2. Create a branch: `git checkout -b sync/chirpy-vX.Y`
  3. Merge or cherry-pick relevant changes
  4. Resolve conflicts, test locally, push and create a PR
- [ ] **Establish content authoring workflow**:
  1. Branch per post or batch of content changes
  2. Write in `_posts/` following `YYYY-MM-DD-title.md` naming
  3. Preview locally with `make serve`
  4. Push, create PR, CI validates the build
  5. Merge to `main` to publish
- [ ] **Consider a Docker dev environment** (optional, lower priority) for consistent builds across macOS and Linux without requiring a local Ruby install.
