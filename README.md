# raul-bermejo.github.io

Personal website and portfolio for Raul Bermejo, built with [Jekyll](https://jekyllrb.com/) and the [Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy).

Live at: [raul-bermejo.github.io](https://raul-bermejo.github.io)

## Prerequisites

- Ruby (see `.ruby-version` for the expected version)
- Bundler (`gem install bundler`)

Follow the [Jekyll installation docs](https://jekyllrb.com/docs/installation/) if you need to set up Ruby from scratch.

## Setup

```
bundle install
```

## Local development

```
bundle exec jekyll serve -H 0.0.0.0 --livereload
```

The site will be available at `http://localhost:4000`.

## Build

```
bundle exec jekyll build
```

The output is written to `_site/`.

## Roadmaps

- [Production Roadmap](docs/ROADMAP_PRODUCTION.md) - repo hygiene, CI/CD hardening, theme upgrade, SEO, and maintenance
- [Hosting Roadmap](docs/ROADMAP_HOSTING.md) - custom domain and hosting options with cost comparison
- [Development Plan](docs/DEVELOPMENT_PLAN.md) - full project audit and requirements

## Upstream theme

This site is based on the [Chirpy Jekyll theme](https://github.com/cotes2020/jekyll-theme-chirpy) by Cotes Chung.

## Licence

This work is published under the [MIT](LICENSE) licence.
