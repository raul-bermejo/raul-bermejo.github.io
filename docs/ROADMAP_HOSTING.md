# Hosting and Custom Domain Roadmap

Options for adding a custom domain and/or migrating to a different hosting provider. This is not urgent - the site works fine on GitHub Pages at `raul-bermejo.github.io`. Revisit when ready.

For production improvements, see [ROADMAP_PRODUCTION.md](ROADMAP_PRODUCTION.md).

---

## Option A: GitHub Pages + custom domain (recommended starting point)

| Item | Cost | Notes |
| ---- | ---- | ----- |
| GitHub Pages hosting | Free | Requires public repo on free tier (GitHub Pro ~US$4/mo for private repos) |
| Domain registration | ~US$10-15/yr | Varies by registrar and TLD |
| DNS management | Free | Most registrars include DNS; Cloudflare free tier is excellent |
| SSL/TLS certificate | Free | Automatic HTTPS via Let's Encrypt |
| **Total** | **~US$10-15/yr** | |

**Recommended registrars:** Cloudflare Registrar (at-cost pricing, typically cheapest), Porkbun, Namecheap.

**Steps to implement:**
- [ ] Purchase a domain (e.g. `raulbermejo.com`, `raulbermejo.dev`).
- [ ] Configure DNS: add `A` records pointing to GitHub Pages IPs and a `CNAME` for `www`.
- [ ] Add a `CNAME` file to the repo root containing the custom domain.
- [ ] Update `_config.yml` `url` field to `https://yourdomain.com`.
- [ ] Enable "Enforce HTTPS" in GitHub Pages settings.
- [ ] Verify the site loads on the custom domain with HTTPS.

---

## Option B: Cloudflare Pages (free tier)

| Item | Cost | Notes |
| ---- | ---- | ----- |
| Hosting | Free | 500 builds/month, unlimited bandwidth, unlimited sites |
| Domain (via Cloudflare Registrar) | ~US$9-12/yr | At-cost pricing, often the cheapest |
| SSL/TLS | Free | Automatic via Cloudflare |
| **Total** | **~US$9-12/yr** | |

**Advantages over GitHub Pages:** faster global CDN, private repos on free tier, built-in analytics, Web Application Firewall.

---

## Option C: Netlify (free tier)

| Item | Cost | Notes |
| ---- | ---- | ----- |
| Hosting | Free | 100 GB bandwidth/month, 300 build minutes/month |
| Domain | ~US$10-15/yr | Via any registrar |
| SSL/TLS | Free | Automatic via Let's Encrypt |
| **Total** | **~US$10-15/yr** | |

**Advantages:** deploy previews on PRs (useful for reviewing visual changes), form handling, serverless functions if needed.

---

## Option D: Vercel (free tier)

| Item | Cost | Notes |
| ---- | ---- | ----- |
| Hosting | Free | 100 GB bandwidth/month, hobby tier |
| Domain | ~US$10-15/yr | Via any registrar |
| SSL/TLS | Free | Automatic |
| **Total** | **~US$10-15/yr** | |

Free tier is for personal, non-commercial projects (which this qualifies as). Similar feature set to Netlify.

---

## Option E: Self-hosted (VPS)

| Item | Cost | Notes |
| ---- | ---- | ----- |
| VPS (Hetzner, DigitalOcean, Linode) | ~US$4-6/mo | Smallest tier (1 vCPU, 1 GB RAM) is enough for a static site behind nginx |
| Domain | ~US$10-15/yr | Via any registrar |
| SSL/TLS | Free | Let's Encrypt with certbot |
| **Total** | **~US$60-87/yr** | |

Only justified if running other services on the same VPS (analytics, API backends, etc.). Requires ongoing server maintenance.

---

## Recommendation

Start with **Option A** (GitHub Pages + custom domain) when ready. Simplest, cheapest, zero migration needed. If you later want deploy previews or private-repo free hosting, move to **Option B** (Cloudflare Pages).
