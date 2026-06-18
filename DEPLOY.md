# Deploying Tide

Tide has two faces:

| | What | Where | Cost |
|---|---|---|---|
| **Static snapshot** | The Claude-Design prototype (`docs/index.html`) with **simulated** data | GitHub Pages | free |
| **Live app** | The real FastAPI backend + wired dashboard (calibrated model, paper trading, analyst, smart-money) | a host that runs Python (Render) | free tier |

GitHub Pages can only serve static files, so it gets the simulated snapshot. The
live app needs a Python host.

---

## 1. Static snapshot → GitHub Pages

Already wired: this repo's Pages is served from **`/docs`** on the default branch.
The site is `https://<user>.github.io/tide/`. Every push to `main` redeploys it.
It runs entirely client-side on simulated data — no backend, no keys.

To change Pages source later: repo **Settings → Pages → Build and deployment →
Deploy from a branch → `main` / `/docs`**.

---

## 2. Live app → Render (free)

The repo ships a `render.yaml` blueprint and a `Procfile`.

1. Push this repo to GitHub (done).
2. Go to **https://render.com → New → Blueprint**, connect the repo. Render reads
   `render.yaml` and provisions a free web service.
   - Build: `pip install -r backend/requirements.txt`
   - Start: `cd backend && uvicorn server:app --host 0.0.0.0 --port $PORT`
   - Health check: `/api/health`
3. In the service's **Environment** tab, set the secrets (left blank in the blueprint):
   - `ANTHROPIC_API_KEY` — enables Claude analyst notes
   - `POLYGON_API_KEY` — enables real market data; then set `DATA_PROVIDER=polygon`
   - (optional) `QUIVER_API_KEY` or `FINNHUB_API_KEY` — smart-money overlay
4. The deployed URL serves **both** the API and the wired dashboard at `/`
   (FastAPI mounts `frontend/` after the API routes).

**Notes**
- Without `POLYGON_API_KEY` + a trained model, the live app runs in mock/no-model
  mode (leaderboard ranks by the activity score; probabilities are suppressed).
  To get real probabilities, set the key and run `python backend/train.py --start … --end …`
  to produce `backend/.model/model.joblib`, then redeploy (or train on a worker).
- The free Render instance sleeps when idle and cold-starts on the next request.
- Secrets are **never** committed — `.env`, `.model/`, `.paper/`, and all caches
  are git-ignored. Set every key in the host's dashboard, not in the repo.

---

## Local

```
cd backend
pip install -r requirements.txt
cp .env.example .env          # add keys here for local use (git-ignored)
python train.py --synthetic   # or --start/--end with a Polygon key
uvicorn server:app --port 8000
```
Open http://localhost:8000/ for the wired dashboard, or open
`frontend/index.html` directly (it falls back to `http://localhost:8000`).
