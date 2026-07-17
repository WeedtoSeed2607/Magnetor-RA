# Deploying the Magnetor dashboard (Streamlit Community Cloud)

The repo is deploy-ready: a bundled read-only snapshot (`sample_data/`), a cloud
entry point (`streamlit_app.py`), `requirements.txt`, and a **password-gated**
search box so the hosted link can't drain your API key.

You do the account + push + connect steps (I can't act as you for those).

## 0. Prerequisites
- A **GitHub account** (free) — https://github.com/join
- A **Streamlit Community Cloud account** (free, sign in with GitHub) —
  https://share.streamlit.io
- Your Voyage API key (in your local `.env`, **never** committed).

## 1. Push the repo to GitHub
Create a **private** repo named `magnetor` on GitHub (private keeps the code +
bundled corpus off the public web; the abstracts are public anyway, but there's
no reason to broadcast the whole set). Then, from `C:\Users\janak\Magnetor`:

```powershell
git remote add origin https://github.com/<your-username>/magnetor.git
git branch -M main
git push -u origin main
```

`.gitignore` already excludes `.env`, `*.key`, and your live `/data/`, so no
secret and no local corpus leaves your machine — only the vetted `sample_data/`
snapshot goes up.

## 2. Create the app on Streamlit Cloud
1. Go to https://share.streamlit.io → **New app** → **Deploy from GitHub**.
2. Repository: `<your-username>/magnetor`  ·  Branch: `main`
3. **Main file path:** `streamlit_app.py`
4. Click **Advanced settings → Secrets** and paste (TOML):

   ```toml
   MAGNETOR_VOYAGE_API_KEY = "pa-your-real-voyage-key"
   MAGNETOR_SEARCH_PASSWORD = "pick-a-password-to-share"
   # MAGNETOR_S2_API_KEY = "optional-semantic-scholar-key"
   ```

5. **Deploy.** First build takes a few minutes (installs `requirements.txt`).

## 3. Share
Send people the app URL **and** the `MAGNETOR_SEARCH_PASSWORD`. Without the
password they can browse the Topic-Trend Banner, Frontier Feed, and paper links
(all served from the snapshot, zero API cost); with it they can run the live
deep-dive. If you ever suspect the password leaked, change the secret and the
app redeploys — the old password stops working.

## 4. Updating the data later
The hosted app shows the committed snapshot, not your live corpus. To refresh it
after acquiring/embedding/recomputing trends locally:

```powershell
python scripts/refresh_snapshot.py
git add sample_data
git commit -m "refresh dashboard snapshot"
git push
```

Streamlit Cloud auto-redeploys on push.

## Security notes
- The Voyage key lives **only** in Streamlit's Secrets store, never in the repo.
- The search box is gated by `MAGNETOR_SEARCH_PASSWORD`; the banner/feed/links
  need no key and stay open.
- The snapshot contains only public paper metadata + embedding vectors — no
  credentials (verified before commit).
