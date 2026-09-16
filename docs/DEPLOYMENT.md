# Deployment Guide: SIH26038 Diabetic Retinopathy Screening

This guide covers deploying the production stack:
- **Backend (FastAPI + ML Pipeline):** Hosted on [Render](https://render.com)
- **Frontend (Next.js 14 Web UI):** Hosted on [Vercel](https://vercel.com)

---

## 1. Architecture & Deployment Overview

```
┌────────────────────────────────┐         HTTPS / REST API          ┌───────────────────────────────────┐
│        Vercel Frontend         │ ────────────────────────────────> │          Render Backend           │
│           (Next.js)            │                                   │             (FastAPI)             │
│   https://<app>.vercel.app     │ <──────────────────────────────── │     https://<app>.onrender.com    │
└────────────────────────────────┘          CORS protected           └─────────────────┬─────────────────┘
                                                                                       │
                                                                           Build Step  │ Download & Verify
                                                                                       ▼ SHA-256
                                                                     ┌───────────────────────────────────┐
                                                                     │         Hugging Face Hub          │
                                                                     │manudaza/retinal-triage-           │
                                                                     │efficientnetb0                     │
                                                                     └───────────────────────────────────┘
```

- **Frontend:** Pure client/static web application deployed on Vercel. Points to the Render backend via `NEXT_PUBLIC_API_URL`.
- **Backend:** FastAPI service on Render running with Uvicorn. Model weights (`efficientnetb0_finetuned_patched.keras`) are automatically downloaded from Hugging Face and verified against SHA-256 checksum `2a993b3affc96051aedc79d7b4c20f1538c6c9199d91032b22cfcc4642a24dd4` during the **Build Command** phase.
- **Data Protection:** No external medical datasets (APTOS, IDRiD, DRIVE, Drishti) are downloaded during deployment. The application uses deterministic synthetic fixtures for demonstration and testing.

---

## 2. Render Backend Deployment (FastAPI)

### Step-by-Step Setup
1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** and select **Web Service**.
3. Connect your GitHub repository: `https://github.com/ayushgaur-dev/Diabetic-Retinography`.
4. Configure the service settings:
   - **Name:** `sih26038-retina-backend` (or your preferred name)
   - **Region:** Choose the region closest to your users (e.g., Singapore, Frankfurt, Oregon)
   - **Branch:** `main`
   - **Root Directory:** Leave blank (repository root)
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install -r backend/requirements.txt && python scripts/download_model.py
     ```
   - **Start Command:**
     ```bash
     uvicorn backend.api:app --host 0.0.0.0 --port $PORT
     ```
   - **Instance Type:** Minimum **Starter** (1 CPU, 1 GB RAM) or higher recommended. *Note on Free tier: TensorFlow and computer vision evidence analysis require ~600MB–900MB RAM during heavy evidence extraction; Free instances with 512MB RAM may experience Out-Of-Memory (OOM) termination on unconstrained images.*

### Backend Environment Variables
In the **Environment** tab of your Render service, add the following variables:

| Variable Name | Recommended Value | Description |
|---|---|---|
| `FRONTEND_URL` | `https://<your-vercel-app>.vercel.app` | Comma-separated allowed frontend origins for CORS. Trailing slashes are automatically sanitized. |
| `PYTHON_VERSION` | `3.11.11` (or `3.12.x`) | Python runtime version for Render build environment. |

---

## 3. Vercel Frontend Deployment (Next.js)

### Step-by-Step Setup
1. Log into your [Vercel Dashboard](https://vercel.com).
2. Click **Add New...** → **Project**.
3. Select the `Diabetic-Retinography` repository.
4. In the **Configure Project** screen:
   - **Project Name:** `sih26038-retina-frontend` (or your preferred name)
   - **Framework Preset:** `Next.js`
   - **Root Directory:** Click **Edit** and select **`frontend`**.
   - **Build and Output Settings:** Leave default (`npm run build`, output directory automatically handled).
5. In the **Environment Variables** section:
   - Add:
     - **Key:** `NEXT_PUBLIC_API_URL`
     - **Value:** `https://<your-render-backend-url>.onrender.com` (use your actual Render backend URL without trailing slash)
6. Click **Deploy**.

---

## 4. Model Download Behavior

- The model downloader is defined in `scripts/download_model.py`.
- **Target Repository:** Hugging Face Hub `manudaza/retinal-triage-efficientnetb0`
- **Target File:** `efficientnetb0_finetuned_patched.keras` (~22.3 MB)
- **Destination:** `models/efficientnetb0_finetuned_patched.keras`
- **Integrity:** The script computes the SHA-256 checksum and compares it against:
  ```
  2a993b3affc96051aedc79d7b4c20f1538c6c9199d91032b22cfcc4642a24dd4
  ```
- **Caching & Idempotency:** If the model already exists in `models/` with the correct checksum, download is skipped.
- **Fail-Safe:** If checksum verification fails, the file is removed and the build process exits with code 1, preventing corrupted models from being deployed.

---

## 5. Local Development Workflow

Both frontend and backend are configured to work locally out-of-the-box without editing configuration files.

### Terminal 1: Backend
```powershell
# Windows
.\venv\Scripts\Activate.ps1
uvicorn backend.api:app --host 127.0.0.1 --port 8001
```
```bash
# Linux/macOS
source venv/bin/activate
uvicorn backend.api:app --host 127.0.0.1 --port 8001
```

### Terminal 2: Frontend
```bash
cd frontend
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) or [http://localhost:3001](http://localhost:3001) in your browser. The frontend automatically resolves to `http://localhost:8001` or `http://127.0.0.1:8001` when `NEXT_PUBLIC_API_URL` is omitted.

---

## 6. Production Testing & Verification

Once deployed, verify the full stack with these test requests:

### 1. Health Check
```bash
curl https://<your-render-backend>.onrender.com/api/health
```
Expected Response:
```json
{"status": "ok", "pipeline": "frozen", "llm": "none"}
```

### 2. Synthetic Demo Job Creation
```bash
curl -X POST https://<your-render-backend>.onrender.com/api/demo
```
Expected Response:
```json
{
  "job_id": "<hex_id>",
  "image_hash": "<sha256_hash>",
  "synthetic": true,
  "note": "Synthetic fixture; not a patient image."
}
```

### 3. Poll Job Status
```bash
curl https://<your-render-backend>.onrender.com/api/jobs/<job_id>
```
When complete (`"state": "done"`), response contains full clinical screening summary, calibrated confidence, triage classification, and rendered overlay layers.

---

## 7. Common Deployment Problems & Troubleshooting

### Problem 1: Render Cold Starts (Free Tier)
- **Symptom:** First request to backend takes 45–60 seconds, or frontend displays "API connection failed" initially.
- **Cause:** Render free tier instances spin down after 15 minutes of inactivity.
- **Solution:** Wait for instance spin-up or upgrade the Render service to the Starter tier ($7/mo) to keep it always alive.

### Problem 2: Out of Memory (OOM) Kill on Render
- **Symptom:** The Render backend crashes or restarts during image analysis with `Exit status 137` or memory alert.
- **Cause:** Loading TensorFlow + executing full anatomical evidence extraction (vessels, disc, fovea, lesion candidates) requires ~700MB–1GB RAM.
- **Solution:** Select the **Starter** instance (1 GB RAM) or higher on Render instead of Free tier (512 MB).

### Problem 3: CORS Errors in Browser Console
- **Symptom:** Browser shows `Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource at ...`
- **Cause:** `FRONTEND_URL` on Render does not match your Vercel domain.
- **Solution:** Verify the `FRONTEND_URL` environment variable in Render. Ensure it includes the scheme (`https://`) and matches your exact Vercel URL (e.g. `https://my-app.vercel.app`). Note: trailing slashes are automatically handled by the backend.

### Problem 4: Missing Model File at Startup
- **Symptom:** Backend log displays `FileNotFoundError: ... models/efficientnetb0_finetuned_patched.keras`.
- **Cause:** Build command did not run `python scripts/download_model.py`.
- **Solution:** Ensure Render Build Command is set to:
  `pip install -r backend/requirements.txt && python scripts/download_model.py`

### Problem 5: Vercel 404 on Root Route
- **Symptom:** Vercel deployment renders 404 or directory listing instead of Next.js app.
- **Cause:** Vercel Root Directory was left as repository root instead of `frontend`.
- **Solution:** In Vercel Project Settings → General → Root Directory, set to **`frontend`** and redeploy.
