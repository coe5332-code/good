Streamlit deployment instructions

This file describes how to run the Streamlit frontend locally and deploy to Streamlit Cloud.

1) Local run

- (Optional) Create and activate a Python virtual environment.
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Run Streamlit from the project root:

```bash
streamlit run frontend/app.py
```

- By default the app will point to the deployed backend at:

https://bsk-backend-uywi.onrender.com

- To use a local backend, set the environment variable `API_BASE_URL` before running Streamlit.

Windows (PowerShell):

```powershell
$env:API_BASE_URL = "http://localhost:54300"
streamlit run frontend/app.py
```

macOS / Linux:

```bash
export API_BASE_URL="http://localhost:54300"
streamlit run frontend/app.py
```

2) Deploy to Streamlit Cloud

- Push your repository to GitHub (see commit instructions below).
- In Streamlit Cloud, create a new app and point it to the GitHub repo and branch.
- Optionally set the `API_BASE_URL` secret or env var in Streamlit Cloud to the backend URL.

3) Commit & push (example)

```bash
git add frontend/app.py frontend/DEPLOY_STREAMLIT.md
git commit -m "Streamlit: default backend URL + deployment guide"
# Add remote if not present
git remote add origin https://github.com/coe5332-code/good.git
# Push (you may be prompted for credentials)
git push -u origin main
```

If your repo uses a different branch name (e.g., `master`), replace `main` accordingly.
