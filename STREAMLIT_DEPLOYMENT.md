# Streamlit Cloud Deployment Guide

## ✅ Setup Complete!

Your Streamlit frontend has been configured and pushed to GitHub. Here's what was done:

### Changes Made:

1. **Updated API URLs**: All frontend files now use the deployed backend at `https://bsk-backend-uywi.onrender.com`
   - `frontend/app.py`
   - `frontend/utils.py`
   - `frontend/pages/05_Service_Recommendation.py`
   - `frontend/pages/06_Underperforming_BSKs.py`

2. **Created Streamlit Configuration**:
   - `.streamlit/config.toml` - Streamlit Cloud settings
   - `requirements.txt` - Optimized dependencies for frontend only

3. **Created Git Configuration**:
   - `.gitignore` - Excludes unnecessary files
   - `README_STREAMLIT.md` - Deployment documentation

4. **Pushed to GitHub**:
   - Repository: `https://github.com/coe5332-code/good.git`
   - Frontend submodule: `https://github.com/coe5332-code/bsk-good-system.git`

## 🚀 Deploy to Streamlit Cloud

### Step 1: Go to Streamlit Cloud
Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.

### Step 2: Create New App
1. Click **"New app"**
2. Select repository: **`coe5332-code/good`**
3. Set **Main file path**: `frontend/app.py`
4. Set **Branch**: `main`
5. Click **"Deploy"**

### Step 3: Wait for Deployment
- Streamlit Cloud will install dependencies from `requirements.txt`
- The app will connect to your backend at `https://bsk-backend-uywi.onrender.com`
- First deployment may take 2-5 minutes

### Step 4: Access Your App
Once deployed, you'll get a URL like:
`https://your-app-name.streamlit.app`

## 📋 Important Notes

1. **Backend URL**: The app is configured to use `https://bsk-backend-uywi.onrender.com` by default. If you need to change it, set the `API_BASE_URL` environment variable in Streamlit Cloud settings.

2. **Backend Health**: Make sure your backend is running and accessible. Check health at:
   `https://bsk-backend-uywi.onrender.com/health`

3. **Dependencies**: The `requirements.txt` includes only frontend dependencies. Backend dependencies are not needed for Streamlit Cloud.

4. **Database**: The PostgreSQL connection string you provided is for the backend, not needed for the frontend.

## 🔧 Troubleshooting

### App won't start
- Check Streamlit Cloud logs for errors
- Verify `requirements.txt` has all dependencies
- Ensure backend is accessible

### Backend connection errors
- Verify backend URL is correct
- Check backend health endpoint
- Ensure CORS is enabled on backend

### Import errors
- Check that all dependencies in `requirements.txt` are correct
- Verify Python version (3.8+ required)

## 📁 Repository Structure

```
good/
├── frontend/                    # Streamlit frontend (submodule)
│   ├── app.py                  # Main app file
│   ├── pages/                  # Streamlit pages
│   └── utils.py                # Utility functions
├── requirements.txt            # Frontend dependencies
├── .streamlit/
│   └── config.toml             # Streamlit config
├── .gitignore                  # Git ignore rules
└── README_STREAMLIT.md         # Documentation
```

## 🎉 You're Ready!

Your frontend is now ready to be deployed on Streamlit Cloud. Just follow the steps above and you'll have your app running in minutes!

