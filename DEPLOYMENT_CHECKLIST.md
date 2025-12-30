# Hugging Face Spaces Deployment Checklist

## ✅ Changes Made

### 1. Requirements Updated
- ✅ Added `huggingface-hub>=0.20.0` for HuggingFace dataset access
- ✅ Added `pyarrow>=14.0.0` for parquet file reading
- ✅ Added `plotly>=5.18.0` for visualizations

### 2. Data Loader Fixed
- ✅ Updated to read `HF_TOKEN` from environment (Hugging Face Spaces uses this)
- ✅ Added fallback to `HUGGINGFACE_TOKEN` for local development
- ✅ Added `sub_division` parquet file to the list
- ✅ Improved error handling for missing files

### 3. Frontend Updated
- ✅ All API URLs now use environment variable `API_BASE_URL`
- ✅ Defaults to `http://localhost:54300` for local dev
- ✅ Added backend health check on startup
- ✅ Updated files:
  - `frontend/app.py`
  - `frontend/utils.py`
  - `frontend/pages/05_Service_Recommendation.py`
  - `frontend/pages/06_Underperforming_BSKs.py`

### 4. Dockerfile Optimized
- ✅ Fixed paths for Hugging Face Spaces structure
- ✅ Added proper startup sequence (backend first, then frontend)
- ✅ Added 5-second wait for backend to initialize
- ✅ Set environment variables
- ✅ Exposed port 7860 (Hugging Face Spaces requirement)

## 📋 Deployment Steps

### Step 1: Verify Your Dataset
Make sure all these files are in your Hugging Face dataset (`coe5332/bsk-database`):
- [ ] `dbo_ml_block_municipality.parquet`
- [ ] `dbo_ml_bsk_master.parquet`
- [ ] `dbo_ml_citizen_master_v2.parquet`
- [ ] `dbo_ml_deo_master.parquet`
- [ ] `dbo_ml_department_master.parquet`
- [ ] `dbo_ml_district.parquet`
- [ ] `dbo_ml_gp_ward_master.parquet`
- [ ] `dbo_ml_post_office_master.parquet`
- [ ] `dbo_ml_provision.parquet`
- [ ] `dbo_ml_service_master.parquet`
- [ ] `dbo_ml_sub_division.parquet`

### Step 2: Create Hugging Face Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Configure:
   - **SDK**: Docker
   - **Visibility**: Public (or Private)
   - **Hardware**: CPU Basic (upgrade if needed for large datasets)

### Step 3: Add Secret Token
1. In your Space, go to **Settings** → **Variables and secrets**
2. Click **"New secret"**
3. Add:
   - **Key**: `HF_TOKEN`
   - **Value**: Your Hugging Face token (starts with `hf_`)

### Step 4: Upload Code
Upload all files from your repository to the Space. The structure should be:
```
your-space/
├── Dockerfile
├── requirements.txt
├── README.md (optional)
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── data_loader.py
│   │   └── ...
│   └── run.py
├── frontend/
│   ├── app.py
│   ├── pages/
│   └── ...
└── ...
```

### Step 5: Monitor Build
1. Check the **Logs** tab in your Space
2. Look for:
   - ✅ "Starting FastAPI backend..."
   - ✅ "Data loaded successfully"
   - ✅ "Starting Streamlit frontend..."
3. If you see errors, check:
   - Token is set correctly
   - Dataset repository name matches
   - All parquet files are uploaded

## 🔍 Troubleshooting

### Issue: Blank page or "Backend not available"
**Solution**: 
- Wait 30-60 seconds for backend to fully start
- Check logs for data loading progress
- Verify `HF_TOKEN` secret is set

### Issue: "Authentication failed" or "401 Unauthorized"
**Solution**:
- Verify `HF_TOKEN` secret is correct
- Check token has access to dataset repository
- Token should start with `hf_`

### Issue: "Failed to load [table_name]"
**Solution**:
- Verify parquet file exists in dataset
- Check file name matches exactly (case-sensitive)
- Verify token has read access to dataset

### Issue: Frontend shows but no data
**Solution**:
- Check backend logs for data loading errors
- Verify all parquet files are uploaded
- Check `/health` endpoint: `https://your-space.hf.space/health` (if exposed)

## 📊 Expected Behavior

### On Startup:
1. Backend starts and loads data from HuggingFace (takes 10-30 seconds)
2. Frontend starts and connects to backend
3. Health check verifies backend is ready
4. Data displays in the UI

### Logs Should Show:
```
🟢 Starting FastAPI backend...
Backend started with PID: [number]
Loading 11 Parquet files from HuggingFace...
Dataset: coe5332/bsk-database
Loading bsk_master...
✅ bsk_master: [rows] rows, [cols] columns
...
✅ Data loaded successfully: 11 tables, [total] total rows
🟢 Starting Streamlit frontend...
```

## 🎯 Quick Test

After deployment, check:
1. **Backend Health**: Look for "✅ Backend connected" message
2. **Data Loading**: Check if metrics show numbers > 0
3. **Navigation**: Try clicking through different pages
4. **API Endpoints**: Backend should respond at `/health`, `/bsk/`, etc.

## 📝 Notes

- **First Load**: Initial data download may take 1-2 minutes
- **Subsequent Loads**: Data is cached, so faster
- **Memory**: Large datasets may require upgraded hardware
- **Ports**: Backend runs on 54300 (internal), Frontend on 7860 (public)

## 🆘 Still Having Issues?

1. Check the **Logs** tab for detailed error messages
2. Verify all files are uploaded correctly
3. Ensure token has proper permissions
4. Try rebuilding the Space (Settings → Rebuild)

