# BSK Training Optimization System - Hugging Face Spaces Deployment

This document explains how to deploy the BSK Training Optimization System on Hugging Face Spaces.

## Prerequisites

1. **Hugging Face Account**: You need a Hugging Face account
2. **Dataset Uploaded**: All parquet files should be uploaded to your Hugging Face dataset repository
3. **Access Token**: Your Hugging Face token should be added to the Space secrets

## Setup Instructions

### 1. Create a Hugging Face Space

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **SDK**: Docker
   - **Visibility**: Public (or Private)
   - **Hardware**: CPU Basic (or upgrade if needed)

### 2. Configure Secrets

1. Go to your Space settings
2. Navigate to "Variables and secrets"
3. Add a secret with:
   - **Key**: `HF_TOKEN`
   - **Value**: Your Hugging Face token (starts with `hf_`)

### 3. Upload Your Code

Upload all files from this repository to your Space. The structure should be:
```
your-space/
├── Dockerfile
├── requirements.txt
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

### 4. Verify Dataset Repository

Make sure your dataset repository (`coe5332/bsk-database`) contains all the parquet files:
- `dbo_ml_block_municipality.parquet`
- `dbo_ml_bsk_master.parquet`
- `dbo_ml_citizen_master_v2.parquet`
- `dbo_ml_deo_master.parquet`
- `dbo_ml_department_master.parquet`
- `dbo_ml_district.parquet`
- `dbo_ml_gp_ward_master.parquet`
- `dbo_ml_post_office_master.parquet`
- `dbo_ml_provision.parquet`
- `dbo_ml_service_master.parquet`
- `dbo_ml_sub_division.parquet`

### 5. Build and Deploy

Hugging Face Spaces will automatically build and deploy your application when you push the code.

## How It Works

1. **Backend (FastAPI)**: Runs on port 54300 internally
   - Loads data from Hugging Face dataset on startup
   - Provides REST API endpoints for the frontend

2. **Frontend (Streamlit)**: Runs on port 7860 (exposed publicly)
   - Connects to backend API at `http://localhost:54300`
   - Provides interactive UI for data visualization

## Troubleshooting

### Application shows blank page

1. **Check logs**: Click on "Logs" tab in your Space
2. **Verify token**: Ensure `HF_TOKEN` secret is set correctly
3. **Check dataset access**: Verify your token has access to the dataset
4. **Backend startup**: Check if backend started successfully (look for "✅ Data loaded successfully" in logs)

### Data not loading

1. **Check dataset repository**: Verify dataset name matches `coe5332/bsk-database`
2. **Verify parquet files**: Ensure all parquet files are uploaded
3. **Check token permissions**: Token should have read access to the dataset

### Backend connection errors

1. **Wait time**: Backend takes a few seconds to start. The Dockerfile includes a 5-second wait.
2. **Check API_BASE_URL**: Should be `http://localhost:54300` (default)
3. **View backend logs**: Check if FastAPI started successfully

## Environment Variables

The following environment variables are used:

- `HF_TOKEN`: Your Hugging Face token (set as secret)
- `HF_DATASET_REPO`: Dataset repository (default: `coe5332/bsk-database`)
- `API_BASE_URL`: Backend API URL (default: `http://localhost:54300`)

## Architecture

```
┌─────────────────────────────────────┐
│   Hugging Face Spaces Container    │
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │   FastAPI    │  │  Streamlit  │ │
│  │  (Port 54300)│  │ (Port 7860) │ │
│  └──────┬───────┘  └──────┬──────┘ │
│         │                 │        │
│         └────────┬────────┘        │
│                  │                 │
│         ┌────────▼────────┐        │
│         │  Data Loader    │        │
│         │ (HuggingFace)   │        │
│         └────────┬────────┘        │
└──────────────────┼─────────────────┘
                   │
         ┌─────────▼─────────┐
         │ HuggingFace       │
         │ Dataset Repository│
         └───────────────────┘
```

## Support

For issues or questions, check the application logs in your Hugging Face Space dashboard.

