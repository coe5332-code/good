# backend/app/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
from dotenv import load_dotenv
import logging
import pandas as pd
import numpy as np
import sys

# Import data loader
from app.data_loader import (
    init_cache, 
    get_provisions_df, 
    get_bsk_master_df, 
    get_service_master_df,
    get_deo_master_df,
    get_cache_stats,
    is_initialized
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="BSK Training Optimization API",
    description="API for AI-Assisted Training Optimization System (Parquet-based)",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize data on application startup."""
    logger.info("=" * 60)
    logger.info("Starting BSK API Server (Parquet Mode)")
    logger.info("=" * 60)
    
    try:
        # Start data initialization in a background thread so startup returns quickly.
        # This prevents the container runtime (e.g. Hugging Face Spaces) from waiting
        # for large dataset loads and reporting "starting" indefinitely.
        import threading

        def _init():
            try:
                init_cache()
                logger.info("✅ Data initialization complete (background)")
            except Exception as e:
                logger.error(f"❌ Background data init failed: {e}")

        thread = threading.Thread(target=_init, daemon=True)
        thread.start()
    except Exception as e:
        logger.error(f"❌ Failed to initialize data: {e}")
        logger.error("Application will continue but endpoints may fail")


@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {
        "message": "Welcome to BSK Training Optimization API (Parquet Mode)",
        "version": "2.0.0",
        "data_source": "Parquet files from HuggingFace",
        "initialized": is_initialized()
    }


@app.get("/health")
def health_check():
    """Health check endpoint with data statistics."""
    if not is_initialized():
        raise HTTPException(status_code=503, detail="Data not initialized")
    
    stats = get_cache_stats()
    
    return {
        "status": "healthy",
        "data_initialized": True,
        "statistics": stats
    }


# BSK Master endpoints
@app.get("/bsk/")
def get_bsk_list(skip: int = 0, limit: int = Query(None)):
    logger.info(f"Fetching BSK list with skip={skip}, limit={limit}")
    
    try:
        bsk_df = get_bsk_master_df()
        
        # Apply pagination
        if limit is not None:
            bsk_df = bsk_df.iloc[skip:skip+limit]
        else:
            bsk_df = bsk_df.iloc[skip:]
        
        # Replace NaN with None for JSON serialization
        bsk_df = bsk_df.replace({pd.NA: None, np.nan: None})
        
        records = bsk_df.to_dict('records')
        logger.info(f"Found {len(records)} BSK records")
        
        return records
    except Exception as e:
        logger.error(f"Error fetching BSK list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bsk/{bsk_code}")
def get_bsk(bsk_code: str):
    logger.info(f"Fetching BSK with code: {bsk_code}")
    
    try:
        bsk_df = get_bsk_master_df()
        bsk_record = bsk_df[bsk_df['bsk_code'] == bsk_code]
        
        if bsk_record.empty:
            logger.warning(f"BSK not found with code: {bsk_code}")
            raise HTTPException(status_code=404, detail="BSK not found")
        
        # Replace NaN with None
        bsk_record = bsk_record.replace({pd.NA: None, np.nan: None})
        
        return bsk_record.to_dict('records')[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching BSK: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Service Master endpoints
@app.get("/services/")
def get_services(skip: int = 0, limit: int = Query(None)):
    logger.info(f"Fetching services with skip={skip}, limit={limit}")
    
    try:
        services_df = get_service_master_df()
        
        # Apply pagination
        if limit is not None:
            services_df = services_df.iloc[skip:skip+limit]
        else:
            services_df = services_df.iloc[skip:]
        
        # Replace NaN with None
        services_df = services_df.replace({pd.NA: None, np.nan: None})
        
        records = services_df.to_dict('records')
        logger.info(f"Found {len(records)} service records")
        
        return records
    except Exception as e:
        logger.error(f"Error fetching services: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/services/{service_id}")
def get_service(service_id: int):
    logger.info(f"Fetching service with ID: {service_id}")
    
    try:
        services_df = get_service_master_df()
        service_record = services_df[services_df['service_id'] == service_id]
        
        if service_record.empty:
            logger.warning(f"Service not found with ID: {service_id}")
            raise HTTPException(status_code=404, detail="Service not found")
        
        # Replace NaN with None
        service_record = service_record.replace({pd.NA: None, np.nan: None})
        
        return service_record.to_dict('records')[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# DEO Master endpoints
@app.get("/deo/")
def get_deo_list(skip: int = 0, limit: int = Query(None)):
    logger.info(f"Fetching DEO list with skip={skip}, limit={limit}")
    
    try:
        deo_df = get_deo_master_df()
        
        # Apply pagination
        if limit is not None:
            deo_df = deo_df.iloc[skip:skip+limit]
        else:
            deo_df = deo_df.iloc[skip:]
        
        # Replace NaN with None
        deo_df = deo_df.replace({pd.NA: None, np.nan: None})
        
        records = deo_df.to_dict('records')
        logger.info(f"Found {len(records)} DEO records")
        
        return records
    except Exception as e:
        logger.error(f"Error fetching DEO list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deo/{agent_id}")
def get_deo(agent_id: int):
    logger.info(f"Fetching DEO with agent ID: {agent_id}")
    
    try:
        deo_df = get_deo_master_df()
        deo_record = deo_df[deo_df['agent_id'] == agent_id]
        
        if deo_record.empty:
            logger.warning(f"DEO not found with agent ID: {agent_id}")
            raise HTTPException(status_code=404, detail="DEO not found")
        
        # Replace NaN with None
        deo_record = deo_record.replace({pd.NA: None, np.nan: None})
        
        return deo_record.to_dict('records')[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching DEO: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Provision endpoints
@app.get("/provisions/")
def get_provisions(skip: int = 0, limit: int = Query(None)):
    logger.info(f"Fetching provisions with skip={skip}, limit={limit}")
    
    try:
        provisions_df = get_provisions_df()
        
        # Apply pagination
        if limit is not None:
            provisions_df = provisions_df.iloc[skip:skip+limit]
        else:
            provisions_df = provisions_df.iloc[skip:]
        
        # Replace NaN with None
        provisions_df = provisions_df.replace({pd.NA: None, np.nan: None})
        
        # Convert dates to strings for JSON
        if 'prov_date' in provisions_df.columns:
            provisions_df['prov_date'] = provisions_df['prov_date'].dt.strftime('%Y-%m-%d')
        
        records = provisions_df.to_dict('records')
        logger.info(f"Found {len(records)} provision records")
        
        return records
    except Exception as e:
        logger.error(f"Error fetching provisions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/underperforming_bsks/")
def get_underperforming_bsks(
    num_bsks: int = 50,
    sort_order: str = 'asc'
):
    """Get underperforming BSKs using analytics."""
    try:
        # Import analytics function
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../ai_service')))
        from bsk_analytics import find_underperforming_bsks
        
        # Get data
        bsks_df = get_bsk_master_df()
        provisions_df = get_provisions_df()
        deos_df = get_deo_master_df()
        services_df = get_service_master_df()
        
        # Compute underperforming BSKs
        result_df = find_underperforming_bsks(bsks_df, provisions_df, deos_df, services_df)
        
        # Sort by score
        ascending = sort_order == 'asc'
        result_df = result_df.sort_values(by="score", ascending=ascending).head(num_bsks)
        
        # Replace NaN/inf with None for JSON serialization
        result_df = result_df.replace([np.nan, np.inf, -np.inf], None)
        
        # Return as JSON
        return result_df.to_dict(orient='records')
    
    except Exception as e:
        logger.error(f"Error computing underperforming BSKs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refresh_data")
def refresh_data():
    """Refresh data from Parquet files."""
    try:
        logger.info("Refreshing data...")
        init_cache(force_reload=True)
        stats = get_cache_stats()
        
        return {
            "status": "success",
            "message": "Data refreshed successfully",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
