# backend/app/data_loader_hf.py
"""
Alternative data loader using official huggingface_hub library.
This is more reliable for authenticated downloads.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import huggingface_hub
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not installed. Run: pip install huggingface-hub")

# --- HuggingFace configuration (Pulled from Environment) ---
# Hugging Face Spaces uses HF_TOKEN, but we also support HUGGINGFACE_TOKEN for local dev
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "coe5332/bsk-database")

# File names in the HuggingFace dataset
PARQUET_FILES = {
    "block_municipality": "dbo_ml_block_municipality.parquet",
    "bsk_master": "dbo_ml_bsk_master.parquet",
    "citizen_master": "dbo_ml_citizen_master_v2.parquet",
    "deo_master": "dbo_ml_deo_master.parquet",
    "department_master": "dbo_ml_department_master.parquet",
    "district": "dbo_ml_district.parquet",
    "gp_ward_master": "dbo_ml_gp_ward_master.parquet",
    "post_office_master": "dbo_ml_post_office_master.parquet",
    "provision": "dbo_ml_provision.parquet",
    "service_master": "dbo_ml_service_master.parquet",
    "sub_division": "dbo_ml_sub_division.parquet",
}

# Local cache directory
_LOCAL_CACHE_DIR = Path(os.getenv("PARQUET_CACHE_DIR", "./.parquet_cache"))
_LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Global cache for loaded DataFrames
_DFS_CACHE: Dict[str, pd.DataFrame] = {}
_INITIALIZED = False


def download_from_huggingface(filename: str) -> Path:
    """Download a file from HuggingFace dataset using official hub library."""
    if not HF_HUB_AVAILABLE:
        raise RuntimeError(
            "huggingface_hub library not installed. "
            "Install with: pip install huggingface-hub"
        )
    
    logger.info(f"Downloading {filename} from HuggingFace dataset: {HF_DATASET_REPO}")
    
    try:
        # Download using official hub library (handles auth automatically)
        local_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN,
            cache_dir=str(_LOCAL_CACHE_DIR),
            local_dir=str(_LOCAL_CACHE_DIR),
            local_dir_use_symlinks=False  # Copy file instead of symlink
        )
        
        logger.info(f"✅ Downloaded successfully: {local_path}")
        return Path(local_path)
        
    except Exception as e:
        if "401" in str(e) or "Unauthorized" in str(e):
            logger.error("❌ Authentication failed!")
            logger.error(f"   Please check your HUGGINGFACE_TOKEN in .env file")
        elif "403" in str(e) or "Forbidden" in str(e):
            logger.error("❌ Access forbidden!")
            logger.error(f"   Your token may not have access to dataset: {HF_DATASET_REPO}")
        else:
            logger.error(f"❌ Failed to download {filename}: {e}")
        raise


def load_parquet_from_huggingface(table_name: str) -> pd.DataFrame:
    """Load a Parquet file from HuggingFace into DataFrame."""
    if table_name not in PARQUET_FILES:
        raise ValueError(f"Unknown table: {table_name}. Available: {list(PARQUET_FILES.keys())}")
    
    filename = PARQUET_FILES[table_name]
    
    # Check if already cached locally
    local_path = _LOCAL_CACHE_DIR / filename
    
    if not local_path.exists():
        # Download from HuggingFace
        local_path = download_from_huggingface(filename)
    else:
        logger.info(f"Using cached file: {local_path}")
    
    # Load using pyarrow (fast)
    logger.info(f"Loading Parquet file: {local_path}")
    df = pd.read_parquet(local_path, engine="pyarrow")
    
    return df


def normalize_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Normalize DataFrame columns (types, dates, etc.)."""
    df = df.copy()
    
    # Parse date columns
    date_columns = ["prov_date", "date_of_engagement", "dob"]
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Ensure numeric ID columns
    id_columns = ["bsk_id", "service_id", "district_id", "agent_id", 
                  "department_id", "block_mun_id", "gp_id", "sub_div_id"]
    for col in id_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Handle boolean columns
    bool_columns = ["is_active", "is_paid_service", "is_aadhar_center"]
    for col in bool_columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    
    # Fill NaN in string columns with empty string
    string_columns = df.select_dtypes(include=["object"]).columns
    for col in string_columns:
        if col not in date_columns:
            df[col] = df[col].fillna("")
    
    logger.info(f"Normalized {name}: shape={df.shape}")
    return df


def load_all() -> Dict[str, pd.DataFrame]:
    """Load all Parquet files into memory from HuggingFace."""
    dfs: Dict[str, pd.DataFrame] = {}
    
    logger.info(f"Loading {len(PARQUET_FILES)} Parquet files from HuggingFace...")
    logger.info(f"Dataset: {HF_DATASET_REPO}")
    
    for table_name, filename in PARQUET_FILES.items():
        try:
            logger.info(f"Loading {table_name}...")
            df = load_parquet_from_huggingface(table_name)
            df = normalize_dataframe(df, table_name)
            dfs[table_name] = df
            logger.info(f"✅ {table_name}: {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"❌ Failed to load {table_name}: {e}")
            logger.error(f"   This table will be unavailable. Continuing with other tables...")
            dfs[table_name] = pd.DataFrame()
    
    return dfs


def init_cache(force_reload: bool = False):
    """Initialize the global DataFrame cache."""
    global _DFS_CACHE, _INITIALIZED
    
    if _INITIALIZED and not force_reload:
        logger.info("Data already loaded. Use force_reload=True to reload.")
        return
    
    if not HF_TOKEN:
        logger.warning("⚠️ HUGGINGFACE_TOKEN not set in .env file!")
    
    logger.info("=" * 60)
    logger.info("Initializing BSK Data Loader (HuggingFace)")
    logger.info("=" * 60)
    
    _DFS_CACHE = load_all()
    _INITIALIZED = True
    
    total_rows = sum(len(df) for df in _DFS_CACHE.values())
    logger.info("=" * 60)
    logger.info(f"✅ Data loaded successfully: {len(_DFS_CACHE)} tables, {total_rows:,} total rows")
    logger.info("=" * 60)


def get_df(name: str) -> pd.DataFrame:
    """Get a DataFrame by name."""
    if not _INITIALIZED:
        raise RuntimeError("Data not initialized. Call init_cache() at startup.")
    
    if name not in _DFS_CACHE:
        logger.warning(f"DataFrame '{name}' not found. Returning empty.")
        return pd.DataFrame()
    
    return _DFS_CACHE[name].copy()


# Convenience getters
def get_provisions_df() -> pd.DataFrame: return get_df("provision")
def get_bsk_master_df() -> pd.DataFrame: return get_df("bsk_master")
def get_service_master_df() -> pd.DataFrame: return get_df("service_master")
def get_deo_master_df() -> pd.DataFrame: return get_df("deo_master")
def get_block_municipality_df() -> pd.DataFrame: return get_df("block_municipality")
def get_citizen_master_df() -> pd.DataFrame: return get_df("citizen_master")
def get_department_master_df() -> pd.DataFrame: return get_df("department_master")
def get_district_df() -> pd.DataFrame: return get_df("district")
def get_sub_division_df() -> pd.DataFrame: return get_df("sub_division")

def is_initialized() -> bool:
    return _INITIALIZED

def get_cache_stats() -> Dict[str, any]:
    """Get statistics about cached data."""
    if not _INITIALIZED:
        return {"initialized": False}
    
    stats = {
        "initialized": True,
        "tables": {},
        "total_rows": 0,
        "cache_dir": str(_LOCAL_CACHE_DIR),
        "hf_token_set": bool(HF_TOKEN),
        "hf_dataset": HF_DATASET_REPO,
    }
    
    for name, df in _DFS_CACHE.items():
        stats["tables"][name] = {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": df.memory_usage(deep=True).sum() / (1024 ** 2),
        }
        stats["total_rows"] += len(df)
    
    return stats