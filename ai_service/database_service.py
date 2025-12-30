# ai_service/database_service.py
"""
Database service utilities for fetching data for service recommendations.
Updated to use Parquet files via data_loader instead of PostgreSQL.
"""

import pandas as pd
from typing import Optional, Dict
import sys
import os

# Add backend to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

try:
    from app.data_loader import (
        get_service_master_df,
        get_bsk_master_df,
        get_deo_master_df,
        get_provisions_df,
        is_initialized,
        init_cache
    )
    DATALOADER_AVAILABLE = True
    print("✅ Data loader imported successfully")
except ImportError as e:
    DATALOADER_AVAILABLE = False
    print(f"Warning: Could not import data loader: {e}")


def ensure_data_loaded():
    """Ensure data is loaded before accessing."""
    if not DATALOADER_AVAILABLE:
        raise RuntimeError("Data loader not available. Check import paths.")
    
    if not is_initialized():
        print("Data not initialized. Initializing now...")
        init_cache()


def fetch_services_from_db(include_inactive: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch services data from Parquet files.
    
    Args:
        include_inactive: Whether to include inactive services
        
    Returns:
        DataFrame with service data or None if error
    """
    if not DATALOADER_AVAILABLE:
        print("Data loader not available")
        return None
    
    try:
        ensure_data_loaded()
        
        # Get services DataFrame
        services_df = get_service_master_df()
        
        if services_df.empty:
            print("No services found in data")
            return None
        
        # Filter active services if requested
        if not include_inactive and 'is_active' in services_df.columns:
            services_df = services_df[services_df['is_active'] == True].copy()
        
        # Ensure required columns exist with defaults
        required_cols = {
            'service_id': 0,
            'service_name': '',
            'service_type': '',
            'service_desc': '',
            'common_name': '',
            'department_name': '',
            'department_id': 0,
            'how_to_apply': '',
            'eligibility_criteria': '',
            'required_doc': '',
            'is_active': True,
            'is_paid_service': False
        }
        
        for col, default in required_cols.items():
            if col not in services_df.columns:
                services_df[col] = default
            else:
                services_df[col] = services_df[col].fillna(default)
        
        print(f"✅ Loaded {len(services_df)} services")
        return services_df
        
    except Exception as e:
        print(f"Error fetching services: {e}")
        return None


def fetch_bsks_from_db(include_inactive: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch BSK data from Parquet files.
    
    Args:
        include_inactive: Whether to include inactive BSKs
        
    Returns:
        DataFrame with BSK data or None if error
    """
    if not DATALOADER_AVAILABLE:
        print("Data loader not available")
        return None
    
    try:
        ensure_data_loaded()
        
        # Get BSK DataFrame
        bsk_df = get_bsk_master_df()
        
        if bsk_df.empty:
            print("No BSKs found in data")
            return None
        
        # Filter active BSKs if requested
        if not include_inactive and 'is_active' in bsk_df.columns:
            bsk_df = bsk_df[bsk_df['is_active'] == True].copy()
        
        # Ensure required columns exist with defaults
        required_cols = {
            'bsk_id': 0,
            'bsk_name': '',
            'bsk_code': '',
            'district_name': '',
            'district_id': 0,
            'block_municipalty_name': '',
            'bsk_lat': '0.0',
            'bsk_long': '0.0',
            'bsk_address': '',
            'is_active': True,
            'no_of_deos': 0
        }
        
        for col, default in required_cols.items():
            if col not in bsk_df.columns:
                bsk_df[col] = default
            else:
                bsk_df[col] = bsk_df[col].fillna(default)
        
        print(f"✅ Loaded {len(bsk_df)} BSKs")
        return bsk_df
        
    except Exception as e:
        print(f"Error fetching BSKs: {e}")
        return None


def fetch_deos_from_db(include_inactive: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch DEO data from Parquet files.
    
    Args:
        include_inactive: Whether to include inactive DEOs
        
    Returns:
        DataFrame with DEO data or None if error
    """
    if not DATALOADER_AVAILABLE:
        print("Data loader not available")
        return None
    
    try:
        ensure_data_loaded()
        
        # Get DEO DataFrame
        deo_df = get_deo_master_df()
        
        if deo_df.empty:
            print("No DEOs found in data")
            return None
        
        # Filter active DEOs if requested
        if not include_inactive and 'is_active' in deo_df.columns:
            deo_df = deo_df[deo_df['is_active'] == True].copy()
        
        # Ensure required columns exist with defaults
        required_cols = {
            'agent_id': 0,
            'user_name': '',
            'agent_code': '',
            'agent_email': '',
            'agent_phone': '',
            'bsk_id': 0,
            'bsk_name': '',
            'date_of_engagement': '',
            'bsk_post': '',
            'is_active': True
        }
        
        for col, default in required_cols.items():
            if col not in deo_df.columns:
                deo_df[col] = default
            else:
                deo_df[col] = deo_df[col].fillna(default)
        
        print(f"✅ Loaded {len(deo_df)} DEOs")
        return deo_df
        
    except Exception as e:
        print(f"Error fetching DEOs: {e}")
        return None


def fetch_provisions_from_db(limit: Optional[int] = None) -> Optional[pd.DataFrame]:
    """
    Fetch provisions data from Parquet files.
    
    Args:
        limit: Maximum number of records to fetch (None for all)
        
    Returns:
        DataFrame with provisions data or None if error
    """
    if not DATALOADER_AVAILABLE:
        print("Data loader not available")
        return None
    
    try:
        ensure_data_loaded()
        
        # Get provisions DataFrame
        provisions_df = get_provisions_df()
        
        if provisions_df.empty:
            print("No provisions found in data")
            return None
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            provisions_df = provisions_df.head(limit)
        
        # Ensure required columns exist with defaults
        required_cols = {
            'bsk_id': 0,
            'bsk_name': '',
            'customer_id': '',
            'customer_name': '',
            'customer_phone': '',
            'service_id': 0,
            'service_name': '',
            'prov_date': '',
            'docket_no': ''
        }
        
        for col, default in required_cols.items():
            if col not in provisions_df.columns:
                provisions_df[col] = default
            else:
                provisions_df[col] = provisions_df[col].fillna(default)
        
        print(f"✅ Loaded {len(provisions_df)} provisions")
        return provisions_df
        
    except Exception as e:
        print(f"Error fetching provisions: {e}")
        return None


def fetch_all_data_for_recommendations(include_inactive: bool = False) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetch all data needed for service recommendations.
    
    Args:
        include_inactive: Whether to include inactive records
        
    Returns:
        Dictionary with DataFrames for services, bsks, deos, and provisions
    """
    print("Fetching all data from Parquet files...")
    
    data = {
        'services_df': fetch_services_from_db(include_inactive),
        'bsk_df': fetch_bsks_from_db(include_inactive),
        'deos_df': fetch_deos_from_db(include_inactive),
        'provisions_df': fetch_provisions_from_db()
    }
    
    # Print summary
    for key, df in data.items():
        if df is not None:
            print(f"Loaded {len(df)} records for {key}")
        else:
            print(f"Failed to load {key}")
    
    return data


def test_database_connection() -> bool:
    """Test if data loading is working."""
    try:
        ensure_data_loaded()
        
        # Try to fetch services
        services_df = fetch_services_from_db()
        if services_df is not None:
            print(f"✅ Data connection successful. Found {len(services_df)} services.")
            return True
        else:
            print("❌ Data connection failed: no services found")
            return False
            
    except Exception as e:
        print(f"❌ Data connection failed: {e}")
        return False
