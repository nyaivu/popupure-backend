import ee
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

def initialize_ee():
    PROJECT_ID = "wefgis" 
    print("Starting GEE...")
    try:
        path = "/app/private-key.json"
        # Memaksa inisialisasi menggunakan file fisik
        creds = ee.ServiceAccountCredentials('', key_file=path)
        ee.Initialize(creds, project=PROJECT_ID)
        print("GEE initialized successfully!")
        return True
    except Exception as e:
        print(f"GEE Initialization failed: {e}")
        return False

# --- EXECUTION FLOW ---

# Run the initialization and capture the result
success = initialize_ee()

if not success:
    # This only runs if BOTH the env var and local login failed
    raise RuntimeError("Earth Engine failed to initialize. Check your GEE_JSON variable.")

ROI = ee.Geometry.Rectangle([113.79, -2.13, 113.96, -2.31])
SCALE = 30
MAX_PIXELS = 1e13