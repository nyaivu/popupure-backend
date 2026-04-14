import ee
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

def initialize_ee():
    PROJECT_ID = "wefgis"
    print("Starting GEE")
    t0 = time.time()
    gee_json_str = os.environ.get('GEE_JSON')

    # 1. Try Environment Variable
    if gee_json_str:
        try:
            key_data = json.loads(gee_json_str)
            credentials = ee.ServiceAccountCredentials(
                key_data['client_email'], 
                key_data=key_data['private_key']
            )
            ee.Initialize(credentials, project=PROJECT_ID)
            print(f"Earth Engine initialized via Environment Variable in {time.time()-t0:.2f}s")
            return True # <--- CRITICAL: Tell the script we succeeded
        except Exception as e:
            print(f"Failed to initialize with Env Var: {e}")
    
    # 2. Try Local Fallback (ADC)
    try:
        print("Env var not found/failed, attempting local file login...")
        ee.Initialize(project=PROJECT_ID)
        print("Earth Engine initialized via local default credentials")
        return True # <--- CRITICAL: Success
    except Exception as e:
        print(f"Local auth failed: {e}")
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