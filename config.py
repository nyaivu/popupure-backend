import ee
import time
import os
import json
from dotenv import load_dotenv

load_dotenv()

def initialize_ee():
    # Gunakan ID Project kamu
    PROJECT_ID = "wefgis" 
    print("Starting GEE...")
    t0 = time.time()
    
    try:
        # BEST PRACTICE: ee.Initialize() tanpa argumen credentials 
        # akan otomatis mencari file di path yang ada pada variabel 
        # GOOGLE_APPLICATION_CREDENTIALS
        ee.Initialize(project=PROJECT_ID)
        
        print(f"Earth Engine initialized successfully in {time.time()-t0:.2f}s")
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