import ee
import time

RELATIVE_KEY_PATH = 'private-key.json'

def initialize_ee():
  PROJECT_ID = "wefgis"

  print("Starting GEE")
  t0 = time.time()

  try:
    import json
    with open(RELATIVE_KEY_PATH) as f:
        key_data = json.load(f)
        service_account = key_data['client_email']

    # 3. Authenticate and Initialize
    credentials = ee.ServiceAccountCredentials(service_account, RELATIVE_KEY_PATH)
    ee.Initialize(credentials)
    print("Earth Engine initialized successfully!")

  except Exception as e:
    print("EE: Login required. Opening browser...")
    ee.Authenticate()
    ee.Initialize()

initialize_ee()

ROI = ee.Geometry.Rectangle([113.79, -2.13, 113.96, -2.31])

SCALE = 30
MAX_PIXELS = 1e13