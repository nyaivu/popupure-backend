from flask import Flask, jsonify, request 
from flask_restful import Resource, Api
from flask_caching import Cache
from config import initialize_ee, ROI, SCALE, MAX_PIXELS
import ee
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
cache = Cache(app)
api = Api(app)

BOUNDARY_DATASET = "FAO/GAUL/2015/level2" 

COL_NAME = "ADM2_NAME"
COL_COUNTRY = "ADM0_NAME"

@app.after_request
def add_header(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,ngrok-skip-browser-warning'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

class Hello(Resource):
    def get(self):
        initialize_ee()
        return jsonify({'message': "GEE initialized"})

def getLandsat(start, end):
    return (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(ROI)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
    )

def get_city_roi(cityName, is_kabupaten):
    city_string = str.title(cityName.replace('-', ' ')) if is_kabupaten else f'Kota {str.title(cityName.replace("-", " "))}'
    
    roi = ee.FeatureCollection(BOUNDARY_DATASET).filter(ee.Filter.eq(COL_NAME, city_string))
    return roi

def get_cluster_metadata(center):
    pop, no2, ndvi = center
    
    # 1. Zona Merah (Bahaya)
    if no2 > 0.5 and pop > 0.5:
        return {
            "label": "Zona Merah (Kepadatan & Polusi Tinggi)",
            "description": "Area urban padat dengan tingkat emisi gas buang yang signifikan.",
            "color": "#ef4444"
        }
    
    # 2. Zona Oranye (Industri/Transportasi)
    if no2 > 0.5 and pop <= 0.5:
        return {
            "label": "Zona Industri/Transportasi",
            "description": "Polusi udara tinggi namun pemukiman jarang. Jalur lintas atau kawasan industri.",
            "color": "#f97316"
        }
    
    # 3. Zona Kuning (Pusat Keramaian)
    if pop > 2:
        return {
            "label": "Pusat Keramaian Utama",
            "description": "Area dengan konsentrasi penduduk sangat tinggi (Pusat Kota).",
            "color": "#fbbf24"
        }
    
    # 4. Zona Hijau (Asri)
    if ndvi > 0.2 or (no2 < 0 and pop < 0):
        return {
            "label": "Zona Hijau & Asri",
            "description": "Area dengan vegetasi baik dan tingkat polusi di bawah rata-rata.",
            "color": "#22c55e"
        }
    
    return {
        "label": "Zona Penyangga (Buffer)",
        "description": "Area transisi dengan karakteristik pemukiman dan polusi moderat.",
        "color": "#3b82f6"
    }

@app.route("/list-cities")
@cache.memoize(timeout=86400)
def get_list_cities():
    indo_cities = ee.FeatureCollection(BOUNDARY_DATASET) \
        .filter(ee.Filter.eq(COL_COUNTRY, 'Indonesia'))
    
    city_list = indo_cities.aggregate_array(COL_NAME).sort().getInfo()
    
    unique_cities = list(dict.fromkeys(city_list))
    
    formatted_cities = []
    for city in unique_cities:
        is_kab = "Kota" not in city
        slug = city.lower().replace('kota ', '').replace(' ', '-')
        formatted_cities.append({
            "name": city,
            "slug": slug,
            "is_kabupaten": is_kab
        })

    return jsonify({
        "total": len(formatted_cities),
        "cities": formatted_cities
    })

@app.route("/search-city")
def search_city():
    query = request.args.get("q", "").title()
    indo_cities = ee.FeatureCollection(BOUNDARY_DATASET).filter(ee.Filter.eq(COL_COUNTRY, 'Indonesia'))
    
    # Filter kota yang mengandung kata kunci
    filtered = indo_cities.filter(ee.Filter.stringContains(COL_NAME, query))
    results = filtered.aggregate_array(COL_NAME).getInfo()
    
    return jsonify({"results": results})

@app.route("/geojson/<city>")
def get_city_geojson(city):
    is_kabupaten = request.args.get("kabupaten") == "true"

    roi = get_city_roi(city, is_kabupaten)
    
    # Mengambil format GeoJSON untuk dirender oleh L.geoJSON di React
    geojson_data = roi.getInfo()
    
    return jsonify(geojson_data)

@app.route("/map/all-boundaries")
def get_all_boundaries():
    # Ambil semua kabupaten/kota di Indonesia
    all_indo = ee.FeatureCollection(BOUNDARY_DATASET) \
                 .filter(ee.Filter.eq(COL_COUNTRY, 'Indonesia'))
    
    # Beri warna outline saja
    empty = ee.Image().byte()
    outline = empty.paint(featureCollection=all_indo, color=1, width=1)
    
    map_id = outline.getMapId({'palette': '0000FF'}) # Garis biru
    return jsonify({'tile_url': map_id['tile_fetcher'].url_format})

@app.route("/find-city")
def find_city():
    lat = float(request.args.get('lat'))
    lng = float(request.args.get('lng'))
    point = ee.Geometry.Point([lng, lat])
    
    # Cari feature yang memuat titik tersebut
    match = ee.FeatureCollection("FAO/GAUL/2015/level2") \
              .filterBounds(point).first()
    
    if not match.getInfo(): return jsonify({"error": "Luar wilayah"}), 404

    
    name = match.get('ADM2_NAME').getInfo()
    slug = name.lower().replace('kota ', '').replace(' ', '-')
    return jsonify({
        "name": name,
        "slug": slug,
        "is_kabupaten": "Kota" not in name
    })

@app.route("/map/<city>")
def get_city_map(city):
    is_kabupaten = request.args.get("kabupaten") == "true"

    cityWithSpace = city.replace('-', ' ')

    if is_kabupaten:
        cityString = str.title(cityWithSpace)
    else:
        cityString = f'Kota {str.title(cityWithSpace)}'

    population = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020")
    indo_cities = ee.FeatureCollection(BOUNDARY_DATASET).filter(ee.Filter.eq(COL_NAME, cityString))

    if indo_cities.size().getInfo() == 0:
        return jsonify({
            "error": f"City '{cityString}' not found in the GAUL database.",
            "suggestion": "Check if it should be a 'kabupaten' or if the spelling matches 'Kota ...'"
        }), 404

    targetCity = indo_cities.first()

    if targetCity is None:
        return jsonify({"error": f"City {cityString} not found"}), 404

    centroid = targetCity.geometry().centroid()
    coords = centroid.getInfo()['coordinates']
    zoom_level = 11

    vis_params = {
        'min': 0,
        'max': 100,
        'palette': ['f7f7f7', 'cccccc', '969696', '525252', '000000']
    }

    masked_population = population.updateMask(population.gt(0))

    map_data = masked_population.getMapId(vis_params)

    return jsonify({
        "city": cityString,
        "lat": coords[1],
        "lng": coords[0],
        'tile_url': map_data['tile_fetcher'].url_format
    })

@app.route("/population/<city>")
def getCityPopulation(city):
    is_kabupaten = request.args.get("kabupaten") == "true"

    cityWithSpace = city.replace('-', ' ')

    if is_kabupaten:
        cityString = str.title(cityWithSpace)
    else:
        cityString = f'Kota {str.title(cityWithSpace)}'

    population = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020")
    indo_cities = ee.FeatureCollection(BOUNDARY_DATASET).filter(ee.Filter.eq(COL_NAME, cityString))

    populationStats = population.reduceRegions(
        collection = indo_cities, 
        reducer = ee.Reducer.sum(), 
        scale = 100
    )
    cityStats = populationStats.select([COL_NAME, 'sum'], retainGeometry=False).getInfo()
    
    features = cityStats.get('features', [])
    currentFeature = features[0]

    # 2. Use a list comprehension to extract only what you need
    clean_list = {
        "data": {
            "id": currentFeature.get("id"),
            "city": currentFeature["properties"].get(COL_NAME),
            "population": round(currentFeature["properties"].get("sum", 0))
        }
    } 

    return jsonify(clean_list)

@app.route("/correlation/air-quality/<city>")
def getAirCorrelation(city):
    is_kabupaten = request.args.get("kabupaten") == "true"
    roi = get_city_roi(city, is_kabupaten)

    
    pop = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020").clip(roi).unmask(0)
    no2 = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2") \
        .filterDate('2024-01-01', '2024-12-31') \
        .select('tropospheric_NO2_column_number_density') \
        .median().clip(roi).unmask(0)

    stats = no2.addBands(pop).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi.geometry(),
        scale=1000,
        maxPixels=1e9
    ).getInfo()

    return jsonify({
        "pop_density": stats.get('population_count'),
        "no2_level": stats.get('tropospheric_NO2_column_number_density'),
        "status": "High Pollution" if stats.get('tropospheric_NO2_column_number_density', 0) > 0.0001 else "Clean Air"
    })



@app.route("/ai/classify-zone/<city>")
@cache.memoize(timeout=86400)
def classify_zone(city):
    is_kabupaten = request.args.get("kabupaten") == "true"
    cityString = str.title(city.replace('-', ' ')) if is_kabupaten else f'Kota {str.title(city.replace("-", " "))}'
    roi = get_city_roi(city, is_kabupaten)

    
    pop = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020").clip(roi).unmask(0)
    no2 = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2").filterDate('2024-01-01', '2024-12-31').median().clip(roi).unmask(0)

    landsat_raw = getLandsat('2023-01-01', '2024-12-31')
    
    count = landsat_raw.size().getInfo()

    if count > 0:
        img = landsat_raw.median().clip(roi)
        ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename('nd').unmask(0)
    else:
        ndvi = ee.Image.constant(0).rename('nd').clip(roi)

    combined = ee.Image.cat([
        pop.rename('population_count'),
        no2.select('tropospheric_NO2_column_number_density'),
        ndvi
    ])

    samples_raw = combined.sample(
        region=roi.geometry(),
        scale=1000, 
        numPixels=100,
        seed=42,
        geometries=True
    ).getInfo()

    if not samples_raw or 'features' not in samples_raw:
        return jsonify({"error": "Data GEE tidak tersedia untuk kota ini"}), 404

    features = []
    valid_features = [] 
    
    for f in samples_raw['features']:
        if f is None or 'geometry' not in f or f['geometry'] is None:
            continue
            
        props = f.get('properties', {})
        p = props.get('population_count', 0)
        n = props.get('tropospheric_NO2_column_number_density', 0)
        v = props.get('nd', 0)
        
        features.append([p, n, v])
        valid_features.append(f)

    if len(features) < 3:
        return jsonify({
            "error": "Data tidak cukup untuk clustering",
            "details": f"Hanya ditemukan {len(features)} titik dengan geometri valid.",
            "city": cityString
        }), 400

    # AI Process
    df_features = np.array(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(scaled_features)

    cluster_centers = kmeans.cluster_centers_.tolist()
    cluster_metadata = []
    for center in cluster_centers:
        cluster_metadata.append({
            "scores": center,
            "metadata": get_cluster_metadata(center)
        })

    # Gabungkan koordinat
    points_data = []
    for i in range(len(features)):
        cluster_idx = int(kmeans.labels_[i])
        assigned_color = cluster_metadata[cluster_idx]["metadata"]["color"]
        
        coords = valid_features[i]['geometry']['coordinates']
        points_data.append({
            "lng": coords[0],
            "lat": coords[1],
            "cluster": cluster_idx,
            "color": assigned_color, 
            "pop": features[i][0], 
            "no2": features[i][1],
            "ndvi": features[i][2]
        })
    
    return jsonify({
        "city": cityString,
        "centers": cluster_metadata,
        "points": points_data
    })

@app.route("/ai/compare")
def compare_cities():
    city1 = request.args.get("city1")
    city2 = request.args.get("city2")
    is_kab1 = request.args.get("kab1") == "true"
    is_kab2 = request.args.get("kab2") == "true"

    if not city1 or not city2:
        return jsonify({"error": "Harap pilih dua kota untuk dibandingkan"}), 400

    def get_city_data(city_slug, is_kabupaten):
        # Logika penentuan cityString sama seperti sebelumnya
        name = city_slug.replace('-', ' ').title()
        city_string = name if is_kabupaten else f"Kota {name}"
        
        roi = get_city_roi(city_slug, is_kabupaten)

        
        # Ambil data rata-rata
        pop = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020").clip(roi).unmask(0)
        no2 = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2") \
                .filterDate('2023-01-01', '2024-12-31').median().clip(roi).unmask(0)
        
        stats = no2.addBands(pop).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi.geometry(),
            scale=1000,
            maxPixels=1e9
        ).getInfo()
        
        return {
            "name": city_string,
            "population_avg": stats.get('population_count', 0),
            "no2_avg": stats.get('tropospheric_NO2_column_number_density', 0)
        }

    try:
        data1 = get_city_data(city1, is_kab1)
        data2 = get_city_data(city2, is_kab2)
        
        return jsonify({
            "comparison": [data1, data2],
            "timestamp": "2024"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)