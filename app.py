from flask import Flask, jsonify, request 
from flask_restful import Resource, Api
from config import initialize_ee, ROI, SCALE, MAX_PIXELS
import ee
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

app = Flask(__name__)
api = Api(app)

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

def scale(img):
    optical = img.select("SR_B.*").multiply(0.0000275).add(-0.2)
    return img.addBands(optical, None, True)


api.add_resource(Hello, '/')
@app.route("/get-stats")
def get_stats():
    start = request.args.get("start")
    end = request.args.get("end")

    if not start or not end:
        return jsonify({"error": "Please provide start and end date"}), 400
    
    img = scale(getLandsat(start, end).median()).clip(ROI)
    ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename("ndvi")
    stats = ndvi.reduceRegion(
        reducer = ee.Reducer.mean(),
        geometry = ROI,
        scale = SCALE,
        maxPixels = MAX_PIXELS
    )

    ndvi_value = stats.get("ndvi").getInfo()

    return jsonify({
        "start": start,
        "end": end,
        "mean_ndvi": ndvi_value
    })

@app.route("/list-cities")
def get_list_cities():
    # 1. Ambil dataset GAUL Level 2 khusus Indonesia
    indo_cities = ee.FeatureCollection("FAO/GAUL/2015/level2") \
        .filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'))
    
    # 2. Ambil hanya kolom ADM2_NAME dan urutkan
    # Kita gunakan aggregate_array untuk mengambil list nama kota langsung
    city_list = indo_cities.aggregate_array('ADM2_NAME').sort().getInfo()
    
    # 3. Hilangkan duplikat (jika ada) menggunakan set
    unique_cities = list(dict.fromkeys(city_list))
    
    # 4. Format data agar mudah dibaca frontend
    # Kamu bisa kirim dalam bentuk array of strings atau array of objects
    formatted_cities = []
    for city in unique_cities:
        # Menghilangkan awalan "Kota " atau "Kabupaten " untuk slug URL jika perlu
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

@app.route("/map/<city>")
def getCityMap(city):
    isKabupaten = request.args.get("kabupaten") == "true"

    cityWithSpace = city.replace('-', ' ')

    if isKabupaten:
        cityString = str.title(cityWithSpace)
    else:
        cityString = f'Kota {str.title(cityWithSpace)}'

    population = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020")
    indoCities = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', cityString))

    if indoCities.size().getInfo() == 0:
        return jsonify({
            "error": f"City '{cityString}' not found in the GAUL database.",
            "suggestion": "Check if it should be a 'kabupaten' or if the spelling matches 'Kota ...'"
        }), 404

    targetCity = indoCities.first()

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
    isKabupaten = request.args.get("kabupaten")

    cityWithSpace = city.replace('-', ' ')

    if isKabupaten:
        cityString = str.title(cityWithSpace)
    else:
        cityString = f'Kota {str.title(cityWithSpace)}'

    population = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020")
    indoCities = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', cityString))

    populationStats = population.reduceRegions(
        collection = indoCities, 
        reducer = ee.Reducer.sum(), 
        scale = 100
    )
    cityStats = populationStats.select(['ADM2_NAME', 'sum'], retainGeometry=False).getInfo()
    
    features = cityStats.get('features', [])
    currentFeature = features[0]

    # 2. Use a list comprehension to extract only what you need
    clean_list = {
        "data": {
            "id": currentFeature.get("id"),
            "city": currentFeature["properties"].get("ADM2_NAME"),
            "population": round(currentFeature["properties"].get("sum", 0))
        }
    } 

    return jsonify(clean_list)

@app.route("/correlation/air-quality/<city>")
def getAirCorrelation(city):
    isKabupaten = request.args.get("kabupaten")
    cityString = str.title(city.replace('-', ' ')) if isKabupaten else f'Kota {str.title(city.replace("-", " "))}'
    roi = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', cityString))
    
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
def classify_zone(city):
    # (Logika filter ROI GAUL tetap sama seperti sebelumnya)
    isKabupaten = request.args.get("kabupaten")
    cityString = str.title(city.replace('-', ' ')) if isKabupaten else f'Kota {str.title(city.replace("-", " "))}'
    roi = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM2_NAME', cityString))
    
    # 1. Ambil Data dari GEE
    pop = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020").clip(roi).unmask(0)
    no2 = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2").filterDate('2024-01-01', '2024-12-31').median().clip(roi).unmask(0)
    # NDVI sebagai penyeimbang (ruang hijau)
    landsat_raw = getLandsat('2023-01-01', '2024-12-31')
    
    # Cek jumlah gambar dalam koleksi
    count = landsat_raw.size().getInfo()

    if count > 0:
        # Jika ada gambar, baru lakukan median dan hitung NDVI
        img = landsat_raw.median().clip(roi)
        ndvi = img.normalizedDifference(["SR_B5", "SR_B4"]).rename('nd').unmask(0)
    else:
        # Jika kosong (karena awan < 20%), buat citra kosong bernilai 0
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
        geometries=True # WAJIB TRUE agar koordinat ikut terambil
    ).getInfo()

    if not samples_raw or 'features' not in samples_raw:
        return jsonify({"error": "Data GEE tidak tersedia untuk kota ini"}), 404

    features = []
    valid_features = [] # Simpan feature asli untuk koordinat
    
    for f in samples_raw['features']:
        # --- TAMBAHKAN PENGECEKAN GEOMETRY DI SINI ---
        if f is None or 'geometry' not in f or f['geometry'] is None:
            continue
            
        props = f.get('properties', {})
        # Pastikan data tidak None sebelum masuk ke AI
        p = props.get('population_count', 0)
        n = props.get('tropospheric_NO2_column_number_density', 0)
        v = props.get('nd', 0)
        
        # Tambahkan hanya jika data valid
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

    # Gabungkan koordinat
    points_data = []
    for i in range(len(features)):
        coords = valid_features[i]['geometry']['coordinates']
        points_data.append({
            "lng": coords[0],
            "lat": coords[1],
            "cluster": int(kmeans.labels_[i]),
            "pop": features[i][0], 
            "no2": features[i][1],
            "ndvi": features[i][2]
        })
    
    return jsonify({
        "city": cityString,
        "centers": kmeans.cluster_centers_.tolist(),
        "points": points_data
    })

if __name__ == '__main__':
    app.run(debug=True)