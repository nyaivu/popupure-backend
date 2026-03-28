from flask import Flask, jsonify, request 
from flask_restful import Resource, Api
from config import initialize_ee, ROI, SCALE, MAX_PIXELS
import ee

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

@app.route("/map/<city>")
def getCityMap(city):
    isKabupaten = request.args.get("kabupaten")

    cityWithSpace = city.replace('-', ' ')

    if isKabupaten:
        cityString = str.title(cityWithSpace)
    else:
        cityString = f'Kota {str.title(cityWithSpace)}'

    population = ee.Image("JRC/GHSL/P2023A/GHS_POP/1975")
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

    population = ee.Image("JRC/GHSL/P2023A/GHS_POP/1975")
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

@app.route("/population/all")
def population():
    population = ee.Image("JRC/GHSL/P2023A/GHS_POP/2020")
    indoCities = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'))

    populationStats = population.reduceRegions(
        collection = indoCities, 
        reducer = ee.Reducer.sum(), 
        scale = 100
    )
    cityStats = populationStats.select(['ADM2_NAME', 'sum'], retainGeometry=False).getInfo()
    features = cityStats.get('features', [])

    # 2. Use a list comprehension to extract only what you need
    clean_list = {
        "data": [
            {
            "id": f.get("id"),
            "city": f["properties"].get("ADM2_NAME"),
            "population": round(f["properties"].get("sum", 0))
        }
        for f in features
        ]
    } 

    return jsonify(clean_list)

if __name__ == '__main__':
    app.run(debug=True)