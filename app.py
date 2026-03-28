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

if __name__ == '__main__':
    app.run(debug=True)