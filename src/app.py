import ee
import geemap

# 1. Initialize and authenticate Google Earth Engine
try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# 2. Define the Area of Interest (AOI) - e.g., Paris, France
# Format: ee.Geometry.Point([longitude, latitude])
aoi = ee.Geometry.Point([2.3522, 48.8566]).buffer(10000) 

# 3. Load Sentinel-2 Satellite Imagery and filter it
image = (
    ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterBounds(aoi)
    .filterDate('2023-01-01', '2023-12-31')
    # Filter for low cloud cover
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .median() # Take the median pixel value to get a cloud-free composite
    .clip(aoi)
)

# Select the spectral bands we want to use for classification
bands = ['B2', 'B3', 'B4', 'B8'] # Blue, Green, Red, Near-Infrared

# 4. Define Training Data (Manually creating sample points)
# In a real scenario, you would draw these geometries or upload a shapefile.
# We assign a 'landcover' property: 0 = Water, 1 = Urban, 2 = Vegetation

# Coordinates below are placeholders; you'll adjust these based on your region
water_points = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([2.345, 48.855]), {'landcover': 0}),
])
urban_points = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([2.352, 48.860]), {'landcover': 1}),
])
veg_points = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([2.450, 48.830]), {'landcover': 2}),
])

# Combine all training points into one dataset
training_features = water_points.merge(urban_points).merge(veg_points)

# 5. Sample the satellite image at the training point locations
training_data = image.select(bands).sampleRegions(
    collection=training_features,
    properties=['landcover'],
    scale=10 # Sentinel-2 resolution is 10 meters per pixel
)

# 6. Train the Machine Learning Classifier (Random Forest)
classifier = ee.Classifier.smileRandomForest(numberOfTrees=50).train(
    features=training_data,
    classProperty='landcover',
    inputProperties=bands
)

# 7. Classify the entire image
classified_image = image.select(bands).classify(classifier)

# 8. Visualize the Results using an interactive map
Map = geemap.Map()
Map.centerObject(aoi, 12)

# Add the original satellite image (True Color)
vis_params = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
Map.addLayer(image, vis_params, 'Original Satellite Image')

# Add the classified image
# 0 (Water) -> Blue, 1 (Urban) -> Red, 2 (Vegetation) -> Green
class_vis = {'min': 0, 'max': 2, 'palette': ['blue', 'red', 'green']}
Map.addLayer(classified_image, class_vis, 'Land Cover Classification')

# Display the map
Map
