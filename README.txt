AYURVEDIC LEAF IDENTIFIER

Files
- ayurvedic_mobilenetv2.keras / .h5 : trained MobileNetV2 model (80 plants, about 93% test accuracy)
- class_names.json : plant names in the model output order
- plant_info.json : properties and traditional uses for each plant
- centroids.npy, config.json : data for the unknown-image check
- app.py : Flask server
- index.html : mobile web page

How to run on a laptop
1. Use Python 3.10 to 3.12. Run: pip install -r requirements.txt
2. Run: python app.py
3. Open http://localhost:5000
4. On a phone on the same Wi-Fi, open http://YOUR-LAPTOP-IP:5000

Notes
- Model input is a 224x224 RGB image, pixel values 0-255. Scaling is inside the model.
- The information is traditional and educational, not medical advice.
