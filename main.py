import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load models
model_vgg16 = load_model('input/vgg16_model.h5')
model_vgg19 = load_model('input/vgg19_model.h5')
model_resnet50 = load_model('input/resnet50_model.h5')

# Print model summaries for debugging
print("VGG16 Model Summary")
model_vgg16.summary()

print("VGG19 Model Summary")
model_vgg19.summary()

print("ResNet50 Model Summary")
model_resnet50.summary()

def prepare_image(img):
    img = img.resize((224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def get_prediction(model, img):
    prediction = model.predict(img)
    prediction_value = float(prediction[0][0])
    result = "Healthy" if prediction_value >= 0.5 else "Cancer"
    return prediction_value, result

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = prepare_image(img)

        # Get predictions from all models
        prediction_vgg16, result_vgg16 = get_prediction(model_vgg16, img)
        prediction_vgg19, result_vgg19 = get_prediction(model_vgg19, img)
        prediction_resnet50, result_resnet50 = get_prediction(model_resnet50, img)

        # Debugging output
        print(f"VGG16 - Prediction Value: {prediction_vgg16}, Result: {result_vgg16}")
        print(f"VGG19 - Prediction Value: {prediction_vgg19}, Result: {result_vgg19}")
        print(f"ResNet50 - Prediction Value: {prediction_resnet50}, Result: {result_resnet50}")

        return jsonify({
            "VGG16": {
                "prediction": prediction_vgg16,
                "result": result_vgg16
            },
            "VGG19": {
                "prediction": prediction_vgg19,
                "result": result_vgg19
            },
            "ResNet50": {
                "prediction": prediction_resnet50,
                "result": result_resnet50
            }
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
