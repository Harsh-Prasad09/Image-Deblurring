from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from torch import nn
from PIL import Image
import io
import torchvision.transforms as transforms
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Activation
import numpy as np
from waitress import serve

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# -------------------- PyTorch Model for General Section --------------------

# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load the PyTorch model
general_model = ConvAutoencoder()
general_model.load_state_dict(torch.load('TextModelWeights.pth', map_location=torch.device('cpu')))
general_model.eval()

# Define the transformation for the input image
general_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# -------------------- TensorFlow Model for Text Section --------------------

# Define the TensorFlow generator model
def generator_model():
    inputs = Input(shape=(256, 256, 3))
    x = Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    outputs = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(x)
    model = Model(inputs, outputs)
    return model

# Load the TensorFlow model
text_model_weights = "GeneralModelWeights.h5"
text_model = generator_model()
text_model.load_weights(text_model_weights)

# -------------------- API Endpoints --------------------

# Endpoint for general section (PyTorch model)
@app.route('/predict/general', methods=['POST'])
def predict_general():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = Image.open(file).convert('RGB')

    # Preprocess the image
    input_image = general_transform(image).unsqueeze(0)

    # Pass the image through the model
    with torch.no_grad():
        reconstructed_image = general_model(input_image)

    # Convert the reconstructed image to a PIL image
    reconstructed_image = reconstructed_image.squeeze(0).permute(1, 2, 0).numpy()
    reconstructed_image = (reconstructed_image * 255).clip(0, 255).astype('uint8')
    reconstructed_image = Image.fromarray(reconstructed_image)

    # Save the reconstructed image to a BytesIO object
    img_io = io.BytesIO()
    reconstructed_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

# Endpoint for text section (TensorFlow model)
@app.route('/predict/text', methods=['POST'])
def predict_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = Image.open(file).convert('RGB')

    # Preprocess the image
    img = image.resize((256, 256))
    img_array = np.array(img) / 127.5 - 1.0
    preprocessed_image = np.expand_dims(img_array, axis=0)

    # Pass the image through the model
    output_array = text_model.predict(preprocessed_image)

    # Postprocess the output image
    output_array = (output_array + 1.0) * 127.5
    output_array = np.clip(output_array, 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_array[0])

    # Save the output image to a BytesIO object
    img_io = io.BytesIO()
    output_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

# -------------------- Run the Flask App --------------------
if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5000, debug=True)
    serve(app, '0.0.0.0', port = 8080)