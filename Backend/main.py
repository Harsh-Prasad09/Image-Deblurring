from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from torch import nn
from PIL import Image
import io
import torchvision.transforms as transforms
import numpy as np
from waitress import serve
import os

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
def _load_checkpoint_into_model(model, checkpoint_path: str):
    """Attempt to load a PyTorch checkpoint into `model` in a robust way.

    Handles checkpoints that are either a state_dict, a dict with 'state_dict',
    or a state_dict saved with a 'module.' prefix (from DataParallel).
    Returns a tuple (model, info_str).
    """
    # Ensure the checkpoint file exists and provide a helpful error if not
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = None

    # If checkpoint is a dict with possible keys
    if isinstance(ckpt, dict):
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            # It may already be a state_dict
            state_dict = ckpt
    elif isinstance(ckpt, nn.Module):
        # saved entire model
        return ckpt, f"loaded full model object from {checkpoint_path}"
    else:
        state_dict = ckpt

    # Strip 'module.' prefix if present
    new_state = {}
    for k, v in state_dict.items():
        new_key = k
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        new_state[new_key] = v

    # Load with strict=False to allow mismatch (safer across architectures)
    load_info = model.load_state_dict(new_state, strict=False)
    return model, f"loaded state_dict from {checkpoint_path}, missing_keys={load_info.missing_keys}, unexpected_keys={load_info.unexpected_keys}"


# Primary checkpoint path to use for both endpoints (resolved relative to this file)
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'nafnet_checkpoint.pth')

# Helpful startup log showing the resolved checkpoint path
try:
    print(f"[INFO] Resolved CHECKPOINT_PATH = {os.path.abspath(CHECKPOINT_PATH)}")
except Exception:
    pass

# Lazy-loaded model instances (loaded on first request)
general_model = None
text_model = None

def load_general_model():
    global general_model
    if general_model is None:
        model = ConvAutoencoder()
        model, info = _load_checkpoint_into_model(model, CHECKPOINT_PATH)
        try:
            print(f"[INFO] general model load: {info}")
        except Exception:
            pass
        model.eval()
        general_model = model
    return general_model

def load_text_model():
    global text_model
    if text_model is None:
        model = ConvAutoencoder()
        model, info = _load_checkpoint_into_model(model, CHECKPOINT_PATH)
        try:
            print(f"[INFO] text model load: {info}")
        except Exception:
            pass
        model.eval()
        text_model = model
    return text_model

# Define the transformation for the input image
general_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# -------------------- PyTorch Model for Text Section --------------------

# For the text section we will use the same PyTorch architecture and checkpoint
# as the general section. This keeps a single source of truth for weights
# (the provided nafnet_checkpoint.pth). We create a separate model instance
# in case concurrent requests modify internal state.
text_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

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

    # Ensure model is loaded
    try:
        model = load_general_model()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'Failed to load model: ' + str(e)}), 500

    # Pass the image through the model
    with torch.no_grad():
        reconstructed_image = model(input_image)

    # Convert the reconstructed image to a PIL image
    reconstructed_image = reconstructed_image.squeeze(0).permute(1, 2, 0).numpy()
    reconstructed_image = (reconstructed_image * 255).clip(0, 255).astype('uint8')
    reconstructed_image = Image.fromarray(reconstructed_image)

    # Save the reconstructed image to a BytesIO object
    img_io = io.BytesIO()
    reconstructed_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

# Endpoint for text section (PyTorch model)
@app.route('/predict/text', methods=['POST'])
def predict_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = Image.open(file).convert('RGB')

    # Preprocess the image to match the PyTorch model's expected input
    input_image = text_transform(image).unsqueeze(0)

    # Ensure model is loaded
    try:
        model = load_text_model()
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'Failed to load model: ' + str(e)}), 500

    # Pass the image through the model
    with torch.no_grad():
        output_tensor = model(input_image)

    # Convert tensor to PIL image
    output_array = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    output_array = (output_array * 255).clip(0, 255).astype('uint8')
    output_image = Image.fromarray(output_array)

    # Save the output image to a BytesIO object
    img_io = io.BytesIO()
    output_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

# -------------------- Run the Flask App --------------------
if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5000, debug=True)
    serve(app, host='0.0.0.0', port=8080)