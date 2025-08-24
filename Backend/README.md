# Image Deblurring Backend

This backend provides an API for image deblurring using either a Hugging Face pretrained model (if configured) or a local Richardson–Lucy deconvolution algorithm.

## Features
- `/deblur` endpoint: POST an image file, get a deblurred image in response.
- Uses Hugging Face Inference API if `HF_MODEL` and `HF_TOKEN` environment variables are set.
- Otherwise, falls back to a local algorithm (no GPU required).

## Usage

### Install dependencies
```powershell
pip install -r requirements.txt
```

### Run the server
```powershell
python main.py
```

### Environment variables (optional)
- `HF_MODEL`: Hugging Face model name (e.g. "saikatdebnath/image-deblurring")
- `HF_TOKEN`: Your Hugging Face API token

If these are set, the backend will use the Hugging Face model for deblurring.

### API
- `POST /deblur` with form-data `file`: image file
- Returns: deblurred image (PNG)

## Example curl
```powershell
curl -X POST "http://localhost:8000/deblur" -F "file=@blurred.png" --output sharp.png
```

## Notes
- The local algorithm uses Richardson–Lucy deconvolution and works for simple motion blur.
- For best results, use a Hugging Face model.
