import os
import io
import tempfile
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
from PIL import Image
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Image Deblurring Backend')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


def call_hf_inference(model: str, token: str, image_bytes: bytes, timeout: int = 120) -> bytes:
    """
    Send raw image bytes to the Hugging Face Inference API for an image-to-image model.
    Returns raw bytes of response (expected image) or raises HTTPException on error.
    """
    url = f'https://api-inference.huggingface.co/models/{model}'
    headers = {'Authorization': f'Bearer {token}'}
    logger.info('Calling HF model %s', model)
    try:
        resp = requests.post(url, headers=headers, data=image_bytes, timeout=timeout)
    except requests.RequestException as e:
        logger.exception('HF request failed')
        raise HTTPException(status_code=502, detail=f'Hugging Face request failed: {e}')

    if resp.status_code != 200:
        # HF returns JSON error messages sometimes
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        logger.error('HF model error: %s', err)
        raise HTTPException(status_code=502, detail={'status_code': resp.status_code, 'error': err})

    return resp.content


def motion_psf(length: int = 15, angle: float = 0) -> np.ndarray:
    # create a simple linear motion blur PSF
    EPS = 1e-8
    size = length
    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    angle_rad = np.deg2rad(angle)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    for i in range(size):
        x = center + (i - center) * cos_a
        y = center + (i - center) * sin_a
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= ix < size and 0 <= iy < size:
            psf[iy, ix] = 1
    s = psf.sum()
    if s < EPS:
        psf[center, center] = 1.0
        s = 1.0
    return psf / s


def richardson_lucy(img: np.ndarray, psf: np.ndarray, iterations: int = 30) -> np.ndarray:
    # Expect img float64 in range [0,1] or [0,255]; normalize inside
    if img.dtype != np.float64:
        img = img.astype(np.float64)
    # normalize to [0,1]
    maxv = img.max() if img.max() > 1 else 1.0
    img = img / maxv

    # pad PSF to image size by center placement
    estimate = np.full(img.shape, 0.5)
    psf_flip = psf[::-1, ::-1]
    # operate per channel
    if img.ndim == 2:
        channels = 1
        img_ch = [img]
    else:
        channels = img.shape[2]
        img_ch = [img[:, :, c] for c in range(channels)]

    out_ch = []
    for ch in img_ch:
        # init estimate as the input
        est = ch.copy()
        for i in range(iterations):
            conv = cv2.filter2D(est, -1, psf, borderType=cv2.BORDER_REPLICATE)
            relative_blur = ch / (conv + 1e-6)
            est = est * cv2.filter2D(relative_blur, -1, psf_flip, borderType=cv2.BORDER_REPLICATE)
        out_ch.append(est)

    if channels == 1:
        result = out_ch[0]
    else:
        result = np.stack(out_ch, axis=2)

    result = np.clip(result * maxv, 0, 255).astype(np.uint8)
    return result


def local_deblur(pil_img: Image.Image) -> bytes:
    # Convert to OpenCV BGR
    img = np.array(pil_img.convert('RGB'))
    img_cv = img[:, :, ::-1]
    # estimate a motion PSF; parameters could be tuned or exposed
    psf = motion_psf(length=21, angle=0)
    # run RL deconvolution per channel
    result = richardson_lucy(img_cv, psf, iterations=25)
    # convert back to RGB PIL bytes
    result_rgb = result[:, :, ::-1]
    out_pil = Image.fromarray(result_rgb)
    buf = io.BytesIO()
    out_pil.save(buf, format='PNG')
    return buf.getvalue()


@app.post('/deblur')
async def deblur_image(file: UploadFile = File(...)):
    contents = await file.read()
    # if HF env vars are present, call the HF inference API using those credentials
    hf_model = os.getenv('HF_MODEL')
    hf_token = os.getenv('HF_TOKEN')

    if hf_model and hf_token:
        try:
            out_bytes = call_hf_inference(hf_model, hf_token, contents)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception('HF inference failed')
            raise HTTPException(status_code=500, detail=str(e))
        return StreamingResponse(io.BytesIO(out_bytes), media_type='image/png')

    # fallback: local algorithm
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception as e:
        logger.exception('Invalid image uploaded')
        raise HTTPException(status_code=400, detail='Invalid image')

    out_bytes = local_deblur(pil_img)
    return StreamingResponse(io.BytesIO(out_bytes), media_type='image/png')


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('Backend.main:app', host='0.0.0.0', port=int(os.getenv('PORT', 8000)), reload=False)