
import os
import cv2
import pandas as pd
import numpy as np


IMG_SIZE = 512

import cv2
import numpy as np

def resize_odir_image(img_bgr, target_size=IMG_SIZE):
    """
    Combines Circular Cropping, Aspect-Ratio Resizing, and Padding.
    """
    # 1. Load and initial crop to remove obvious black dead space
    if img_bgr is None: return None
    
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Thresholding to find the retina boundary
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(mask)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img_bgr = img_bgr[y:y+h, x:x+w]

    # 2. Letterbox Resize (Preserve Aspect Ratio)
    h, w = img_bgr.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Use INTER_AREA for high-quality downsampling
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 3. Create Square Canvas and Center
    final_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    offset_y = (target_size - new_h) // 2
    offset_x = (target_size - new_w) // 2
    final_img[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    
    return final_img

def usuyama_prep(img_bgr):
    """Enhances vessels and normalizes lighting."""
    # Circular Crop: Find non-black pixels and crop
    img_bgr = resize_odir_image(img_bgr, target_size=IMG_SIZE)  # Crop to circular region
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 10)
    enhanced = cv2.addWeighted(img_bgr, 4, blurred, -4, 128)
    return enhanced

def usuyama_green_prep(img_bgr):
    """Extracts the green channel and normalizes."""
    # Circular Crop: Find non-black pixels and crop
    img_bgr = resize_odir_image(img_bgr, target_size=IMG_SIZE)  # Crop to circular region
    green_ch = img_bgr[:, :, 1] # x,y,channel = 0 -> r, 1->green, 2->red
    blurred = cv2.GaussianBlur(green_ch, (0, 0), 10)
    green_ben = cv2.addWeighted(green_ch, 4, blurred, -4, 128)
    
    # Convert to 3-channel for Model Input
    return cv2.merge([green_ben, green_ben, green_ben])

def hybrid_clahe_green_ben(img_bgr, clip_limit=2.0, grid_size=(8,8)):
    # Circular Crop: Find non-black pixels and crop
    img_bgr = resize_odir_image(img_bgr, target_size=IMG_SIZE)  # Crop to circular region
    # 1. Green Channel
    green = img_bgr[:, :, 1] # x,y,channel = 0 -> r, 1->green, 2->red
    
    # 2. CLAHE (Do this first to boost local signal)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced_green = clahe.apply(green)
    
    # 3. Milder Ben Graham (Usuyama style)
    # This cleans up any "glow" the CLAHE might have amplified
    blurred = cv2.GaussianBlur(enhanced_green, (0, 0), 10)
    final = cv2.addWeighted(enhanced_green, 4, blurred, -4, 128)
    
    return cv2.merge([final, final, final])


def color_clahe(img, clip_limit=2.0):
    # Convert to LAB (L is Lightness, A/B are color dimensions)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L-channel only
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l) 
    
    # Merge and convert back to BGR
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def hybrid_clahe_usuyama_prep(img_bgr):
    """Enhances vessels and normalizes lighting."""
    # Circular Crop: Find non-black pixels and crop
    img_bgr = resize_odir_image(img_bgr, target_size=IMG_SIZE)  # Crop to circular region
    img_bgr = color_clahe(img_bgr, clip_limit=2.0)  # First apply CLAHE to boost local contrast
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 10)
    enhanced = cv2.addWeighted(img_bgr, 4, blurred, -4, 128)
    return enhanced


def crop_fundus_circle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]
def center_retina(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y+h, x:x+w]

    # pad to square
    h, w = crop.shape[:2]
    size = max(h, w)
    padded = np.zeros((size, size, 3), dtype=crop.dtype)

    y0 = (size - h) // 2
    x0 = (size - w) // 2
    padded[y0:y0+h, x0:x0+w] = crop

    return padded
def gamma_correction(img, gamma=0.9):
    img_float = img.astype(np.float32) / 255.0
    # Gamma correction
    img_gamma = np.power(img_float, gamma)
    # Convert back to 0–255 for saving
    img_result = (img_gamma * 255).astype(np.uint8)
    return img_result
def mask_outside_retina(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    return cv2.bitwise_and(img, mask)
def custom_gamma(img_bgr):
    # this is our preprocessing we honed from multi-class method
    img_bgr = crop_fundus_circle(img_bgr)
    img_bgr = center_retina(img_bgr)
    img_bgr = mask_outside_retina(img_bgr)
    img_bgr = resize_odir_image(img_bgr, target_size=IMG_SIZE)  
    img_bgr = gamma_correction(img_bgr)
    return img_bgr
