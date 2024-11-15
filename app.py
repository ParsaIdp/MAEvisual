import tkinter as tk
from tkinter import Canvas
import cv2
import numpy as np
from PIL import Image, ImageTk
from flask import Flask, request, jsonify, render_template
import base64
import io
import random
import torch
from torchvision.models import vit_b_16



app = Flask(__name__)

patch_size = 32
img = cv2.imread('augmentations.png')
h, v, _ = img.shape
h = h // patch_size
v = v // patch_size
mask = np.zeros((h, v), dtype=np.uint8)

mae_model = vit_b_16(pretrained=True)
mae_model.eval()


def apply_mask(img, mask):
    img_mask = img.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                y_start = i * patch_size
                y_end = y_start + patch_size
                x_start = j * patch_size
                x_end = x_start + patch_size
                img_mask[y_start:y_end, x_start:x_end] = [128, 128, 128]
    return img_mask

def encode_image(img):
    _, buffer = cv2.imencode('.png', img)
    b64_image = base64.b64encode(buffer)
    return b64_image.decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_image')
def get_image():
    global img
    b64_image = encode_image(img)
    return jsonify({'image': b64_image})

@app.route('/update_mask', methods=['POST'])
def update_mask():
    global mask
    data = request.json
    x = data['x']
    y = data['y']
    p_x = int(x // patch_size)
    p_y = int(y // patch_size)
    mask[p_y, p_x] = 1 - mask[p_y, p_x]
    img_mask = apply_mask(img, mask)
    b64_image = encode_image(img_mask)
    return jsonify({'image': b64_image})

@app.route('/random_mask', methods=['POST'])
def random_mask():
    global mask
    p = float(request.json['percent'])
    num_mask = int(p/100 * mask.size)

    indices = list(range(mask.size))
    random.shuffle(indices)
    selected = indices[:num_mask]
    mask.fill(0)

    for idx in selected:
        i, j = divmod(idx, mask.shape[1])
        mask[i, j] = 1

    img_mask = apply_mask(img, mask)
    b64_image = encode_image(img_mask)
    return jsonify({'image': b64_image})


@app.route('/process_with_mae', methods=['POST'])
def process_with_mae():
    global mask
    img_mask = apply_mask(img, mask)
    input_image = cv2.resize(img_mask, (224, 224)) / 255.0 
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_image = (input_image - mean) / std
    input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        output = mae_model(input_tensor)
        print(f"Output shape from MAE: {output.shape}")

    output_image = output.squeeze().numpy()  
    output_image = np.clip((output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255, 0, 255).astype(np.uint8)
    output_image = output_image.transpose(1, 2, 0) 

    _, buffer = cv2.imencode('.png', output_image)
    output_image_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'output_image': output_image_b64})

if __name__ == '__main__':
    app.run(debug=True)

        

