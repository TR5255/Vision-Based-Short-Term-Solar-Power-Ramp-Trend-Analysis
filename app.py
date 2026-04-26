import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
import cv2
import pandas as pd
import pvlib
from ModelClass import SequenceModel

app = Flask(__name__)

# -------------------------
# CONFIG
# -------------------------
SEQ = 12
K_MINUTES = 60
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# DETERMINISM
# -------------------------
torch.manual_seed(42)
np.random.seed(42)

# -------------------------
# LOAD MODEL
# -------------------------
model = SequenceModel().to(device)
state_dict = torch.load("best_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# -------------------------
# FISHEYE TRANSFORMATION
# -------------------------
def apply_fisheye_effect(img):
    h, w = img.shape[:2]

    y, x = np.indices((h, w), dtype=np.float32)
    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(r)
    r_distorted = theta / (np.pi / 2)

    r[r == 0] = 1

    x_new = r_distorted * x / r
    y_new = r_distorted * y / r

    map_x = ((x_new + 1) * (w - 1)) / 2
    map_y = ((y_new + 1) * (h - 1)) / 2

    return cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)

def apply_circular_mask(img):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    img[mask == 0] = 0
    return img

# -------------------------
# IMAGE PREPROCESS
# -------------------------
def get_image_from_bytes(img_bytes):
    file_bytes = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = apply_fisheye_effect(img)
    img = apply_circular_mask(img)

    img = cv2.resize(img, (64, 64))
    img = img / 255.0

    return img

# -------------------------
# SOLAR FEATURES (FIXED)
# -------------------------
def compute_future_aux(lat, lon, datetime_str):
    tz = "Asia/Kolkata"

    # ✅ FULL datetime (NO dynamic today)
    start_dt = pd.Timestamp(datetime_str, tz=tz)

    t_now = start_dt + pd.Timedelta(minutes=SEQ - 1)
    t_future = t_now + pd.Timedelta(minutes=K_MINUTES)

    solpos = pvlib.solarposition.get_solarposition(
        time=pd.DatetimeIndex([t_future]),
        latitude=lat,
        longitude=lon
    )

    zen_f = solpos["zenith"].values[0]
    azi_f = solpos["azimuth"].values[0]

    aux = np.array([
        np.cos(np.radians(zen_f)),
        np.sin(np.radians(azi_f))
    ], dtype=np.float32)
    print("Zenith cos:", np.cos(np.radians(zen_f)))
    print("Azimuth sin:", np.sin(np.radians(azi_f)))
    return aux

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    files = request.files.getlist("images")

    # ✅ ensure correct order
    files = sorted(files, key=lambda x: x.filename)

    lat = float(request.form["latitude"])
    lon = float(request.form["longitude"])

    # ✅ expect full datetime now
    datetime_str = request.form["datetime"]  
    # example: "2026-04-08 10:19"

    if len(files) != SEQ:
        return jsonify({"error": "Upload exactly 12 images"}), 400

    # -------------------------
    # IMAGE SEQUENCE
    # -------------------------
    imgs = [get_image_from_bytes(f.read()) for f in files]

    x_np = np.stack(imgs)
    x_np = np.transpose(x_np, (0, 3, 1, 2))

    x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)

    # -------------------------
    # AUX FEATURES
    # -------------------------
    aux_np = compute_future_aux(lat, lon, datetime_str)
    aux = torch.from_numpy(aux_np).unsqueeze(0).to(device)

    # -------------------------
    # INFERENCE
    # -------------------------
    with torch.no_grad():
        pred = model(x, aux).item()

    # optional clamp
    pred = max(0.0, min(1.2, pred))

    return jsonify({
        "pv_ratio": float(pred)
    })

# -------------------------
app.run(debug=True)