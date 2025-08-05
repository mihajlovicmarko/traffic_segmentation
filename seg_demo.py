import cv2
import numpy as np
from openvino.runtime import Core

# === Load model ===
core = Core()
model_path = "intel/semantic-segmentation-adas-0001/FP32/semantic-segmentation-adas-0001.xml"
model = core.read_model(model_path)
compiled_model = core.compile_model(model, "CPU")

input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# === Load image ===
image = cv2.imread("test/test2.jpg")
if image is None:
    raise FileNotFoundError("Could not load test.png")

input_height, input_width = input_layer.shape[2], input_layer.shape[3]
image_resized = cv2.resize(image, (input_width, input_height))
input_tensor = np.expand_dims(image_resized.transpose(2, 0, 1), axis=0)

# === Inference ===
result = compiled_model([input_tensor])[output_layer]
seg_map = result.squeeze().astype(np.uint8)  # shape = (H, W)
print("Seg map shape:", seg_map.shape)
print("Unique labels:", np.unique(seg_map))

# === Overlay segmentation ===
np.random.seed(42)
color_map = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
seg_overlay = color_map[seg_map]  # shape = (H, W, 3)
blended = cv2.addWeighted(image_resized, 0.5, seg_overlay, 0.5, 0)

# === Show or save result ===
cv2.imwrite("segmented.jpg", blended)
print("Saved as segmented.jpg")
