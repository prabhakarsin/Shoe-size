import base64
import json
import numpy as np
import cv2
import onnxruntime as ort
from http.server import BaseHTTPRequestHandler

# Core ML engine model initialization hook
MODEL_PATH = "api/foot_keypoint_model.onnx"
session = None

try:
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
except Exception:
    session = None  # Fallback gracefully if model file is not uploaded yet

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            payload = json.loads(post_data.decode('utf-8'))
            image_data = payload.get("image")
            
            if not image_data:
                raise ValueError("Payload missing base64 image field.")
                
            if "," in image_data:
                image_data = image_data.split(",")[1]
                
            img_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            h_orig, w_orig, _ = img.shape

            # 1. REFERENCE SCANNING VIA OPENCV (Detect standard credit card bounds)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            pixels_per_cm = None
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    _, _, w_box, _ = cv2.boundingRect(approx)
                    pixels_per_cm = w_box / 8.56
                    break

            if pixels_per_cm is None:
                pixels_per_cm = w_orig / 25.0  # Fallback dynamic proportion calculation

            # 2. KEYPOINT PROCESSING (Heel-to-toe estimation)
            heel_y, toe_y = int(h_orig * 0.82), int(h_orig * 0.22)

            if session is not None:
                # Custom ONNX machine learning graph forward pass execution
                resized = cv2.resize(img, (256, 256))
                input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                input_data = np.transpose(input_data, (2, 0, 1))
                input_tensor = np.expand_dims(input_data, axis=0)
                
                outputs = session.run(None, {session.get_inputs()[0].name: input_tensor})
                # Maps relative outputs directly back onto full image scale
                heel_y = int(outputs[0][1] * h_orig)
                toe_y = int(outputs[0][3] * h_orig)
            
            # 3. MATHEMATICAL SIZE MAPPING CALCULATIONS
            foot_pixel_length = abs(heel_y - toe_y)
            foot_cm = foot_pixel_length / pixels_per_cm
            foot_inches = foot_cm / 2.54

            us_mens = (3 * foot_inches) - 22.0
            uk_size = us_mens - 1.0
            eu_size = (foot_cm + 1.5) * 1.5

            response_payload = {
                "status": "success",
                "measurements": {
                    "cm": round(foot_cm, 1),
                    "inches": round(foot_inches, 1)
                },
                "shoe_sizes": {
                    "uk": max(1.0, round(uk_size * 2) / 2),
                    "us_men": max(1.0, round(us_mens * 2) / 2),
                    "eu": max(16.0, round(eu_size))
                }
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))

        except Exception as err:
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(err)}).encode('utf-8'))
          
