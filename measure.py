import base64
import json
import numpy as np
import cv2
from http.server import BaseHTTPRequestHandler

# ISO/IEC 7810 ID-1 card (standard credit/debit card): 85.60mm x 53.98mm
CARD_WIDTH_CM = 8.56
CARD_RATIO = 8.56 / 5.398  # ~1.586


def find_card(contours):
    """Find the largest quadrilateral contour whose proportions match a
    standard credit card, using its actual aspect ratio rather than just
    'any 4-sided shape' (which false-positives on all sorts of things)."""
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
        if cv2.contourArea(c) < 500:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        rect = cv2.minAreaRect(c)
        rw, rh = rect[1]
        if rw == 0 or rh == 0:
            continue
        long_side, short_side = max(rw, rh), min(rw, rh)
        ratio = long_side / short_side
        if abs(ratio - CARD_RATIO) / CARD_RATIO < 0.18:  # ~18% tolerance
            return c, long_side
    return None, None


def find_foot(gray, card_contour, img_shape):
    """Isolate the foot as the largest contour left once the card region is
    masked out, using Otsu thresholding against a (assumed) plain background."""
    h, w = img_shape[:2]
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if card_contour is not None:
        cv2.drawContours(thresh, [card_contour], -1, 0, thickness=cv2.FILLED)

    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    foot_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(foot_contour) < (h * w) * 0.01:
        return None

    return foot_contour


def half_round(x):
    return max(1.0, round(x * 2) / 2)


def cm_to_sizes(foot_cm):
    foot_in = foot_cm / 2.54
    us_men = (3 * foot_in) - 22.0
    us_women = us_men + 1.5
    uk = us_men - 1.0
    eu = max(16.0, round((foot_cm + 1.5) * 1.5 * 2) / 2)

    return {
        "cm": round(foot_cm, 1),
        "inches": round(foot_in, 1),
        "uk": half_round(uk),
        "us_men": half_round(us_men),
        "us_women": half_round(us_women),
        "eu": eu,
    }


class handler(BaseHTTPRequestHandler):
    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        # Browsers preflight JSON POST requests — without this, the actual
        # POST never even gets sent from a browser.
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            if length == 0:
                raise ValueError("Empty request body.")
            post_data = self.rfile.read(length)
            payload = json.loads(post_data.decode('utf-8'))
            image_data = payload.get("image")

            if not image_data:
                raise ValueError("Payload missing 'image' field (base64 string).")
            if "," in image_data:
                image_data = image_data.split(",")[1]

            img_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image data.")

            # Downscale very large photos for speed/consistency
            h0, w0 = img.shape[:2]
            max_dim = 1200
            if max(h0, w0) > max_dim:
                scale = max_dim / max(h0, w0)
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.dilate(cv2.Canny(blurred, 50, 150), None, iterations=1)
            contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            card_contour, card_long_px = find_card(contours)
            if card_contour is None:
                raise ValueError(
                    "Couldn't find a credit-card-shaped reference in the photo. "
                    "Make sure the whole card is flat, fully visible, and not overlapping the foot."
                )
            pixels_per_cm = card_long_px / CARD_WIDTH_CM

            foot_contour = find_foot(gray, card_contour, img.shape)
            if foot_contour is None:
                raise ValueError(
                    "Couldn't isolate the foot from the background. "
                    "Try a plain, well-lit, contrasting surface (e.g. a sheet of paper)."
                )

            (fw, fh) = cv2.minAreaRect(foot_contour)[1]
            foot_cm = max(fw, fh) / pixels_per_cm

            if not (10 <= foot_cm <= 40):
                raise ValueError(
                    f"Measured length ({round(foot_cm, 1)} cm) looks implausible. "
                    "Retake the photo from directly above, with the card lying flat next to the foot."
                )

            sizes = cm_to_sizes(foot_cm)
            self._send_json(200, {
                "status": "success",
                "measurements": {"cm": sizes["cm"], "inches": sizes["inches"]},
                "shoe_sizes": {
                    "uk": sizes["uk"],
                    "us_men": sizes["us_men"],
                    "us_women": sizes["us_women"],
                    "eu": sizes["eu"],
                },
            })

        except Exception as err:
            self._send_json(400, {"status": "error", "message": str(err)})
