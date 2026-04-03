import cv2
import numpy as np
import onnxruntime as ort
import time

# Load ONNX model
session = ort.InferenceSession("best.onnx")

input_name = session.get_inputs()[0].name

# Camera
cap = cv2.VideoCapture(0)

# Tracking storage
object_positions = {}
object_speeds = {}

prev_time = time.time()

# Pixel to cm scale (CALIBRATE THIS!)
PIXEL_TO_CM = 0.1  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for speed
    img = cv2.resize(frame, (320, 320))
    img_input = img / 255.0
    img_input = img_input.transpose(2, 0, 1)
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)

    # Run inference
    outputs = session.run(None, {input_name: img_input})

    predictions = outputs[0][0]  # YOLO output

    current_time = time.time()
    dt = current_time - prev_time

    for pred in predictions:
        conf = pred[4]

        if conf > 0.5:
            x, y, w, h = pred[:4]

            # Scale back to original frame
            x = int(x * frame.shape[1] / 320)
            y = int(y * frame.shape[0] / 320)

            # Fake ID (basic tracking)
            obj_id = int(x + y)

            if obj_id in object_positions:
                prev_x, prev_y = object_positions[obj_id]

                dx = x - prev_x
                dy = y - prev_y

                dist_pixels = (dx**2 + dy**2) ** 0.5
                dist_cm = dist_pixels * PIXEL_TO_CM

                speed = dist_cm / dt
                object_speeds[obj_id] = speed

            object_positions[obj_id] = (x, y)

            # Draw box
            cv2.circle(frame, (x, y), 5, (0,255,0), -1)

            if obj_id in object_speeds:
                cv2.putText(frame, f"{object_speeds[obj_id]:.2f} cm/s",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 2)

    prev_time = current_time

    cv2.imshow("Waste Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()