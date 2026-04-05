import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time

# Load model
interpreter = tflite.Interpreter(model_path="best_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get input shape from model itself
input_shape = input_details[0]['shape']  # e.g. [1, 320, 320, 3]
HEIGHT, WIDTH = input_shape[1], input_shape[2]

cap = cv2.VideoCapture(0)
object_positions = {}
object_speeds = {}
prev_time = time.time()
PIXEL_TO_CM = 0.1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, (WIDTH, HEIGHT))
    img_input = (img / 255.0).astype(np.float32)
    img_input = np.transpose(img_input, (2, 0, 1))        # HWC -> CHW
    img_input = np.expand_dims(img_input, axis=0)          # add batch dim

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    predictions = output[0].T  # YOLOv8: [4+classes, 8400] -> [8400, 4+classes]

    current_time = time.time()
    dt = max(current_time - prev_time, 1e-5)

    for pred in predictions:
        conf = pred[4:].max()
        if conf > 0.5:
            x, y, w, h = pred[:4]
            x = int(x * frame.shape[1] / WIDTH)
            y = int(y * frame.shape[0] / HEIGHT)

            obj_id = int(x + y)
            if obj_id in object_positions:
                prev_x, prev_y = object_positions[obj_id]
                dist_px = ((x - prev_x)**2 + (y - prev_y)**2) ** 0.5
                object_speeds[obj_id] = (dist_px * PIXEL_TO_CM) / dt

            object_positions[obj_id] = (x, y)

            # Draw
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            label = f"{conf:.2f}"
            if obj_id in object_speeds:
                label += f" | {object_speeds[obj_id]:.1f} cm/s"
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    prev_time = current_time
    cv2.imshow("Waste Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()