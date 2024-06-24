import cv2
import numpy as np
import serial
import time
from roboflow import Roboflow

# Initialize USB camera
cap = cv2.VideoCapture(0)  # Use the correct camera index if you have multiple cameras

# Load Roboflow model
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace().project("weed-detection")
model = project.version("YOUR_MODEL_VERSION").model

# Initialize Arduino serial communication
ser = serial.Serial('/dev/ttyUSB0', 9600)  # Adjust the port and baud rate as needed
time.sleep(2)  # Wait for the serial connection to initialize

frame_count = 0
coordinates_sum = np.zeros(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(frame, confidence=40, overlap=30).json()

    for prediction in results['predictions']:
        x1 = int(prediction['x'] - prediction['width'] / 2)
        y1 = int(prediction['y'] - prediction['height'] / 2)
        x2 = int(prediction['x'] + prediction['width'] / 2)
        y2 = int(prediction['y'] + prediction['height'] / 2)
        cls = prediction['class']
        conf = prediction['confidence']

        if cls == 'weed':  # Replace with the actual class name for weed
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            coordinates_sum += np.array([center_x, center_y])
            frame_count += 1

            print(f'Center coordinates of weed: ({center_x}, {center_y})')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'weed {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if frame_count == 5:
        average_coordinates = coordinates_sum / 5
        print("Average center coordinates after 5 frames:", tuple(average_coordinates))
        frame_count = 0
        coordinates_sum = np.zeros(2)
        x_coordinate = average_coordinates[0]
        print("X-coordinate:", x_coordinate)

        if x_coordinate <= 100:
            ser.write(b'1')
        elif 100 < x_coordinate <= 200:
            ser.write(b'2')
        elif 200 < x_coordinate <= 300:
            ser.write(b'3')
        elif 300 < x_coordinate <= 400:
            ser.write(b'4')

cap.release()
cv2.destroyAllWindows()
ser.close()
