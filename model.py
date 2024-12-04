# BitPyre Prototype 2
# install this >>> pip install opencv-python numpy tensorflow

import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('fire_detection_model.h5')

# image size for model 
image_size = (224, 224)

# Function to process each frame and predict fire
def process_frame(frame):
    height, width, _ = frame.shape

    # Define sliding window parameters (You can adjust these)
    window_size = 224  # Use the input size of the model
    step_size = 100  # Step size for sliding window (adjust for performance)
    fire_detected = False

    for y in range(0, height - window_size, step_size):
        for x in range(0, width - window_size, step_size):
            # Extract the region of interest (ROI)
            roi = frame[y:y + window_size, x:x + window_size]

            # Resize frame for the model input
            resized_roi = cv2.resize(roi, image_size)

            # Normalize the frame
            normalized_roi = resized_roi / 255.0

            # Add batch dimension (model expects a batch)
            input_frame = np.expand_dims(normalized_roi, axis=0)

            # Get model predictions
            predictions = model.predict(input_frame)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class]

            # If fire is detected and confidence is above threshold
            if predicted_class == 0 and confidence > 0.3:  # Adjust confidence threshold as needed
                fire_detected = True
                break
        if fire_detected:
            break

    return fire_detected
def main():
    # Open the camera (0 is usually the default camera; change if needed)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Process the frame to detect fire
        fire_detected = process_frame(frame)

        # Display the result
        if fire_detected:
            cv2.putText(frame, "Fire Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No Fire", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the results
        cv2.imshow("Fire Detection", frame)

        # Press 'q' to exit the loop and close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
