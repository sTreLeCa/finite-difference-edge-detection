import cv2
import numpy as np

def central_finite_difference(image, degree):
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert image to float32 for accurate computations
    image = image.astype(np.float32)
    
    # Define central finite difference kernel
    kernel = np.array([[-1, 0, 1]])
    
    # Compute the derivative in the horizontal direction
    dx = cv2.filter2D(image, -1, kernel)
    
    # Compute the derivative in the vertical direction
    dy = cv2.filter2D(image, -1, kernel.T)
    
    # Compute the magnitude of the gradient
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Normalize the magnitude to [0, 255]
    magnitude = (magnitude / np.max(magnitude)) * 255
    
    # Convert to uint8
    magnitude = magnitude.astype(np.uint8)
    
    return magnitude

def main():
    # Open the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Apply edge detection
        edges = central_finite_difference(frame, degree=1)

        # Display the resulting frame
        cv2.imshow('Edge Detection', edges)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
