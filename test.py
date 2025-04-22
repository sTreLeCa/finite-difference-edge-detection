import cv2
import numpy as np

def sobel_edge_detection(image):
    # Apply Sobel Edge Detection
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    edges = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2.0), 1.0, cv2.pow(sobely, 2.0), 1.0, 0.0))
    
    # Convert edges to 8-bit unsigned integer
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Convert edges to binary image
    _, binary = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def central_finite_difference(image, order, degree):
    # Convert image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Convert image to float32 for accurate computations
    image = image.astype(np.float32)
    
    # Define central finite difference kernels for different degrees
    if degree == 1:
        kernel = np.array([[-1, 0, 1]])
    elif degree == 2:
        kernel = np.array([[1, -2, 1]])
    elif degree == 3:
        kernel = np.array([[-1, 2, -2, 1]])
    elif degree == 4:
        kernel = np.array([[-1, 16, -30, 16, -1]])
    elif degree == 5:
        kernel = np.array([[1, -8, 8, -1]])
    elif degree == 6:
        kernel = np.array([[1, -6, 15, -20, 15, -6, 1]])
    else:
        raise ValueError("Degree must be in the range 1 to 6.")
    
    # Compute the derivative in the horizontal direction
    dx = cv2.filter2D(image, -1, kernel) if order == 1 else cv2.filter2D(image, -1, np.flip(kernel, axis=1))
    
    # Compute the derivative in the vertical direction
    dy = cv2.filter2D(image, -1, kernel.T) if order == 1 else cv2.filter2D(image, -1, np.flip(kernel.T, axis=0))
    
    # Compute the magnitude of the gradient
    magnitude = np.sqrt(dx**2 + dy**2)
    
    # Normalize the magnitude to [0, 255]
    magnitude = (magnitude / np.max(magnitude)) * 255
    
    threshold = 50
    # Convert to uint8
    magnitude = magnitude.astype(np.uint8)
   # magnitude = magnitude.astype(np.uint8)
    
    return magnitude

def main():
    # Load the image
    image = cv2.imread("orangegutang.jpg", cv2.IMREAD_GRAYSCALE)
    
    # Apply Sobel edge detection
    sobel_edges = sobel_edge_detection(image)
    cv2.imshow("Sobel Edges", sobel_edges)
    
    # Display central finite difference edge detection for different orders
  #  for order in range(1, 7):  # Range for orders
   #     edges_order = central_finite_difference(image, order, order)  # Degree is same as order
    #    cv2.imshow(f"Edges (Order {order})", edges_order)
     #   cv2.waitKey(1000)  # Wait 1 second between showing each image
    
    # Display central finite difference edge detection for order 1 and varying degrees
    for degree in range(1,7):  # Range for degrees
        edges_degree = central_finite_difference(image, 1, degree)  # Order 1 for all degrees
        cv2.imshow(f"Edges (Degree {degree})", edges_degree)
        cv2.waitKey(1000)  # Wait 1 second between showing each image
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
