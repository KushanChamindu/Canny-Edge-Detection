import cv2 as cv
# from google.colab.patches import cv2_imshow
import sys
import numpy as np


# Load image
def load_image(image_name):
  image_path = "./"
  img = cv.imread(cv.samples.findFile(image_path+image_name))   # Read the file
  if img is None:
    sys.exit("Could not read the image.")
  return img

def convert2gray(img):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]    ## seperate r,gb, values
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     ## convert to gray scale based on sensivity level in human eye
    return gray

## Gaussian kernal generate
def generate_gaussian_kernel(kernal_size, sigma=1):  
  gaussian_kernal = np.zeros((kernal_size, kernal_size), np.float32)
  i = kernal_size//2
  j = kernal_size//2

  for x in range(-i, i+1):
      for y in range(-j, j+1):
          normal  = 2*np.pi*(sigma**2)
          g = np.exp(-(x**2 + y**2)/(2*sigma**2))
          gaussian_kernal[x+i, y+j] = (1/normal)*g 
  return gaussian_kernal/gaussian_kernal.sum()

## Convolution operation
def matrix_convolution(image, kernal): 
  image_copy = image.copy()
  kernal_size = len(kernal)
  #### Wrap the image for filtering
  for k in range(len(image)):
    for i in range(kernal_size//2):
      image_copy[k].insert(0,image[k][-1-(i*2)])
      image_copy[k].append(image[k][1+(i*2)])
  for i in range(kernal_size//2):
    image_copy.insert(0,image[-1-i].copy())
    image_copy.append(image[i].copy())

  image_x = len(image_copy)
  image_y = len(image_copy[0])
  result= []
  kernal_middle = kernal_size//2
  for x in range(kernal_middle, image_x - kernal_middle):
    temp = []
    for y in range(kernal_middle, image_y - kernal_middle):
      value = 0
      for i in range(len(kernal)):
        for j in range(len(kernal)):
          xn = x + i - kernal_middle
          yn = y + j - kernal_middle
          filtered_value = kernal[i][j]*image_copy[xn][yn]
          value += filtered_value
      temp.append(value)
    result.append(temp)
  return np.array(result)

## gradient estimation
def gradient_estimation(img, estimation_type):
  assert (estimation_type in ["Prewitt", "Sobel", "Robert"]), "estimation_type should be  in [\"Prewitt\", \"Sobel\", \"Robert\"]"
  if estimation_type == "Prewitt":
    M_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
    M_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float32)
  elif estimation_type == "Sobel":
    M_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    M_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
  else:
    M_x = np.array([[0, 1], [-1, 0]], np.float32)
    M_y = np.array([[1, 0], [0, -1]], np.float32)
   
  G_x = matrix_convolution(img.tolist(), M_x.tolist())
  G_y = matrix_convolution(img.tolist(), M_y.tolist())

  G = (G_x**2 + G_y**2)**0.5
  G = (G / G.max()) * 255
  theta = np.arctan2(G_y,G_x)

  return (G,theta)

## non maxima suppression
def non_max_suppression(img, angles):
  size = img.shape
  angles = angles * 180. / np.pi  ## convert radian to degrees
  angles[angles < 0] += 180
  suppressed = np.zeros(size)
  for i in range(1, size[0] - 1):
      for j in range(1, size[1] - 1):
          if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
              value_to_compare = max(img[i, j+1], img[i, j-1])
          elif (22.5 <= angles[i, j] < 67.5):
              value_to_compare = max(img[i+1, j-1], img[i-1, j+1])
          elif (67.5 <= angles[i, j] < 112.5):
              value_to_compare = max(img[i+1, j], img[i-1, j])
          elif (112.5 <= angles[i,j] < 157.5):
              value_to_compare = max(img[i-1, j-1], img[i+1, j+1])
          
          if img[i, j] >= value_to_compare:
            suppressed[i, j] = img[i, j]
          else:
            suppressed[i, j] = 0
  # suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
  return suppressed

## double threshold 
def double_threshold(img, lower_threshold=0.05, high_threshold= 0.09):
  high_threshold = img.max() * high_threshold
  lower_threshold = high_threshold * lower_threshold

  size = img.shape
  res = np.zeros(size, dtype=np.int32)
  
  weak = np.int32(25)
  strong = np.int32(255)
  
  for i in range(1, size[0] - 1):
    for j in range(1, size[1] - 1):
      if img[i,j] >= high_threshold:
        res[i, j] = strong
      elif img[i,j] >= lower_threshold and img[i,j] <= high_threshold:
        res[i, j] = weak
  return res

## hysteresis edge tracking
def hysteresis_edge_tracked(img):
  size = img.shape
  weak = np.int32(25)
  strong = np.int32(255)
  for i in range(1, size[0]-1):
    for j in range(1, size[1]-1):
      if (img[i,j] == weak):
        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
            img[i, j] = strong
        else:
            img[i, j] = 0
  return img

def canny_edge_detection(img,gaussian_kernal_size, estimation_method= "Prewitt",lower_threshold=0.05, high_threshold= 0.3):
  # cv.imwrite('original image.jpg',img)
  # Step 1 : Convert image into gray scal image
  img_gray = convert2gray(img)
  cv.imwrite('Gray scalled image.jpg',img_gray)

  # Step 2 : Apply gaussian filter
  gaussian_kernal = generate_gaussian_kernel(gaussian_kernal_size)
  gaussian_filtered_img = matrix_convolution(img_gray.tolist(),gaussian_kernal)
  cv.imwrite('Gaussian filtered image.jpg',gaussian_filtered_img)
  
  # Step 3 : Gradient estimation
  (G,theta) = gradient_estimation(gaussian_filtered_img, estimation_method)
  cv.imwrite('Gradient estimated image.jpg',G)

  # Step 4 : Non-Maximum Suppression
  suppressed = non_max_suppression(G, theta)
  cv.imwrite('Suppressed image.jpg',suppressed)

  # Step 5 : Double threshold
  double_threshold_image = double_threshold(suppressed, lower_threshold=lower_threshold, high_threshold= high_threshold)
  cv.imwrite('After double treshold image.jpg',double_threshold_image.astype(np.uint8))

  hysteresis_edge_tracked_image = hysteresis_edge_tracked(double_threshold_image)
  cv.imwrite('Final image.jpg',hysteresis_edge_tracked_image)
canny_edge_detection(load_image("cameramanN1.jpg"), 5, "Sobel")

