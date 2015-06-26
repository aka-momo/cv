import cv2
import cv2.cv as cv
import sys
import numpy as np

def load_image(img_src):
  print "Loading Image ..."
  img = cv2.imread('lenna.png', cv2.CV_LOAD_IMAGE_COLOR)
  if (img is None):                      ## Check for invalid input
    print "Could not open or find the image"
  return img

def display_image(mat, window_name):
  cv2.namedWindow(window_name)
  cv2.imshow(window_name,mat)

def gray_scale(img):
  grayImage = np.zeros(shape=(img.shape[0],img.shape[1]), dtype=np.uint8)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      grayImage[i,j] = np.mean(img[i,j])
  return grayImage

def equalize(img):
  return cv2.equalizeHist(img)

def lbp(img):
  result = np.zeros(shape=(img.shape[0],img.shape[1]), dtype=np.uint8)
  for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
      binaryValues = np.ones(8, dtype=np.uint8)
      center = img[i,j]
      binaryValues[7] =  img[i, j-1] > center
      binaryValues[6] =  img[i-1, j-1] > center
      binaryValues[5] =  img[i-1, j] > center
      binaryValues[4] =  img[i-1, j+1] > center
      binaryValues[3] =  img[i, j+1] > center
      binaryValues[2] =  img[i+1, j+1] > center
      binaryValues[1] =  img[i+1, j] > center
      binaryValues[0] =  img[i+1, j-1] > center
      newValue = 0
      for index, binaryValue in enumerate(binaryValues):
        newValue += binaryValue*(2^index)
      result[i,j] = newValue
  return result

x = load_image("Original Image")
display_image(equalize(gray_scale(x)), "Sample Image")
cv2.waitKey(0)
cv2.destroyAllWindows()