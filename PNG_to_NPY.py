import cv2
import numpy as np
from numpy import save

image = cv2.imread('./spiegazione_7.png', cv2.IMREAD_GRAYSCALE)
save('./spiegazione_7.npy', image)