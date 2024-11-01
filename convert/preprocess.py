import numpy as np
import cv2
import sys
import os 

imgs_path = "./coco128"
imgs_data_list = []
#generate opt calibration_data or data
#image_path = sys.argv[1]
for i in os.listdir(imgs_path):
    img = os.path.join(imgs_path, i)
    original_image = cv2.imread(img)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (416, 416))
    image_data = image_data / 255.
    imgs_data_list.append(image_data)


merge_data = np.stack(imgs_data_list, axis=0)
print(merge_data.shape)
np.save('calibration.npy', merge_data)

#generate input.bin
input_top_scale = 255.0  #quantIR input op top_scale
input_top_zp = 0  #quantIR input op top_zp
input = np.round(merge_data[99] * input_top_scale - input_top_zp).astype(np.uint8)
input = np.expand_dims(input, axis=0)
print(input.shape)
input.tofile('input.bin')

input = np.squeeze(input, axis=0)
print(input.shape)

test_img = cv2.cvtColor(input, cv2.COLOR_RGB2BGR)
cv2.imwrite("input.jpg", test_img)
