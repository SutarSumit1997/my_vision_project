import cv2
import pytesseract
import numpy as np
from pytesseract import Output
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)

def canny(image):
    return cv2.Canny(image, 50, 20)

def erode(image):
    kernel = np.ones((4,4),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

image=cv2.imread("ufi1.jpg") 
# Crop image
cropped_image = image[438:520,417:710]
canny_img=canny(cropped_image)
dilate_img=dilate(canny_img)
erod_img=erode(dilate_img)

# Display cropped image
#cv2.imshow("Cropped image", cropped_image)
#cv2.waitKey(0)
'''d = pytesseract.image_to_data(canny_img, output_type=Output.DICT)
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 60:
        (text, x, y, w, h) = (d['text'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # don't show empty text
        if text and text.strip() != "":
            canny_img = cv2.rectangle(canny_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            canny_img = cv2.putText(canny_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            file1 = open("myfile.txt", "a")  # append mode
            file1.write(text+"\n")'''
options = "outputbase digits"
config = ("-l eng --oem 1 --psm 7")
text = pytesseract.image_to_string(erod_img,config=config)
print(text)
cv2.imshow("Frame1", erod_img)
cv2.waitKey(0)
print("done")