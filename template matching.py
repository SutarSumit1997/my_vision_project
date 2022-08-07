import cv2
import numpy as np
key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
def remove_noise(image):
    return cv2.medianBlur(image,5)
while True:
    try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
       # print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img.jpg', cv2.COLOR_BAYER_GR2BGR)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
            print("Processing image...")
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break
# Read the main image
img_rgb = cv2.imread('saved_img.jpg')
 
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# Read the template
template = cv2.imread('saved_img1.jpg', 0)
 
# Store width and height of template in w and h
w, h = template.shape[::-1]
 
# Perform match operations.
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
 
# Specify a threshold
threshold = 0.55
 
# Store the coordinates of matched area in a numpy array
loc = np.where(res >= threshold)
print(str(loc))
# Draw a rectangle around the matched region.
flag=False
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 1)
    print(pt)
    print((pt[0] + w, pt[1] + h))
    flag=True
    #print ((pt[0] + w, pt[1] + h))

# Show the final image with the matched area.

if flag==False:
    start_point = (240,186)
    end_point = (528,311)
    color = (0, 0, 255)
    thickness = 4
    cv2.rectangle(img_rgb, start_point, end_point, color, thickness)
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)