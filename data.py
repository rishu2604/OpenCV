import cv2
import os
from PIL import Image

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id and press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

# Initialize individual sampling face count
count = 0

while(True):
    ret, img = cam.read()
    # Flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect frontal faces
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    # Detect profile faces
    profiles = profile_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite(r"C:\Users\Rishu Anand\OneDrive\Desktop\cv\images\User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow(r'C:\Users\Rishu Anand\OneDrive\Desktop\cv\images', img)

    for (x,y,w,h) in profiles:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        count += 1
        # Save the captured image into the datasets folder
        cv2.imwrite(r"C:\Users\Rishu Anand\OneDrive\Desktop\cv\images\User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow(r'C:\Users\Rishu Anand\OneDrive\Desktop\cv\images', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27 or count >= 30: # Take 30 face samples and stop video
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()