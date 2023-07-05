import cv2

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
img = cv2.imread('images/Sami.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.08, 5)

for (x, y, w, h) in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
cv2.namedWindow('Sami Detected!')
cv2.imshow('Sami Detected!', img)
cv2.imwrite('./Sami_detected.jpg', img)
cv2.waitKey(0)
