import cv2 as cv


image_path = r'/home/mohsen/Pictures/team.jpg'
def rescaleframe(frame,scale = 0.5):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width,height)

    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

img1 = cv.imread(image_path)
img = rescaleframe(img1)
# cv.imshow('face1',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray',gray)

haar_cascade = cv.CascadeClassifier('/home/mohsen/Documents/trainee/tutorial/open_cv/face_recognition/haar_face.xml')
faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
print(f'num of faces = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('detect_face',img)


cv.waitKey(0)