import cvzone
import cv2 as cv
cap = cv.VideoCapture()
myClassifier = cvzone.Classifier('mymodel/keras_model.h5','mymodel/labels.txt')


while True:
    sucess , img = cap.read()
    predictions = myClassifier.getPredictions('img')

    cv.imshow("image",img)
    cv.waitkey(1)