#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# OpenCV experimentation with motion and face detection
#
# By: Per Øyvind Kanestrøm
#
import cv2
import numpy as np

# Located in openCV source: samples/python2/video.py and samples/python2/common.py
from video import create_capture
#from common import clock, draw_str

def detectFaces(image):
    # Detect the faces (probably research for the options!):
    return cascade.detectMultiScale(image)


def detectMotion(image):
    ## gray to binary: threshold = 100 (arbitrary); maxValue = 255; type = cv2.THRESH_BINARY
    flag, binaryImage = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)  # | cv2.THRESH_OTSU) # cv2.THRESH_BINARY = 0

    # Use the Running Average as the static background
    # md_weight = 0.020 leaves artifacts lingering way too long.
    # md_weight = 0.320 works well at 320x240, 15fps.
    # md_weight should be roughly 1/a = num frames.
    cv2.accumulateWeighted(binaryImage, md_average, md_weight)

    # Convert the scale of the moving average.
    runAvg = cv2.convertScaleAbs(md_average)

    # Subtract the current frame from the moving average.
    md_result = cv2.absdiff(binaryImage, runAvg)

    cv2.imshow('Motion detect', md_result)

    #Find the contours
    contours, hierarchy = cv2.findContours(md_result, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return contours


#Pre process the given image. Returns the rezised original image and the fully processed image
def preProcess(image):

    # Do a little preprocessing:
    orig_img_rezised = cv2.resize(image, (image.shape[1]/2, image.shape[0]/2))
    #orig_img_rezised = image

    # Smooth to get rid of false positives
    orig_img_blurred = cv2.GaussianBlur(orig_img_rezised, (5, 5), 0)
    #orig_img_rezised = cv2.medianBlur(orig_img_rezised, (5, 5), 0)

    # Convert to greyscale
    gray = cv2.cvtColor(orig_img_blurred, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    return (orig_img_rezised, gray)


def md_weight_change(value):
    print "md_weight changed to:", value * 0.01

#Variables
video_src = 0
highgui_name = "Face and motion #mafakka"


#Located in openCV source: data/
#Haar cascade files
cascade_alt = "haarcascades/haarcascade_frontalface_alt.xml"
cascade_def = "haarcascades/haarcascade_frontalface_default.xml"

# Create a new CascadeClassifier from given cascade file:
cascade = cv2.CascadeClassifier(cascade_def)

#Setup cam capture
cam = create_capture(video_src)

#Setup motion detection
#md_average needs to be setup with the same type of picture as used in motion detection
ret, img = cam.read()
img, gray = preProcess(img)
flag, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # cv2.THRESH_BINARY = 0

#Motion detection variables
md_average = np.float32(gray)
md_result = np.float32(gray)
md_weight = 0.9


if __name__ == "__main__":
    # Setup default cam window
    cv2.namedWindow(highgui_name)
    cv2.createTrackbar("md_weight", highgui_name, int(md_weight*100), 100, md_weight_change)

    while True:
        #Read and preprocess image
        success, orig_img = cam.read()
        (orig_img, gray) = preProcess(orig_img)

        #Get the new trackbar value
        md_weight = cv2.getTrackbarPos("md_weight", highgui_name) * 0.01

        if not success:
            print "Cam read error. WTF mate"
            break

        #Detect the faces
        faces = detectFaces(gray)

        #Detect motion
        contours = detectMotion(gray)

        #Add all face points aquired
        for (x, y, width, height) in faces:
            cv2.rectangle(orig_img, (x, y), (x+width, y+height), (255, 0, 0), 2)

        #Apply the contours
        for cnt in contours:
            color = np.random.randint(0, 255, (3)).tolist()  # Select a random color. Hippiestyle!
            cv2.drawContours(orig_img, [cnt], 0, color, 2)

        #Add some text
        text_color = (255, 0, 0)  # color as (B,G,R)
        cv2.putText(orig_img, "Hello world", (45, 20), cv2.FONT_HERSHEY_PLAIN, 1, text_color, thickness=1, lineType=cv2.CV_AA)

        cv2.imshow(highgui_name, orig_img)

        if cv2.waitKey(20) == 27:  # Esc pressed
            break

    #End
    cv2.destroyAllWindows()
