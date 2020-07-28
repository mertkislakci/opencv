import numpy as np
import cv2
def nothing(x):
    pass
img = cv2.imread('ellipse.jpg')
# Create a window
cv2.namedWindow('Treshed')
# create trackbars for treshold change
cv2.createTrackbar('Treshold','Treshed',0,255,nothing)

cap = cv2.VideoCapture("video2.mp4")

while (True):
    ret, frame = cap.read()
    output = frame.copy()
    r = cv2.getTrackbarPos('Treshold', 'Treshed')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, gray = cv2.threshold(gray, r, 255, cv2.THRESH_BINARY)
    bilateral_filtered_image = cv2.bilateralFilter(gray, 5, 175, 175)
    cv2.imshow('bbb', bilateral_filtered_image)
    #gray1 = cv2.equalizeHist(gray)
    #cv2.imshow('cannyn', gray1)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.medianBlur(gray, 5)
    cv2.imshow('aa', gray)
    gray = cv2.bilateralFilter(gray,5,175,175)
    cv2.imshow('bbb', gray)

    #İncelenecek Kısımlar
    #gray = cv2.Canny(gray, 60, 60)
    #cv2.imshow('canny', gray)
    #gray = cv2.Canny(gray, v1, v2)
    #ret3, gray = cv2.threshold(gray, 0, t1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \ cv2.THRESH_BINARY, 11, 3.5)
    #kernel = np.ones((20, 20), np.uint64)
    #gray = cv2.erode(gray, kernel, iterations=1)
    #gray = cv2.dilate(gray, kernel, iterations=1)



    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 300, param1=30, param2=50, minRadius=0, maxRadius=0)

    if circles is not None :

        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
            cv2.imshow('Traking', output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()