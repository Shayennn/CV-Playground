import cv2

max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'

def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H-1, low_H)
    cv2.setTrackbarPos(low_H_name, 'OJ', low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv2.setTrackbarPos(high_H_name, 'OJ', high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv2.setTrackbarPos(low_S_name, 'OJ', low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv2.setTrackbarPos(high_S_name, 'OJ', high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv2.setTrackbarPos(low_V_name, 'OJ', low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv2.setTrackbarPos(high_V_name, 'OJ', high_V)


cv2.namedWindow('RW')
cv2.namedWindow('OJ')

cv2.createTrackbar(low_H_name, 'OJ' , low_H, max_value_H, on_low_H_thresh_trackbar)
cv2.createTrackbar(high_H_name, 'OJ' , high_H, max_value_H, on_high_H_thresh_trackbar)
cv2.createTrackbar(low_S_name, 'OJ' , low_S, max_value, on_low_S_thresh_trackbar)
cv2.createTrackbar(high_S_name, 'OJ' , high_S, max_value, on_high_S_thresh_trackbar)
cv2.createTrackbar(low_V_name, 'OJ' , low_V, max_value, on_low_V_thresh_trackbar)
cv2.createTrackbar(high_V_name, 'OJ' , high_V, max_value, on_high_V_thresh_trackbar)

cam = cv2.VideoCapture(0)

while True:
    r, img = cam.read()
    if r:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, (low_H, low_S, low_V), (high_H, high_S, high_V))

        cv2.imshow('RW',img)
        cv2.imshow('OJ',mask)

        k = cv2.waitKey(1)
        if k == ord('e'):
            break