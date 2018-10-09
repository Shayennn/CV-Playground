import cv2

cam = cv2.VideoCapture(0)

while True:
    r, img = cam.read()
    if r:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, (98, 100, 85), (150, 255, 255))

        blur_mask = cv2.medianBlur(mask, 7)

        im2, contours, hierarchy = cv2.findContours(blur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours)>0:
            con_size = [len(x) for x in contours]
            cv2.drawContours(img, contours, con_size.index(max(con_size)), (0,255,0), 3)
            epsilon = cv2.arcLength(contours[con_size.index(max(con_size))],True)
            appr = cv2.approxPolyDP(contours[con_size.index(max(con_size))],0.02*epsilon,True)
            pos_x = 0
            pos_y = 0

            x = []
            y = []

            for point in contours[con_size.index(max(con_size))]:
                x.append(point[0][0])
                y.append(point[0][1])

            pos_x=(max(x)+min(x))/2
            pos_y=(max(y)+min(y))/2

            cv2.putText(img,str(len(appr)),(max(x),max(y)), cv2.FONT_HERSHEY_COMPLEX, 4,(255,255,255),2,cv2.LINE_AA)

            print(f'Object pos: {pos_x:.2f},{pos_y:.2f}')

        cv2.imshow('RW',img)

        k = cv2.waitKey(1)
        if k == ord('e'):
            break