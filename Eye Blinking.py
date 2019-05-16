import cv2
import numpy as np
import dlib
from math import hypot

video = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/admin/Downloads/shape_predictor_68_face_landmarks.dat")


while True:
    ret, frame = video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    font = cv2.FONT_HERSHEY_COMPLEX

    #count = 1

    faces = detector(gray)

    for face in faces:
       # x,y = face.left(),face.top()
        #x1,y1 = face.right(),face.bottom()
        #cv2.rectangle(frame,(x,y),(x1,y1),(0,255,0),2)

       #this is for right eye......

       #horizontal line
        landmark = predictor(gray,face)
        R_Hori_left_x = landmark.part(36).x
        R_Hori_left_y = landmark.part(36).y
        R_Hori_left = (R_Hori_left_x,R_Hori_left_y)
        R_Hori_right_x = landmark.part(39).x
        R_Hori_right_y = landmark.part(39).y
        R_Hori_right = (R_Hori_right_x,R_Hori_right_y)
        cv2.line(frame,R_Hori_left,R_Hori_right,(0,255,0),1)

        #find midpoint for vertical line 
        R_upper_mid_x = int((landmark.part(37).x + landmark.part(38).x)/2)
        R_upper_mid_y = int((landmark.part(37).y + landmark.part(38).y)/2)
        R_upper_mid = (R_upper_mid_x,R_upper_mid_y)
        R_bottom_mid_x = int((landmark.part(41).x + landmark.part(40).x)/2)
        R_bottom_mid_y = int((landmark.part(41).y + landmark.part(40).y)/2)
        R_bottom_mid = (R_bottom_mid_x,R_bottom_mid_y)
        cv2.line(frame,R_upper_mid,R_bottom_mid,(0,255,0),1)
        
        #find the lenght for both horizontal and vertical line
        R_hori_lenght = hypot(R_Hori_left_x - R_Hori_right_x, R_Hori_left_y - R_Hori_right_y)
        R_ver_lenght = hypot(R_upper_mid[0] - R_bottom_mid[0], R_upper_mid[1] - R_bottom_mid[1])

        #Now this is for left eye.......
    
        #horizontal line
        landmark = predictor(gray,face)
        L_Hori_left_x = landmark.part(42).x
        L_Hori_left_y = landmark.part(42).y
        L_Hori_left = (L_Hori_left_x,L_Hori_left_y)
        L_Hori_right_x = landmark.part(45).x
        L_Hori_right_y = landmark.part(45).y
        L_Hori_right = (L_Hori_right_x,L_Hori_right_y)
        cv2.line(frame,L_Hori_left,L_Hori_right,(0,255,0),1)

        #find midpoint for vertical line 
        L_upper_mid_x = int((landmark.part(43).x + landmark.part(44).x)/2)
        L_upper_mid_y = int((landmark.part(43).y + landmark.part(44).y)/2)
        L_upper_mid = (L_upper_mid_x,L_upper_mid_y)
        L_bottom_mid_x = int((landmark.part(47).x + landmark.part(46).x)/2)
        L_bottom_mid_y = int((landmark.part(47).y + landmark.part(46).y)/2)
        L_bottom_mid = (L_bottom_mid_x,L_bottom_mid_y)
        cv2.line(frame,L_upper_mid,L_bottom_mid,(0,255,0),1)

        #find the lenght for both horizontal and vertical line
        L_hori_lenght = hypot(L_Hori_left_x-L_Hori_right_x,L_Hori_left_y-L_Hori_right_y)
        L_ver_lenght = hypot(L_upper_mid[0]-L_bottom_mid[0],L_upper_mid[1]-L_bottom_mid[1])
        
        #find ratio for both right and left eye....
        L_ratio = L_hori_lenght/L_ver_lenght
        R_ratio = R_hori_lenght/R_ver_lenght


        #find average ratio.....

        ratio = (L_ratio + R_ratio)/2
        
        if ratio > 5:
           # count = count+1
            cv2.putText(frame,"Blinking",(50,150),font,1,(0,0,255),1)
           # count = 0
       
            
    cv2.imshow('frame',frame)
    #cv2.imshow('gray',gray)
    key = cv2.waitKey(1)

    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
