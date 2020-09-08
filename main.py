# *__________Hussain Salih_____________*
#  This script function for get facenet 
# *____________________________________*

import cv2
import numpy as np
import dlib

img = cv2.imread("elon.jpg")
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

faces = detector(img_gray)

for face in faces:
    landmarks = predictor(img_gray,face)
    landmark_point = []
    for num in range(0,68):
        x = landmarks.part(num).x
        y = landmarks.part(num).y
        landmark_point.append((x,y))
            
    points = np.array(landmark_point,np.int32)
    convexhull = cv2.convexHull(points)
    
    
    
    # to show the line border of face
    # cv2.polylines(img,[convexhull],True,(255,0,0),3)
    
    #show the image in mask
    cv2.fillConvexPoly(mask,convexhull,255)
    face_image_1 = cv2.bitwise_and(img,img,mask=mask)
    
    #Delaunty triangulation
    rect = cv2.boundingRect(convexhull) #the rect of face
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmark_point)
    traiangles = subdiv.getTriangleList()
    traiangles = np.array(traiangles,dtype=np.int32)
    for t in traiangles:
        pt1 = (t[0],t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])
        
        cv2.line(img,pt1,pt2,(0,0,255),2)
        cv2.line(img,pt2,pt3,(0,0,255),2)
        cv2.line(img,pt1,pt3,(0,0,255),2)
    
cv2.imshow('Masked Face',img)


cv2.waitKey(0)
cv2.distroyAllWindows()
