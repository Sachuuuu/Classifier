# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:01:24 2020

@author: SACHUU
"""

from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import os
import matplotlib.pyplot as plt


class detect_shop:
    
    def __init__(self):

        self.image = cv2.imread("E:/POJJA/SHOP using OCR/opencv-text-detection/images/j.jpg")
        self.origi = self.image
        self.crop = self.image.copy()
        self.H,self.W = self.image.shape[:2]
        self.h,self.w = (640,640)
        self.H_ratio,self.W_ratio = self.H/float(self.h),self.W/float(self.w)
        self.image = cv2.resize(self.image,(self.h,self.w))
        self.H,self.W = self.image.shape[:2]
        self.layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
        self.net = cv2.dnn.readNet('E:/POJJA/SHOP using OCR/opencv-text-detection/frozen_east_text_detection.pb')
        self.construct_blob()
        self.detect_strings()

        
        
    def construct_blob(self):
        print(os.getcwd())
        self.blob = cv2.dnn.blobFromImage(self.image, 1.0, (self.W, self.H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(self.blob)
        
    def detect_strings(self):
        (scores, geometry) = self.net.forward(self.layerNames)
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            
            for x in range(0, numCols):
                if scoresData[x] < 0.5:
                    continue
                
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
                
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        count = 0
        for (startX, startY, endX, endY) in boxes:
            startX = int(startX * self.W_ratio)
            startY = int(startY * self.H_ratio)
            endX = int(endX * self.W_ratio)
            endY = int(endY * self.H_ratio)
            cv2.rectangle(self.origi, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.imshow("Text Detection", self.origi)
            
            croped_image = self.crop[startX:endY,startY:endX]
            imagename = "croped_{}.jpg".format(count)
            cv2.imwrite(imagename,croped_image)
            count += 1
            cv2.waitKey(0)


g = detect_shop()
