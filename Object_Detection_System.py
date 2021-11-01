"""
This code will check random images and their bounding boxes if they are correct or not

"""
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import random

#print("Block-1: Creating the Absolute Paths...")
# Getting the paths
PATH = os.getcwd()

CONFIG_FILE_PATH = os.path.join(PATH, "yolov3_test.cfg")
YOLO_CUSTOM_WEIGHTS_PATH = os.path.join(PATH, "yolov3_last.weights")

CONF_THRESH, NMS_THRESH = 0.5, 0.5

#print("Block-2: Reading the Darknet and Configuration Files...")
# Provide paths of config and trained model files for testing
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE_PATH, YOLO_CUSTOM_WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Give the classes names in a list
classes = ["AIRPLANE", "BIRD", "DRONE", "HELICOPTER"]

#print("Block-3: Reading the Input File...")
# Taking input (We can have 3 options: 1. Image, 2. Video, 3. Live Video)
file_name = "Videos/IR_HELICOPTER_006.mp4"
cap = cv2.VideoCapture(file_name)
# Getting the Frame Rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps/3.0

#print("Block-4: Creating the Video Writer...")
# Creating a VideoWrite for output video generation
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
output_filename = os.path.join(PATH, os.path.splitext(file_name)[0] + "_Processed_Output.mp4")
video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (320, 256))


while True:    
    ret, img = cap.read()
    if ret:
        img = cv2.resize(img, (320, 256))
        hight, width, _ = img.shape
    else:
        break
     
    blob = cv2.dnn.blobFromImage(img, 1/255, (width, hight), (0, 0, 0), swapRB = True, crop = False)
    net.setInput(blob)
    output_layers_name = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_name)
    #print("Block-5: Finding the objects and classes with confidence...")
    # Finding the objects and classes with confidence
    boxes =[]
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3] * hight)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    
    #print("Block-6: Removing the duplicate predictions...")
    # Removing the duplicated using NMSBoxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)
    
    #print("Block-7: Finding the objects and classes with confidence from unique detections...")
    # Finding the objects and classes with confidence from unique detections
    boxes =[]
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > CONF_THRESH:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * hight)
                w = int(detection[2] * width)
                h = int(detection[3]* hight)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
    
    #print("Block-8: Drawing rectangle around object with its class and confidence...")
    
    # Drawing rectangle around object with its class and confidence
    font = cv2.FONT_HERSHEY_PLAIN
    
    if  len(indexes)>0:
        for i in indexes.flatten():
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2) * 100)[:-2]
            
            # Selecting color based upon alert
            # Red - Drone, Gree - others
            if (label == "DRONE"):
                color = (0, 140, 255)
            else:
                color = (0, 255, 0)
            
            # Drawing the rectangle and Adding Class label
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            cv2.putText(img, label + " " + confidence + "%", (x, y - 5), font, 1, color, 1)            
     
    #print("Block-9: Writing Images to video...")
    video_writer.write(img)
    # Finally showing the output
    cv2.imshow('img', img)
    
    # Quitting upon pressing "Q"
    if cv2.waitKey(1) == ord('q'):
        break
#print("Block-10: Releasing all the objects...")
video_writer.release() 
cap.release()
cv2.destroyAllWindows()

