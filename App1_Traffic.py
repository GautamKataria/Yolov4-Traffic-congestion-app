
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations


def is_close(p1, p2):
    """
    :return:
    dst = Euclidean Distance between two 2d points
    """
    dst = math.sqrt(p1**2 + p2**2)

    return dst 


def convertBack(x, y, w, h): 

    """
    :param:
    x, y = midpoint of bounding box
    w, h = width, height of the bounding box
    
    :return:
    xmin, ymin, xmax, ymax
    rectangle coordinates
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    """
    self explanatory
    :param:
    detections = total detections in one frame
    img = image from detect_image method of darknet

    :return:
    img with bbox
    """

    if len(detections) > 0:
        congestion_count = 0
        centroid_dict = dict()
        objectId = 0
        for detection in detections:

            name_tag = str(detection[0].decode())
            a_string = "A string is more than its parts!"
            matches = ['car', 'truck', 'bus','motorbike','bicycle']

            if (name_tag in matches):


                x, y, w, h = detection[2][0],\
                            detection[2][1],\
                            detection[2][2],\
                            detection[2][3]
                xmin, ymin, xmax, ymax = convertBack(float(x), float(y), float(w), float(h))
                centroid_dict[objectId] = (int(x), int(y), xmin, ymin, xmax, ymax)
                objectId += 1

        ## Finding which boxes are closest to each other
        red_zone_list = [] # List containing which Object id is in under threshold distance condition.
        red_line_list = []

        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = is_close(dx, dy)
            if (distance < 80.0):
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                    red_line_list.append(p1[0:2])
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)
                    red_line_list.append(p2[0:2])

        ## Drawing boxes
        for idx, box in centroid_dict.items():
            if idx in red_zone_list:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (255, 0, 0), 2)
            else:
                cv2.rectangle(img, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)

        ## Showing stats
        congestion_count = objectId
        congestion_jam = ["no", "yes"]
        if(len(red_zone_list) > 16):
            var1234 = 1
        else:
            var1234 = 0
        congestion_state = ["light traffic", "moderate traffic", "heavy traffic"]
        if (congestion_count <= 22):
            var123 = 0
        elif (congestion_count > 22 and congestion_count <=30):
            var123 = 1
        else:
            var123 = 2

        font_scale = 0.75
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Traffic condition: %s , number: %d , Traffic jam on scene : %s  , number: %d" %(congestion_state[var123], congestion_count, congestion_jam[var1234], len(red_zone_list))			# Count People at Risk
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

        box_coords = ((10,30), (1280, 25 - text_height -10))
        location = (10, 25)												# Set the location of the displayed text
        cv2.rectangle(img, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
        cv2.putText(img, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,10,10), 2, cv2.LINE_AA)  # Display Text

        for check in range(0, len(red_line_list)-1):					# Draw line between nearby bboxes iterate through redlist items
            start_point = red_line_list[check] 
            end_point = red_line_list[check+1]
            check_line_x = abs(end_point[0] - start_point[0])   		# Calculate the line coordinates for x  
            check_line_y = abs(end_point[1] - start_point[1])			# Calculate the line coordinates for y
            if (check_line_x < 80) and (check_line_y < 25):				# If both are We check that the lines are below our threshold distance.
                cv2.line(img, start_point, end_point, (255, 0, 0), 2)   # Only above the threshold lines are displayed. 

    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov4.cfg"
    weightPath = "./yolov4.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    name123 = 'traffic_congestion'
    cap = cv2.VideoCapture(F"./Input/{name123}.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = (frame_height // 2) * 4, (frame_width // 2)* 4
    print (new_width)

    out = cv2.VideoWriter(
            F"./demo/{name123}_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15.0,
            (new_width, new_height))
    
    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)
    
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.35)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")

if __name__ == "__main__":
    YOLO()
