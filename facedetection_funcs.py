import cv2
import mediapipe as mp
import numpy as np
import math
import time

from deepface.detectors import FaceDetector

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

detector_backend = "retinaface"
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# adapter function that uses multiple face detection libs to output two corners for the blur box
def detectface(image,frame_height,frame_width,highlevelonly=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    highlevel = False
    noface = False
    results = face_detection.process(image)
    face_detector = FaceDetector.build_model(detector_backend)
    if results.detections == None or highlevelonly == True:
      detected_face, img_region = FaceDetector.detect_face(face_detector, detector_backend, image, align = False)
      highlevel = True
      if img_region[0] == 0 and img_region[1] == 0 and img_region[2] == image.shape[0] and img_region[3] == image.shape[1]:
        highlevel = False
        noface = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.detections and highlevelonly == False:
      bestdetection = results.detections[0]
      image_rows, image_cols, _ = image.shape
      location = bestdetection.location_data
      relative_bounding_box = location.relative_bounding_box

      rect_start_point1 = image_cols * relative_bounding_box.xmin
      rect_start_point2 = image_rows * relative_bounding_box.ymin

      rect_end_point1 = image_cols * (relative_bounding_box.xmin + relative_bounding_box.width)
      rect_end_point2 = image_rows * (relative_bounding_box.ymin + relative_bounding_box.height)

      pt1 = int(rect_start_point1),int(rect_start_point2)
      pt2 = int(rect_end_point1),int(rect_end_point2)
    if highlevel:
      print("High level Detection used")
      pt1 = img_region[0], img_region[1]
      pt2 = img_region[0] + img_region[2], img_region[1] + img_region[3]
    if noface:
      #if there is no face, then indicate that the whole frame should be blurred 
      print("No face found")
      pt1 = 0,0
      pt2 = image.shape[1],image.shape[0]

    return pt1,pt2