import cv2
import mediapipe as mp
import numpy as np
import math
import time
import facedetection_funcs as fc_fun

from deepface.detectors import FaceDetector

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

detector_backend = "retinaface"

framecp = False

currentframe = 0

WHITE_COLOR = (255, 255, 255)

#return (k1,k2)
#calculates kernel size depending on the size of the face(with minimums)
def calc_kernel_size(pt1,pt2,k1_min,k2_min):
  k1 = math.floor((pt2[0]-pt1[0])/2)
  k2 = math.floor((pt2[1]-pt1[1])/2)
  if k1 < k1_min:
    k1 = k1_min
  if k2 < k2_min:
    k2 = k2_min

  return(k1,k2)

cap = cv2.VideoCapture("input.mp4")
success, current_frame = cap.read()
previous_frame = current_frame

pt1 = 0,0
pt2 = current_frame.shape[1],current_frame.shape[0]

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
movielength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

maxdatainframe = frame_width * frame_height * 255

k1_min = math.floor(frame_height*0.1)
k2_min = math.floor(frame_width*0.1)

#start recording the output into a file
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('m','p','4','v'), fps, (frame_width,frame_height))

while cap.isOpened():
  # Compare current frame to last frame with significant changes
  # run face detection only when there is a significant change in the frame
  image_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
  if framecp == False:
    previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)    

  frame_diff = cv2.absdiff(image_gray,previous_frame_gray)
  th ,frame_diff = cv2.threshold(frame_diff, 26, 255, cv2.THRESH_BINARY) #lower threshold has to be adjusted for camera noise

  frame_diff_value = np.sum(frame_diff)/maxdatainframe

  previous_frame = current_frame.copy()
  success, image = cap.read()
  current_frame = image.copy()
    
  if not success:
    # dont die if camera disconnects/lags shortly
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    break

  currentframe += 1
  print("Progress: " + str(currentframe) + "/" + str(movielength))
  
  print(frame_diff_value)
  if frame_diff_value > 0.05: # has to be tweaked according to the shot
    framecp = False
    image.flags.writeable = False # disable writing before sending to face detection. Its significantly faster that way for the mediapipe detector
    pt1,pt2 = fc_fun.detectface(image,frame_height,frame_width)
  else:
    framecp = True
    print("no significant changes")

  image.flags.writeable = True # reenable writing
  mask = np.zeros((frame_height,frame_width), dtype=np.uint8) # create a black frame
  cv2.rectangle(mask,pt1,pt2, WHITE_COLOR, -1)
  # blur image according to the mask
  blurred = cv2.blur(image,calc_kernel_size(pt1,pt2,k1_min,k2_min),0)
  blurred = cv2.bitwise_and(blurred,blurred,mask=mask)
  mask = cv2.bitwise_not(mask)
  image = cv2.bitwise_and(image,image,mask=mask)
  image = cv2.add(blurred,image)

  out.write(image) # write to the file
  cv2.imshow('Face Blurring Preview', image)
  cv2.imshow('Framediff Detection', frame_diff)

  if cv2.waitKey(5) & 0xFF == 27:
    break

# release camera and save file
cap.release()
out.release()