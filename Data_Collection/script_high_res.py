"""
This script requires a working instance of a Kinect v1 sensor connected to the machine with Microsoft Kinect for Windows SDK v1.8 installed.
This is the high resolution version (1280 x 1024) of the data collection script and captures one labelled data point every two seconds (This is changeable).
For samples images throwing light on what the script does and how the script captures labelled data points for finger pointing direction estimation, please see the images present in the directory.
This script runs using the Pykinect wrapper for python around Microsoft Kinect for Windows SDK v1.8 and will require Pyhton 2.x
"""

import thread
import itertools
import ctypes

import pykinect
from pykinect import nui
from pykinect.nui import JointId

import pygame
from pygame.color import THECOLORS
from pygame.locals import *
import numpy as np
import cv2

import time
import os

video = 0

KINECTEVENT = pygame.USEREVENT
DEPTH_WINSIZE = 320,240
VIDEO_WINSIZE = 640,480
pygame.init()

INTERVAL = 2
START_TIME = int(time.time())
FLAG = True

FONT = cv2.FONT_HERSHEY_SIMPLEX

vid_update = 0

l = os.listdir("./IMAGE_DATA/")

def file_num(file_name):
    return int(file_name[0:-4])

if not l:
    IMG_NUM = 1
    IMG_NUM_INIT = IMG_NUM
else:
    # print(l)
    l.sort(key = file_num)
    # print(l)
    print(l[-1])
    last_file_name = l[-1]
    last_image_num = int(last_file_name[0:-4])
    IMG_NUM = last_image_num + 1
    IMG_NUM_INIT = IMG_NUM

"""Remember: OpenCV uses BGR instead of RGB"""
SKELETON_COLORS = [THECOLORS["red"][0:3], 
                   THECOLORS["blue"][0:3], 
                   THECOLORS["green"][0:3], 
                   THECOLORS["orange"][0:3], 
                   THECOLORS["purple"][0:3], 
                   THECOLORS["yellow"][0:3], 
                   THECOLORS["violet"][0:3]]

print(SKELETON_COLORS[0][0])
print(type(SKELETON_COLORS[0][0]))

LEFT_ARM = (JointId.ShoulderCenter, 
            JointId.ShoulderLeft, 
            JointId.ElbowLeft, 
            JointId.WristLeft, 
            JointId.HandLeft)
RIGHT_ARM = (JointId.ShoulderCenter, 
             JointId.ShoulderRight, 
             JointId.ElbowRight, 
             JointId.WristRight, 
             JointId.HandRight)
LEFT_LEG = (JointId.HipCenter, 
            JointId.HipLeft, 
            JointId.KneeLeft, 
            JointId.AnkleLeft, 
            JointId.FootLeft)
RIGHT_LEG = (JointId.HipCenter, 
             JointId.HipRight, 
             JointId.KneeRight, 
             JointId.AnkleRight, 
             JointId.FootRight)
SPINE = (JointId.HipCenter, 
         JointId.Spine, 
         JointId.ShoulderCenter, 
         JointId.Head)

skeleton_to_depth_image = nui.SkeletonEngine.skeleton_to_depth_image

def nearer_hand_id(left_hand_distance, right_hand_distance):
    if left_hand_distance < right_hand_distance:
        return JointId.HandLeft
    else:
        return JointId.HandRight

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
############################################################
############################################################
############################################################
############If you change the capture resolution,###########
##########remember to change the skeletion drawing##########
###############scale in the function below.#################
############################################################
############################################################
############################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def draw_skeleton_data(pSkelton, index, positions, width = 4):

    global video
    start = pSkelton.SkeletonPositions[positions[0]]
    start_index = positions[0]

    nearer_hand = nearer_hand_id(pSkelton.SkeletonPositions[JointId.HandLeft].z, pSkelton.SkeletonPositions[JointId.HandRight].z)


    for position in itertools.islice(positions, 1, None):
        
        next = pSkelton.SkeletonPositions[position.value]
        next_index = position.value
        
        curstart = skeleton_to_depth_image(start, 1280, 1024)
        curend = skeleton_to_depth_image(next, 1280, 1024)

        curstart = (int(curstart[0]), int(curstart[1]))
        curend = (int(curend[0]), int(curend[1]))

        if nearer_hand == JointId.HandLeft:
            if ((start_index == JointId.ElbowLeft and next_index == JointId.WristLeft) or (start_index == JointId.WristLeft and next_index == JointId.ElbowLeft)):
                cv2.line(video, curstart, curend, SKELETON_COLORS[2], 3)
            elif ((start_index == JointId.WristLeft and next_index == JointId.HandLeft) or (start_index == JointId.HandLeft and next_index == JointId.WristLeft)):
                cv2.line(video, curstart, curend, SKELETON_COLORS[0], 3)
            else:
                cv2.line(video, curstart, curend, SKELETON_COLORS[1], 3)

        if nearer_hand == JointId.HandRight:
            if ((start_index == JointId.ElbowRight and next_index == JointId.WristRight) or (start_index == JointId.WristRight and next_index == JointId.ElbowRight)):
                cv2.line(video, curstart, curend, SKELETON_COLORS[2], 3)
            elif ((start_index == JointId.WristRight and next_index == JointId.HandRight) or (start_index == JointId.HandRight and next_index == JointId.WristRight)):
                cv2.line(video, curstart, curend, SKELETON_COLORS[0], 3)
            else:
                cv2.line(video, curstart, curend, SKELETON_COLORS[1], 3)
        
        start = next
        start_index = next_index


if hasattr(ctypes.pythonapi, 'Py_InitModule4'):
    Py_ssize_t = ctypes.c_int
elif hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
    Py_ssize_t = ctypes.c_int64
else:
   raise TypeError("Cannot determine type of Py_ssize_t")


_PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
_PyObject_AsWriteBuffer.restype = ctypes.c_int
_PyObject_AsWriteBuffer.argtypes = [ctypes.py_object,
                                  ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.POINTER(Py_ssize_t)]

def surface_to_array(surface):
   buffer_interface = surface.get_buffer()
   address = ctypes.c_void_p()
   size = Py_ssize_t()
   _PyObject_AsWriteBuffer(buffer_interface,
                          ctypes.byref(address), ctypes.byref(size))
   bytes = (ctypes.c_byte * size.value).from_address(address.value)
   bytes.object = buffer_interface
   return bytes

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
############################################################
############################################################
############################################################
############If you change the capture resolution,###########
##########remember to change the skeletion drawing##########
###############scale in the function below.#################
############################################################
############################################################
############################################################
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

def draw_skeletons(skeletons):
    global video
    for index, data in enumerate(skeletons):
        HeadPos = skeleton_to_depth_image(data.SkeletonPositions[JointId.Head], 1280, 1024)
        if HeadPos[0] == 0 and HeadPos[1] == 0:
            continue
        draw_skeleton_data(data, index, SPINE, 10)
        cv2.circle(video, (int(HeadPos[0]), int(HeadPos[1])), 20, SKELETON_COLORS[1], -1)
    
        # Drawing the limbs.
        Draw_skeleton_data(data, index, LEFT_ARM)
        draw_skeleton_data(data, index, RIGHT_ARM)
        draw_skeleton_data(data, index, LEFT_LEG)
        draw_skeleton_data(data, index, RIGHT_LEG)


def depth_frame_ready(frame):
  # print "dFrame"
  if video_display:
    return

  with screen_lock:
    address = surface_to_array(screen)
    frame.image.copy_bits(address)
    del address
    if skeletons is not None and draw_skeleton:
      draw_skeletons(skeletons)
    pygame.display.update()    


surface_capture = pygame.Surface(DEPTH_WINSIZE, 0, 16)

def recorded_hand(near):
    if near == JointId.HandLeft:
        return "LEFT HAND"
    if near == JointId.HandRight:
        return "RIGHT HAND"

def video_frame_ready(frame):
    global INTERVAL
    global START_TIME
    global FLAG
    global IMG_NUM
    # print "vFrame"
    global video
    if not video_display:
        return

    number_of_images_captured = IMG_NUM - IMG_NUM_INIT

    video = np.empty((1024, 1280, 4), np.uint8)
    frame.image.copy_bits(video.ctypes.data)
    time_difference = int(time.time()) - START_TIME
    if skeletons is not None and draw_skeleton:
        draw_skeletons(skeletons)

        cv2.putText(video, str(time_difference % INTERVAL), (10, 55), FONT, 2, (255,255,255), 3, cv2.CV_AA) # Puts a modulus timer on the video stream.

        if (time_difference % INTERVAL == 0) & FLAG:
            video_hold_out = np.empty((1024, 1280, 4), np.uint8)
            frame.image.copy_bits(video_hold_out.ctypes.data)

            print("SAVING_IMAGE " + str(IMG_NUM) + "...")

            cv2.rectangle(video, (2, 2), (1278, 1022), (255,255,255), 10) # Flashes a green rectangle around the stream to indicate a capture.

            video_frame = video_hold_out[:, :, 0:3]
            cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite("IMAGE_DATA/" + str(IMG_NUM) + ".PNG", video_frame)

            for index, data in enumerate(skeletons):
                HeadPos = skeleton_to_depth_image(data.SkeletonPositions[JointId.Head], 1280, 1024)
                if HeadPos[0] != 0 or HeadPos[1] != 0: # Think about what would happen if two people are detected.
                    joints = data.SkeletonPositions
                    nearer_hand = nearer_hand_id(joints[JointId.HandLeft].z, joints[JointId.HandRight].z)

                    skeleton_buffer_elbow_wrist = np.zeros(shape = (1, 8))
                    skeleton_buffer_elbow_wrist[0] = IMG_NUM

                    skeleton_buffer_wrist_hand = np.zeros(shape = (1, 8))
                    skeleton_buffer_wrist_hand[0] = IMG_NUM
                    
                    if nearer_hand == JointId.HandLeft:
                    	"""Assert left hand capture."""
                    	skeleton_buffer_elbow_wrist[0][7] = float(-1)
                        """Capture the elbow-wrist set of points"""
                        skeleton_buffer_elbow_wrist[0][6] = joints[JointId.WristLeft].z
                        skeleton_buffer_elbow_wrist[0][5] = joints[JointId.WristLeft].y
                        skeleton_buffer_elbow_wrist[0][4] = joints[JointId.WristLeft].x
                        skeleton_buffer_elbow_wrist[0][3] = joints[JointId.ElbowLeft].z
                        skeleton_buffer_elbow_wrist[0][2] = joints[JointId.ElbowLeft].y
                        skeleton_buffer_elbow_wrist[0][1] = joints[JointId.ElbowLeft].x

                        """Assert left hand capture."""
                    	skeleton_buffer_wrist_hand[0][7] = float(-1)
                        """Capture the wrist-hand set of points"""
                        skeleton_buffer_wrist_hand[0][6] = joints[JointId.HandLeft].z
                        skeleton_buffer_wrist_hand[0][5] = joints[JointId.HandLeft].y
                        skeleton_buffer_wrist_hand[0][4] = joints[JointId.HandLeft].x
                        skeleton_buffer_wrist_hand[0][3] = joints[JointId.WristLeft].z
                        skeleton_buffer_wrist_hand[0][2] = joints[JointId.WristLeft].y
                        skeleton_buffer_wrist_hand[0][1] = joints[JointId.WristLeft].x

                    elif nearer_hand == JointId.HandRight:
                    	"""Assert left hand capture."""
                    	skeleton_buffer_elbow_wrist[0][7] = float(1)
                        """Capture the elbow-wrist set of points"""
                        skeleton_buffer_elbow_wrist[0][6] = joints[JointId.WristRight].z
                        skeleton_buffer_elbow_wrist[0][5] = joints[JointId.WristRight].y
                        skeleton_buffer_elbow_wrist[0][4] = joints[JointId.WristRight].x
                        skeleton_buffer_elbow_wrist[0][3] = joints[JointId.ElbowRight].z
                        skeleton_buffer_elbow_wrist[0][2] = joints[JointId.ElbowRight].y
                        skeleton_buffer_elbow_wrist[0][1] = joints[JointId.ElbowRight].x

                        """Assert right hand capture."""
                    	skeleton_buffer_wrist_hand[0][7] = float(1)
                        """Capture the wrist-hand set of points"""
                        skeleton_buffer_wrist_hand[0][6] = joints[JointId.HandRight].z
                        skeleton_buffer_wrist_hand[0][5] = joints[JointId.HandRight].y
                        skeleton_buffer_wrist_hand[0][4] = joints[JointId.HandRight].x
                        skeleton_buffer_wrist_hand[0][3] = joints[JointId.WristRight].z
                        skeleton_buffer_wrist_hand[0][2] = joints[JointId.WristRight].y
                        skeleton_buffer_wrist_hand[0][1] = joints[JointId.WristRight].x

                    print("SAVING SKELETON DATA FOR " + recorded_hand(nearer_hand) + "...")

                    with open("_elbow_wrist.txt", "ab+") as f:
                        np.savetxt(f, skeleton_buffer_elbow_wrist)
                        f.write(b'\r\n')

                    with open("_wrist_hand.txt", "ab+") as f:
                        np.savetxt(f, skeleton_buffer_wrist_hand)
                        f.write(b'\r\n')
                    # break # un-comment this line for saving skeleton data for only one person.



            IMG_NUM += 1
            FLAG = False
        if (time_difference % INTERVAL == INTERVAL - 1):
            FLAG = True
    cv2.putText(video, "IMAGES CAPTURED:" + str(number_of_images_captured), (500, 55), FONT, 2, (0, 255, 0), 3, cv2.CV_AA)
    cv2.imshow('KINECT Video Stream', video)

def main():
	full_screen = False
    draw_skeleton = True
    video_display = False

    screen_lock = thread.allocate()

    screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)    
    pygame.display.set_caption('Python Kinect Demo')
    skeletons = None
    screen.fill(THECOLORS["black"])

    kinect = nui.Runtime()
    kinect.skeleton_engine.enabled = True
    def post_frame(frame):
        try:
            pygame.event.post(pygame.event.Event(KINECTEVENT, skeletons = frame.SkeletonData))
        except:
            # event queue full
            pass
        
    kinect.skeleton_frame_ready += post_frame

    kinect.depth_frame_ready += depth_frame_ready    
    kinect.video_frame_ready += video_frame_ready    
    
    kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution1280x1024, nui.ImageType.Color)
    kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution320x240, nui.ImageType.Depth)

    print('Controls: ')
    print('     d - Switch to depth view')
    print('     v - Switch to video view')
    print('     s - Toggle displaing of the skeleton')
    print('     u - Increase elevation angle')
    print('     j - Decrease elevation angle')

    # main game loop
    done = False

    while not done:
        e = pygame.event.wait()
        dispInfo = pygame.display.Info()
        if e.type == pygame.QUIT:
            done = True
            break
        elif e.type == KINECTEVENT:
            skeletons = e.skeletons
            if draw_skeleton:
                draw_skeletons(skeletons)
                pygame.display.update()
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                done = True
                break
            elif e.key == K_d:
                with screen_lock:
                    screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)
                    video_display = False
            elif e.key == K_v:
                with screen_lock:
                    cv2.namedWindow('KINECT Video Stream', cv2.WINDOW_AUTOSIZE)
                    screen = pygame.display.set_mode(DEPTH_WINSIZE,0,16)
                    # screen = pygame.display.set_mode(VIDEO_WINSIZE,0,32)    
                    video_display = True
            elif e.key == K_s:
                draw_skeleton = not draw_skeleton
            elif e.key == K_u:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle + 2
            elif e.key == K_j:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle - 2
            elif e.key == K_x:
                kinect.camera.elevation_angle = 2


if __name__ == '__main__':
    main()