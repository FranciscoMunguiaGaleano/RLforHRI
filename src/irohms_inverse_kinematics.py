#!/usr/bin/env python3



from dis import dis
from simple_script_server import *
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray,MultiArrayDimension 
from tf.transformations import quaternion_from_euler
from cvzone.HandTrackingModule import HandDetector
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf
import numpy as np
import actionlib
import time
import cv2
import math
import matplotlib.pyplot as plt
from scipy import interpolate
import tensorflow as tf
import random
import yaml
import requests
import pyrealsense2 as rs
import torch
import pandas as pd
import re
import threading


fps=5
SH=320
SW=480
PIXELS_TO_METERS=500
ROBOT={'L1':1.5*PIXELS_TO_METERS,'L2':1.5*PIXELS_TO_METERS}
AXIS_LENGHT=20
AXIS_WIDTH=2
RED=[0,0,255]
GREEN=[0,255,0]
BLUE=[255,0,0]
BLUE_GREEN=[255, 255, 0]
YELLOW=[255,0,255]
RED=[0,0,255]
GREEN=[0,255,0]
BLUE=[255,0,0]
BLUE_GREEN=[255, 255, 0]
YELLOW=[0,255,255]

pipeline = rs.pipeline()
config = rs.config()
config.enable_device('141722074327')
objects={}
ref_position=[0,0,0]
sss = simple_script_server()


def recognise_instructions(words):
    pos_list = ['left', 'right', 'here', 'there']
    shape_list = ['circular', 'square','hexagonal','triangular']
    color_list = ['blue', 'red', 'green', 'yellow']
    words=words.lower()
    ins_sentences = re.split(r'move|put|let|get',words)[1::]
    ##### get instruction keys ######
    instructions = []
    for sentence in ins_sentences:
        Goal_pos=None
        aimed_objects=[]
        # obj_descriptions=sentence.split('objects','ones')
        obj_descriptions = re.split(r"; |, |\*|\n|object|one", sentence)
        for description in obj_descriptions:
            shape, color = '*', '*'
            obj_keys = description.split()
            for key in obj_keys:
                if key in pos_list:
                    Goal_pos=key
                elif key in shape_list:
                    shape=key
                elif key in color_list:
                    color=key
            if color!="*"  or shape!='*':
                aimed_objects.append([color[0]+"_"+shape[0]])
        instructions.append([aimed_objects, Goal_pos])
    return instructions



def get_color(name):
    color=[255,255,255]
    if name == 'b':
        color=BLUE
    elif name =='y':
        color=YELLOW
    elif name=='g':
        color=GREEN
    elif name=='r':
        color=RED
    return color
def tagger(img, image,hand1,hand2,name):
    original = img.copy()
    #name="hand"
    hand_2=None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    i=0
    types=""
    
    objective={}
    if hand1 is not None:
            #lmList2 = hand2["lmList"]  # List of 21 Landmarks points
            bbox2 = hand1["bbox"]  # Bounding Box info x,y,w,h
            centerPoint2 = hand1["center"]  # center of the hand cx,cy
            #handType2 = hand2["type"] 
            handType2="left"
            cv2.rectangle(image, (bbox2[0], bbox2[1]), (bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]), (0,255,0), 1)
            cv2.circle(image, (centerPoint2[0], centerPoint2[1]), 2, (255, 255, 255), -1)
            cv2.putText(image, str(handType2), (centerPoint2[0], centerPoint2[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    if hand2 is not None:
            bbox2 = hand2["bbox"]  # Bounding Box info x,y,w,h
            centerPoint2 = hand2["center"]  # center of the hand cx,cy
            #handType2 = hand2["type"] 
            handType2="right"
            cv2.rectangle(image, (bbox2[0], bbox2[1]), (bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]), (0,255,0), 1)
            cv2.circle(image, (centerPoint2[0], centerPoint2[1]), 2, (255, 255, 255), -1)
            cv2.putText(image, str(handType2), (centerPoint2[0], centerPoint2[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    try:
        for c in cnts:
            w=max([xmax[0][0] for xmax in c])-min([xmax[0][0] for xmax in c])
            l=max([xmax[0][1] for xmax in c])-min([xmax[0][1] for xmax in c])
            approx = cv2.approxPolyDP(c,0.055*cv2.arcLength(c,True),True)
            if len(approx)==5:
                types="pentagon"
            elif len(approx)==3:
                types="triangle"
            elif len(approx)==4:
                types="square"
            elif len(approx) == 6:
                types="hexa"
            #elif len(approx) == 8:
            #    types="octa"
            elif len(approx) > 6:
                types="circle"
            else:
                types=""
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            color=[255,255,255]
            #print("Name: ",name," shape: ",types,"id: ",i," X: ",w,"y: ",l, "Center: ",cX ,cY,"color: ",color)
            #'box1':{'position':[342,500,0],'orientation':[0,0,0.0],'lenght':50,'width':50,'height':50,'shape':'rectangle','color':(255,255,255),'path':[],'grid':[],'path_smooth':[],'goal':[]
            #save goal
            if name=='b':
                types="circle"
            if name=='y':
                types="hexa"
            if name=='g':
                types="square"
            if name=='goal':
                if types=="square":
                    objective['right']=[cX,cY+40,0]
                    print("Goal right detected")
                    with open('/home/robot/Desktop/IDQN/goal_1.yaml', 'w') as file:
                        documents = yaml.dump([objective], file)
                else:
                    objective['left']=[cX,cY+40,0]
                    print("Goal left detected")
                    with open('/home/robot/Desktop/IDQN/goal_2.yaml', 'w') as file:
                        documents = yaml.dump([objective], file)
                
            ## save goal position
            #if name=='hand':
            ## save hand position
            ##    object:
            # with open('/home/robot/Desktop/IDQN/goal_1.yaml') as file:
            #     goal = yaml.load(file, Loader=yaml.FullLoader)
            #     x_goal_1=goal[0]['right'][0]
            #     y_goal_1=goal[0]['right'][1]
                
            color=get_color(name)
            if name!='goal' and name!='hand'  and l.astype(float)> 6 and w.astype(float)>6:
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]={}
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['position']=[cX,cY,0]
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['orientation']=[0,0,0]
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['lenght']=float(l.astype(float))
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['width']=float(w.astype(float))
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['height']=50
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['shape']=types    
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['color']=color
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['path']=[]
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['grid']=[]
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['path_smooth']=[]
                objects[name+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['goal']=[]
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.circle(image, (cX, cY), 2, (255, 255, 255), -1)
            cv2.putText(image, name+"_"+str(int(cX/10))+"_"+str(int(cY/10))+"_"+types, (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            i+=1
        if len(objects)>0:
            #print(objects)
            with open('/home/robot/Desktop/IDQN/objects_1.yaml', 'w') as file:
                documents = yaml.dump([objects], file)
        #if len(objective)>0:
        #    #print(objective)
        #    with open('/home/robot/Desktop/IDQN/goal_1.yaml', 'w') as file:
        #        documents = yaml.dump([objective], file)
        
        #print(obj)
    except Exception as e:
        #print("Error: ",e)
        return image
    return image
def observe_environment():
    x1=100
    y1=100
    # Configure depth and color streams
    #pipeline = rs.pipeline()
    #config = rs.config()
    #config.enable_device('141722074327')

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("This equires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    try:
        time=0
        ####TODO######
        #mpHands = mp.solutions.hands
        #hands = mpHands.Hands()
        #mpDraw = mp.solutions.drawing_utils
        #pTime = 0
        #cTime = 0 
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue
            # crop images
            x, y, w, h = 100, 50, 320, 480
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())[x:x+w, y:y+h]
            color_image = np.asanyarray(color_frame.get_data())[x:x+w, y:y+h]

            height, width= depth_image.shape
            #print(height, width)
            depth_image = depth_image[y1:height-y1+10, x1:width-130]
            height, width = depth_image.shape
            #print(height, width)
            #break
            # Apply colormap on depth image (image must be converted to 8-bit per pixel f0.irst)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.06), cv2.COLORMAP_JET)

            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
            #print(color_colormap_dim)
            #break

            # If depth and color resolutions are different, resize color image to match depth image for display
            
            if depth_colormap_dim != color_colormap_dim:
                depth_colormap = cv2.resize(depth_colormap, dsize=(color_colormap_dim[1], color_colormap_dim[0]), interpolation=cv2.INTER_AREA)
                hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                lower = np.array([0,0,122])
                upper = np.array([179,255,255])
                mask = cv2.inRange(hsv, lower, upper)
                result = cv2.bitwise_and(color_image, color_image, mask = mask)
                #print("here",time)
                if time>40:
                    #PINK
                    #(hMin = 146 , sMin = 65, vMin = 148), (hMax = 179 , sMax = 255, vMax = 255)
                    #(hMin = 166 , sMin = 64, vMin = 170), (hMax = 174 , sMax = 255, vMax = 255)
                    img=color_image.copy()
                    lower=np.array([166,64,170])
                    upper=np.array([174,255,255])
                    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)
                    result_pink = cv2.bitwise_and(result, result, mask = mask)
                    #(hMin = 0 , sMin = 72, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)
                    lower=np.array([0,72,0])
                    upper=np.array([179,255,255])
                    hsv = cv2.cvtColor(result_pink, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)
                    result_pink = cv2.bitwise_and(result_pink, result_pink, mask = mask)
                    color_image=tagger(result_pink, color_image, None, None,"goal")
                    #BLUE
                    #(hMin = 65 , sMin = 37, vMin = 0), (hMax = 113 , sMax = 255, vMax = 255)
                    lower=np.array([65,37,0])
                    upper=np.array([113,255,255])
                    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)
                    result_blue = cv2.bitwise_and(result, result, mask = mask)
                    color_image=tagger(result_blue, color_image, None, None,"b")
                    #GREEN
                    #(hMin = 33 , sMin = 28, vMin = 69), (hMax = 74 , sMax = 230, vMax = 255)
                    lower=np.array([33,28,69])
                    upper=np.array([74,230,255])
                    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)
                    result_green = cv2.bitwise_and(result, result, mask = mask)
                    color_image=tagger(result_green, color_image, None, None,"g")
                    #YELLOW
                    #(hMin = 19 , sMin = 34, vMin = 0), (hMax = 35 , sMax = 230, vMax = 255)
                    lower=np.array([19,34,0])
                    upper=np.array([35,230,255])
                    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)
                    result_yellow = cv2.bitwise_and(result, result, mask = mask)
                    #(hMin = 20 , sMin = 37, vMin = 169), (hMax = 51 , sMax = 222, vMax = 255)
                    lower=np.array([20,37,169])
                    upper=np.array([51,222,255])
                    hsv = cv2.cvtColor(result_yellow, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)
                    result_yellow = cv2.bitwise_and(result_yellow, result_yellow, mask = mask)
                    color_image=tagger(result_yellow, color_image, None, None,"y")
                    #RED
                    #(hMin = 0 , sMin = 119, vMin = 0), (hMax = 7 , sMax = 255, vMax = 255)
                    lower=np.array([0,119,0])
                    upper=np.array([7,255,255])
                    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower, upper)
                    result_red = cv2.bitwise_and(result, result, mask = mask)
                    color_image=tagger(result_red, color_image, None, None,"r")
                    
                    ######
                    #####TODO ######
                    '''QINGMENG'''
                    hands, img = detector.findHands(img)  # With Draw
                    hand1=None
                    hand2=None
                    if hands:
                        hand1 = hands[0]
                        if len(hands) == 2:
                            hand2 = hands[1]
                    '''END QINGMENG'''
                    color_image=tagger(result_red, color_image, hand1,hand2,"hand")
                    images = np.hstack((color_image,result,img,result_pink))
                else:
                    images = np.hstack((color_image,result,depth_colormap))
            else:
                images = np.hstack((color_image,result,depth_colormap))
            # Show images
            #print("here")
            cv2.namedWindow('IDQL environment', cv2.WINDOW_AUTOSIZE)
            #print("here")
            cv2.imshow('IDQL environment', images)
            #print("here")
            time+=1
            key=cv2.waitKey(1)
            if time > 80:
                break
            if cv2.getWindowProperty('IDQL environment', cv2.WND_PROP_VISIBLE) <1:
                break
        cv2.destroyAllWindows()
    finally:
        # Stop streaming
        pipeline.stop()
def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)



class armController():
    def __init__(self):
        self.right_arm_state = [0,0,0,0,0,0,0]
        self.torso_state=[0,0]
        rospy.Subscriber('/arm_right/joint_states', JointState, self.rightback)
        rospy.Subscriber('/torso/joint_states', JointState, self.torsoback)
        self.right_pub_vel = rospy.Publisher('/arm_right/joint_group_position_controller/command', Float64MultiArray, queue_size=10)
        self.torso_pub_vel = rospy.Publisher('/torso/joint_group_position_controller/command', Float64MultiArray, queue_size=10)

    def position_controller(self,pos,joint):
        k=1.0
        p=0.1
        i=0.1
        threshold=0.01
        u=[0,0,0,0,0,0,0]
        error=pos-self.right_arm_state[joint]
        rate = rospy.Rate(100)
        p_f=self.right_arm_state[joint]
        p_i=self.right_arm_state[joint]
        while abs(error)>threshold:
            u[joint]=k*error+p*(p_f-p_i)*error
            msg=Float64MultiArray()
            dim = MultiArrayDimension()
            msg.data = u
            dim.size = len(msg.data)
            dim.label = "command"
            dim.stride = len(msg.data)
            msg.layout.dim.append(dim)
            msg.layout.data_offset = 0
            self.right_pub_vel.publish(msg)
            error=pos-self.right_arm_state[joint]
            p_f=self.right_arm_state[joint]
            p_i=p_f
            ###DEBUG###
            print(error)
            ######
            rate.sleep()  
    def torso_controller(self,pos):
        k=30.0
        p=0.0
        i=0.0
        threshold=0.02
        u=[0,0]
        error=pos-self.torso_state[1]
        error_f=error
        rate = rospy.Rate(100)
        integral=0
        while abs(error)>threshold:
            integral+=integral+(error-error_f)+i*integral
            u[1]=(k*error+p*(error-error_f)+integral)
            error_f=error
            msg=Float64MultiArray()
            dim = MultiArrayDimension()
            msg.data = u
            dim.size = len(msg.data)
            dim.label = "command"
            dim.stride = len(msg.data)
            msg.layout.dim.append(dim)
            msg.layout.data_offset = 0
            self.torso_pub_vel.publish(msg)
            error=pos-self.torso_state[1]
            
            ###DEBUG###
            print(pos,error,u[1])
            ######
            rate.sleep()


    def rightback(self,data):
        right_arm_state=data.position
        #rospy.loginfo(data.position)
        #rospy.loginfo(right_arm_state)
    def torsoback(self,data):
        self.torso_state=data.position
        #rospy.loginfo(data.position)

class IDQL():
    def __init__(self,objects):
        self.objects=objects
        self.reset()
    def reset(self):
        self.work_space = np.zeros((SH,SW))
        self.work_space = cv2.merge([self.work_space,self.work_space,self.work_space])

    def set_env(self,key,goal):
        temp_objects=self.objects.copy()
        temp_objects.pop(key,None)
        iQtable=[]
        iEnv={}
        self.objects[key]['goal']=goal
        x_ref=self.objects[key]['width']
        y_ref=self.objects[key]['lenght']
        n=int(math.ceil(SW/self.objects[key]['width']))
        m=int(math.ceil(SH/self.objects[key]['lenght']))
        #print(n,m)
        x=math.floor(self.objects[key]['position'][0]/x_ref)
        y=math.floor(self.objects[key]['position'][1]/y_ref)
        x_offset=int((x*x_ref+x_ref/2)-self.objects[key]['position'][0])
        y_offset=int((y*y_ref+y_ref/2)-self.objects[key]['position'][1])

        vertical=[]
        horizontal=[]
        for i in range(0,n):
            vertical.append([i*x_ref-x_offset,0,i*x_ref-x_offset,SH])
        for j in range(0,m):
            horizontal.append([0,j*y_ref-y_offset,SW,j*y_ref-y_offset])
        #state=np.zeros((n+2,m+2))
        #state[x,y]=0.1
        #when offset x is positive add a column at the begining
        state=np.zeros((n, m))
        #x=math.floor((self.objects[key]['position'][0])/x_ref)
        #y=math.floor((self.objects[key]['position'][1])/y_ref)
        state[x][y]=0.1
        if x_offset>=0:
            vertical.insert(0,[vertical[0][0]-x_ref,0,vertical[0][0]-x_ref,SH])
        else:
            vertical.append([vertical[-1][0]+x_ref,0,vertical[-1][0]+x_ref,SH])
        if y_offset>=0:
            vertical.append([0,horizontal[-1][1]+y_ref,SW,horizontal[-1][1]+y_ref])
        else:
            vertical.insert(0,[0,horizontal[0][1]-y_ref,SW,horizontal[0][1]-y_ref])
        if (horizontal[0][1]-y_ref)<0:
            aux_a_y1=horizontal[0][1]
        else:
            aux_a_y1=y_ref
        if (horizontal[-1][1]-SH)<y_ref:
            aux_a_y2=horizontal[-1][1]-SH
        else:
            aux_a_y2=y_ref
        if (horizontal[0][1]-y_ref)<0:
            aux_a_x1=vertical[0][1]
        else:
            aux_a_x1=x_ref
        if (horizontal[-1][1]-SW)<y_ref:
            aux_a_x2=vertical[-1][1]-SW
        else:
            aux_a_x2=x_ref
        
        #add rest of the objects to the state
        for ele in temp_objects.keys():
            #print(ele)
            #i_ref=temp_objects[ele]['width']
            #j_ref=temp_objects[ele]['lenght']
            #i=math.floor((temp_objects[ele]['position'][0]+x_offset)/x_ref)
            #j=math.floor((temp_objects[ele]['position'][1]+y_offset)/y_ref)
            corners=self.get_corners(ele,x_offset, y_offset)
            for corner in corners:
                i=math.floor((corner[0])/x_ref)
                j=math.floor((corner[1])/y_ref)
                state[i][j]=0.2
        i=math.floor((goal[0]+x_offset)/x_ref)
        j=math.floor((goal[1]+y_offset)/y_ref)
        state[i][j]=0.3
        u,v=state.shape
        #print(u,v)
        #u is x
        #(Sn+1)mod u is y
        #(Sn+1)mod u*v is z
        for s in range(0,u*v):
            a=(int(s/v))
            b=(((s)%v))
            #print(a,b)
            c=(int((s)/(u*v)))
            #print(a,b,c)
            states_,IOTA,q_values=self.iota(state,a,b,c,s,i,j)
            iEnv['s'+str(s)]=[states_,IOTA]
            iQtable.append(q_values)
            
        #print(iQtable)
        #print(len(iQtable),u*v)
        #return hh
        #env={'s1':[['sn','reward','done'],[],[],[]]}
        #iQTable=[[],[],[],..,[]]
        return vertical,horizontal,state,x,y, iQtable,actions,x_offset, y_offset, iEnv
    def iota(self,state,a,b,c,current_s,i,j):
        IOTA=[0,0,0,0,0,0,0,0,0]
        q_values=[0,0,0,0,0,0,0,0,0]
        states_=[None,None,None,None,None,None,None,None,None]
        #current_s=(state.shape[1]*b)+a
        next_s=[current_s+state.shape[1],current_s-1,current_s-state.shape[1],current_s+1]
        if state[a][b]==0.2:
            return states_, IOTA, q_values
        if state[a][b]==0.3:
            states_[-1]=['s'+str(current_s),10,True]
            IOTA[-1]=1
            q_values[-1]=1
            return states_, IOTA, q_values
        ##RIGHT
        #if (b+1)<state.shape[1]:
        if (a+1)<state.shape[0]:
            #if state[a][b+1]==0.0 or state[a][b+1]==0.1:
            if state[a+1][b]==0.0 or state[a+1][b]==0.1:
                IOTA[0]=1
                q_values[0]=1
                states_[0]=['s'+str(next_s[0]),-0,False]
            #elif state[a][b+1]==0.3:
            elif state[a+1][b]==0.3:
                IOTA=[0,0,0,0,0,0,0,0,0]
                q_values=[0,0,0,0,0,0,0,0,0]
                IOTA[0]=1
                states_[0]=['s'+str(next_s[0]),10,False]
                q_values[0]=1.1
                #print("one",'s'+str(next_s[0]))
                return states_, IOTA, q_values
        ##LEFT
        #if (b-1)>=0:
        if (a-1)>=0:
            #if state[a][b-1]==0.0 or state[a][b-1]==0.1:
            if state[a-1][b]==0.0 or state[a-1][b]==0.1:
                IOTA[2]=1
                q_values[2]=1
                states_[2]=['s'+str(next_s[2]),-0,False]
            #elif state[a][b-1]==0.3:
            elif state[a-1][b]==0.3:
                IOTA=[0,0,0,0,0,0,0,0,0]
                q_values=[0,0,0,0,0,0,0,0,0]
                IOTA[2]=1
                states_[2]=['s'+str(next_s[2]),10,False]
                q_values[2]=1.1
                #print("two",'s'+str(next_s[2]))
                return states_, IOTA, q_values
        ###UP
        #if (a-1)>=0:
        if (b-1)>=0:
            #if state[a-1][b]==0.0 or state[a-1][b]==0.1:
            if state[a][b-1]==0.0 or state[a][b-1]==0.1:
                IOTA[1]=1
                q_values[1]=1
                states_[1]=['s'+str(next_s[1]),-0,False]
            #elif state[a-1][b]==0.3:
            elif state[a][b-1]==0.3:
                IOTA=[0,0,0,0,0,0,0,0,0]
                q_values=[0,0,0,0,0,0,0,0,0]
                IOTA[1]=1
                states_[1]=['s'+str(next_s[1]),10,False]
                q_values[1]=1.1
                #print("hree",'s'+str(next_s[1]))
                return states_, IOTA, q_values
        ##DOWN
        #if (a+1)<state.shape[0]:
        if (b+1)<state.shape[1]:
            #if state[a+1][b]==0.0 or state[a+1][b]==0.1:
            if state[a][b+1]==0.0 or state[a][b+1]==0.1:
                IOTA[3]=1
                q_values[3]=1
                states_[3]=['s'+str(next_s[3]),-0,False]
            #elif state[a+1][b]==0.3:
            elif state[a][b+1]==0.3:
                IOTA=[0,0,0,0,0,0,0,0,0]
                q_values=[0,0,0,0,0,0,0,0,0]
                IOTA[3]=1
                states_[3]=['s'+str(next_s[3]),10,False]
                q_values[3]=1.1
                #print("four",'s'+str(next_s[3]))
                return states_, IOTA, q_values
        #
        if a<i and q_values[0]>0 and q_values[0]>0:
             q_values[0]=q_values[0]+0.1
        if a>i and q_values[2]>0 and q_values[2]>0:
             q_values[2]=q_values[2]+0.1
        if b<j and q_values[3]>0 and q_values[3]>0:
             q_values[3]=q_values[3]+0.1
        if b>j and q_values[1]>0 and q_values[1]>0:
             q_values[1]=q_values[1]+0.1
        return states_,IOTA,q_values
    def get_corners(self,obj,x_offset, y_offset):
        w=self.objects[obj]['position'][0]+x_offset
        h=self.objects[obj]['position'][1]+y_offset
        hi=self.objects[obj]['lenght']
        wi=self.objects[obj]['width']
        corners=[[w,h],[w+wi/2,h],[w-wi/2,h],
                 [w,h+hi/2],[w+wi/2,h+hi/2],[w-wi/2,h+hi/2],
                 [w,h-hi/2],[w+wi/2,h-hi/2],[w-wi/2,h-hi/2]
                ]
        return corners
    def digMatrix(self,x,y,z):
        M=np.array([[1,0,0,x],
                   [0,1,0,y],
                   [0,0,1,z],
                   [0,0,0,1]])
        return M
    def xRot(self,theta,x,y,z):
        X_i=np.array[[ 1,            0,             0,x],
                 [ 0,np.cos([theta]),-np.sin([theta]),y],
                 [ 0,np.sin([theta]), np.cos([theta]),z],
                 [ 0,            0,             0,1]]
        return X_i
    def yRot(self,betha,x,y,z):
        Y_i=np.array([np.cos([betha]) ,0,np.sin([betha]),x],
                 [0             ,1,            0,y],
                 [-np.sin([betha]),0,np.cos([betha]),z],
                 [0,             0,            0,1])
        return Y_i
    def zRot(self,alpha,x,y,z):
        Z_i=np.array([[ np.cos([alpha])[0],np.sin([alpha])[0],0,x],
                    [-np.sin([alpha])[0],np.cos([alpha])[0],0,y],
                    [ 0,0,1,z],
                    [ 0,0,0,1]])
        return Z_i
    def tN(self,theta,alpha,r,d):
        return np.matmul(self.zRot(alpha,0,0,d),self.xRot(theta,r,0,0))
    def fowardKinematics(self,thetas,ROBOT):
        return T
    def inverseKinematics(self,x,y,z,roll,pitch,yaw,ROBOT):
        return thetas
    def getCharacteristics(self,obj):
        x=obj['position'][0]
        y=obj['position'][1]
        xmin=x-obj['width']/2
        xmax=x+obj['width']/2
        ymin=y-obj['lenght']/2
        ymax=y+obj['lenght']/2
        color=tuple(obj['color'])
        return int(xmin),int(xmax),int(ymin),int(ymax),color
    def getAxisX(self,obj):
        imin=obj['position'][0]
        jmin=obj['position'][1]
        alpha=obj['orientation'][2]
        rot=np.matmul(self.zRot(alpha,imin,jmin,0),self.digMatrix(AXIS_LENGHT,0,0))
        imax=rot[0][3]
        jmax=rot[1][3]
        return int(imin),int(imax),int(jmin),int(jmax)
    def getAxisY(self,obj):
        imin=obj['position'][0]
        jmin=obj['position'][1]
        alpha=obj['orientation'][2]
        rot=np.matmul(self.zRot(alpha,imin,jmin,0),self.digMatrix(0,-AXIS_LENGHT,0))
        imax=rot[0][3]
        jmax=rot[1][3]
        return int(imin),int(imax),int(jmin),int(jmax)
    def getAxisZ(self,obj):
        return imin,imax,jmin,jmax
    def contextualKeyFrames(self,h,w):
        return
    def render(self,obj,vertical,horizontal,state,x_offset, y_offset,iQtable,m_field):
        for key in self.objects.keys():
            xmin,xmax,ymin,ymax,color=self.getCharacteristics(self.objects[key])
            cv2.rectangle(self.work_space, (xmin, ymin), (xmax, ymax), color, -1)
            imin,imax,jmin,jmax=self.getAxisX(self.objects[key])
            cv2.line(self.work_space, (imin,jmin), (imax,jmax), RED, AXIS_WIDTH)
            imin,imax,jmin,jmax=self.getAxisY(self.objects[key])
            cv2.line(self.work_space, (imin,jmin), (imax,jmax), GREEN, AXIS_WIDTH)
            #draw goal position
            if len(self.objects[key]['goal'])>0:
                imin=self.objects[key]['goal'][0]
                imax=imin+AXIS_LENGHT
                jmin=self.objects[key]['goal'][1]
                jmax=jmin-AXIS_LENGHT
                cv2.line(self.work_space, (imin,jmin), (imax,jmin), RED, AXIS_WIDTH)
                cv2.line(self.work_space, (imin,jmin), (imin,jmax), GREEN, AXIS_WIDTH)
        
        for line in vertical:
            try:
                cv2.line(self.work_space, (int(line[0]),int(line[1])), (int(line[2]),int(line[3])), (255,255,255), 1)
            except:
                print(line)
        for line in horizontal:
            try:
                cv2.line(self.work_space, (int(line[0]),int(line[1])), (int(line[2]),int(line[3])), (255,255,255), 1)
            except:
                print(line)
        if len(self.objects[obj]['path'])>0:
            for line in self.objects[obj]['path']:
                cv2.line(self.work_space, (int(line[4]),int(line[5])), (int(line[2]),int(line[3])), YELLOW, 2)
        if len(self.objects[obj]['path_smooth'])>0:
            for point in self.objects[obj]['path_smooth']:
                cv2.circle(self.work_space, (int(point[0]),int(point[1])), 4, BLUE_GREEN, -1)
        for x in range(0,state.shape[0]):
            for y in range(0,state.shape[1]):
                if state[x][y]==0.1:
                    color=(0,0,255)
                elif state[x][y]==0.2:
                    color=(255,0,0)
                elif state[x][y]==0.3:
                    color=(0,255,0)
                else:
                    color=(255,255,255)
                cv2.putText(self.work_space, 
                            str(state[x][y].astype(float)), (-10+x_offset+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2),int(y_offset-5+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 1, cv2.LINE_AA, False)
                circles_max=iQtable[int(state.shape[1]*x)+y][:4]
                circles_max.append(iQtable[int(state.shape[1]*x)+y][-1])
                if m_field:
                    index=0
                    for cir in circles_max:
                        coloration=GREEN
                        size=1
                        if cir==0:
                            coloration=RED
                            size=1
                        elif cir==1:
                            size=1
                        elif cir>1.05:
                            size=3
                        ##
                        if index==0:
                            cv2.circle(self.work_space, (x_offset-3+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2)+5,int(y_offset-5+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)), size, coloration, -1)
                        if index==2:
                            #print(iQtable[int(state.shape[0]*y)+x],str(int(state.shape[0]*y)+x))
                            #a=1/0
                            cv2.circle(self.work_space, (x_offset-3+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2)-5,int(y_offset-5+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)), size, coloration, -1)
                        if index==1:
                            cv2.circle(self.work_space, (x_offset-3+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2),int(y_offset-5+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)-5), size, coloration, -1)
                        if index==3:
                            cv2.circle(self.work_space, (x_offset-3+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2),int(y_offset-5+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)+5), size, coloration, -1)
                        if index==4 and cir==1:
                            cv2.circle(self.work_space, (x_offset+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2),int(y_offset+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)), 4, BLUE_GREEN, -1)
                        index+=1
        cv2.imshow('Work space',self.work_space)
        
    def step(self,obj,action):
        if action==0:
            xmin=self.objects[obj]['position'][0]+int(self.objects[obj]['width']/2)
            xmid=self.objects[obj]['position'][0]
            self.objects[obj]['position'][0]=self.objects[obj]['position'][0]+self.objects[obj]['width']
            xmax=self.objects[obj]['position'][0]
            ymax=self.objects[obj]['position'][1]
            ymin=ymax
            ymid=ymax
            self.objects[obj]['path'].append([xmin,ymin,xmax,ymax,xmid,ymid])
        elif action==1:
            ymin=self.objects[obj]['position'][1]-int(self.objects[obj]['lenght']/2)
            ymid=self.objects[obj]['position'][1]
            self.objects[obj]['position'][1]=self.objects[obj]['position'][1]-self.objects[obj]['lenght']
            ymax=self.objects[obj]['position'][1]
            xmax=self.objects[obj]['position'][0]
            xmin=xmax
            xmid=xmin
            self.objects[obj]['path'].append([xmin,ymin,xmax,ymax,xmid,ymid])
        elif action==2:
            xmin=self.objects[obj]['position'][0]-int(self.objects[obj]['width']/2)
            xmid=self.objects[obj]['position'][0]
            self.objects[obj]['position'][0]=self.objects[obj]['position'][0]-self.objects[obj]['width']
            xmax=self.objects[obj]['position'][0]
            ymax=self.objects[obj]['position'][1]
            ymin=ymax
            ymid=ymin
            self.objects[obj]['path'].append([xmin,ymin,xmax,ymax,xmid,ymid])
        elif action==3:
            ymin=self.objects[obj]['position'][1]+int(self.objects[obj]['lenght']/2)
            ymid=self.objects[obj]['position'][1]
            self.objects[obj]['position'][1]=self.objects[obj]['position'][1]+self.objects[obj]['lenght']
            ymax=self.objects[obj]['position'][1]
            xmax=self.objects[obj]['position'][0]
            xmin=xmax
            xmid=xmin
            self.objects[obj]['path'].append([xmin,ymin,xmax,ymax,xmid,ymid])
        #elif goal actions
        #elif limits actions
        elif action==6:
            print("Especial")
        elif action==8:
            print("Droped")
        else:
            print("Not an action")
        #add reward here to every one
        #return next state, reward, done
    def choose_action(self,probs,iota,epsilon):
        probabilities=[]
        choices=[]
        for i in range(0,len(iota)):
            if iota[i]>0:
                probabilities.append(probs[0][i])
                choices.append(i)
        if len(probabilities)>0:
            probss=[(p/sum(probabilities)) for p in probabilities] 
            if epsilon > random.uniform(0, 1):
                action=np.random.choice(choices, p=probss)
            else:
                action=choices[np.argmax(np.array(probabilities))]
        else:
            action=0
        return action
    def epsilon(self,time,EPISODES,A,B,C):
            standardized_time=(time-A*EPISODES)/(B*EPISODES)
            cosh=np.cosh(math.exp(-standardized_time))
            epsilon=1.1-(1/cosh+(time*C/EPISODES))
            return epsilon
    def agent(self,iQtable,iEnv,x,y,s):
        actions=[]
        state=int(s.shape[1]*x)+y
        gamma=0.05
        df=0.99
        A=0.5
        B=0.1
        C=0.1
        
        for steps in range(1,10001):
            #get action
            #execute action
            #compute q values
            try:
                iota=np.array(iEnv['s'+str(state)][1])
                target = np.array([iQtable[state]])
                action=self.choose_action(target,iota,self.epsilon(steps,100000,A,B,C))
                state_,reward,done=iEnv['s'+str(state)][0][action]
                state_=int(state_.split('s')[1])
                if done:
                    #print("Training done")
                    iQtable[state][action]=reward
                    #print(iQtable[state])
                    state=int(s.shape[1]*x)+y
                else:
                    #iota_=np.array(iEnv['s'+str(state_)][1])
                    target_ = np.array([iQtable[state_]])
                    #target_iota_=target_+abs(np.amin(target_))
                    #qmax_=np.amax((target_iota_*iota_)-abs(np.amin(target_)))
                    iQtable[state][action]= iQtable[state][action]+gamma*(reward+df*np.amax(target_)-iQtable[state][action])
                    #print(iQtable[state])
                    state=state_
            except:
                #print(state)
                #print(iEnv['s'+str(state)][0])
                return None
                #a=1/0
            

            if steps%150==0:
                state=int(s.shape[1]*x)+y
                
        state=int(s.shape[1]*x)+y
        
        done=False
        #print(iQtable)
        print("Training done")
        
        for i in range(0,100):
            iota=np.array(iEnv['s'+str(state)][1])
            target = np.array([iQtable[state]])
            action=self.choose_action(target,iota,0)
            #print(iQtable[state])
            state_,reward,done=iEnv['s'+str(state)][0][action]
            #print(reward)
            if done:
                print("Finished")
                break
            #print(state,iQtable[state],action,done)
            state_=int(state_.split('s')[1])
            state=state_
            actions.append(action)  
        #print(actions)
        return actions





def running(obj,goal,m_field, robot):
    vertical,horizontal,state,x,y, iQtable,actions, x_offset, y_offset,iEnv =robot.set_env(obj,goal)
    #solve iQTable return set of actions 
    table=np.array(iQtable)
    #print(table[0],iQtable[0])
    actionsss=robot.agent(table,iEnv,x,y,state)
    #actionsss=[0]
    cv2.startWindowThread()
    #print(iQtable)
    #robot.render(obj,vertical,horizontal,state, x_offset, y_offset,iQtable,m_field)
    if actionsss is not None:
        for a in actionsss:
            #get real state
            #get fake state
            #simulate if the goal can be still reached
            #if yes continue
            #otherwise compute iQtable again
            robot.render(obj,vertical,horizontal,state, x_offset, y_offset,iQtable,m_field)
            robot.step(obj,a)
            key=cv2.waitKey(int(round(200/fps)))  # We need to call cv2.waitKey after cv2.imshow
            if key == 0:  # Press Esc for exit
                break
            robot.reset()
        #interpolation
        try:
            x =  [item[0] for item in robot.objects[obj]['path']]
            x.append(robot.objects[obj]['path'][-1][2])
            x.insert(0,robot.objects[obj]['path'][0][4])
            y =  [item[1] for item in robot.objects[obj]['path']]
            y.append(robot.objects[obj]['path'][-1][3])
            y.insert(0,robot.objects[obj]['path'][0][5])
            arr = np.array(list(zip(x,y)))
            x, y = zip(*arr)
            f, u = interpolate.splprep([x, y], s=2)
            xint, yint = interpolate.splev(np.linspace(0, 1, 20), f)
            robot.objects[obj]['path_smooth']=list(zip(xint,yint))
            robot.render(obj,vertical,horizontal,state, x_offset, y_offset,iQtable,True)
            time.sleep(1)
            #key=cv2.waitKey(0)
            return list(zip(xint,yint))
        except Exception as e:
            print("Interpolation error")
            print(e)
            return None
    else:
        print("Not possible to handle piece")
        return None

#############################
def get_pose(frame_1,frame_2):
        listener = tf.TransformListener()
        trans = []
        rot = []
        rate = rospy.Rate(10.0)
        while not trans:
            try:
                (trans,rot) = listener.lookupTransform(frame_1, frame_2, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
            rate.sleep()
        #print ('Translation: ' , trans)
        #print ('Rotation: ' , rot)
        return trans,rot
def rad_to_deg(angle):
    pi=3.1416
    angle=(angle*360)/(2*pi)
    return round(angle,2)
def inverse_kinematics(x,y,z):
    z=(z-1)
    pi=3.1416
    position=[0,0,0,0,0,0,0]
    l_0=0.25
    l_1=0.45
    l_2=0.30
    l_3=0.26
    h_0=1.0
    a=l_1
    b=l_2
    #c=np.sqrt(((np.sqrt((x**2)+(y**2))-l_0)**2)+z**2)
    c=np.sqrt(((x)**2)+((y)**2))-l_0
    c=np.sqrt((c**2)+(z**2))
    try:
        theta_1=np.arctan(x/abs(y))
        gamma=np.arctan((z)/(np.sqrt((x**2)+(y**2))-l_0))
        alpha=np.arccos(((a**2)+(c**2)-(b**2))/(2*a*c))
        theta_2=-(alpha+gamma)
        betha=np.arccos(((a**2)+(b**2)-(c**2))/(2*a*b))
        theta_3=pi-betha
        c=np.cos(alpha)*l_1
        theta_4=-theta_3+pi/2-theta_2
    except Exception as e:
        print(e)
    return round(theta_1.astype(float),2),round(theta_2.astype(float),2),round(theta_3.astype(float),2),round(theta_4.astype(float),2)
def inverse_kinematics_1(x,y,z):
    pi=3.14159
    l0=1.0
    l1=0.25
    l2=0.45
    l3=0.3
    try:
        a1=np.sqrt(((x-l1)**2)+(y**2)+((z-l0)**2))
        a2=np.sqrt(((x-l1)**2)+(y**2))
        a3=z-l0
        b1=y
        b2=x-l1
        betha1=np.arctan(b2/b1)
        betha2=np.arccos(((l2**2)+(a1**2)-(l3**2))/(2*l2*a1))
        alpha1=np.arccos(((l2**2)+(l3**2)-(a1**2))/(2*l2*l3))
        theta_1=np.arctan(a3/a2)
        theta_2=-(pi/2)+betha1+betha2
        theta_3=0.0
        theta_4=-pi+alpha1
        theta_5=-theta_1*np.sin(theta_2)+(pi/2)
        theta_6=theta_1+(pi/2)
        theta_7=-theta_2-theta_4-1.57
        #print(theta_1,theta_2,theta_3,theta_4,theta_5,theta_6,theta_7)
    except Exception as e:
        print(e)
    return round(theta_1.astype(float),2),round(theta_2.astype(float),2),0.0,round(theta_4.astype(float),2),round(theta_5.astype(float),2),round(theta_6.astype(float),2),round(theta_7.astype(float),2)
def action(x,y,z,quat):
    client = actionlib.SimpleActionClient('move_base',MoveBaseAction)
    client.wait_for_server()
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.x = quat[0]
    goal.target_pose.pose.orientation.y = quat[1]
    goal.target_pose.pose.orientation.z = quat[2]
    goal.target_pose.pose.orientation.w = quat[3]
    client.send_goal(goal)
    wait = client.wait_for_result()
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        rospy.loginfo("Goal reached!")
        return client.get_result()

def init_all():
    sss = simple_script_server()
    sss.init("arm_right")
    sss.recover("arm_right")  
    #andle_arm = sss.move("arm_left","side")
    handle_arm = sss.move("arm_right",[[0.7854,-0.7854,0.0,-0.7854,1.5708,1.5708,0.0]])
    handle_arm.wait()
def execute(path,path_torso):
    sss = simple_script_server()
    
    handle_arm = sss.move("arm_right",[path[0]],False)
    handle_arm.wait()
    r = requests.get('http://10.4.13.52:5000/open', verify=False)
    time.sleep(1)
    handle_arm = sss.move("arm_right",[path[1]],False)
    handle_arm.wait()
    r = requests.get('http://10.4.13.52:5000/close', verify=False)
    time.sleep(1)
    handle_arm = sss.move("arm_right",path[2:],False)
    handle_arm.wait()
    r = requests.get('http://10.4.13.52:5000/open', verify=False)
    time.sleep(1)
def get_robot_path(object_path):
    path=[]
    path_torso=[]
    x=-0.535+object_path[0][0]*0.0025
    y= 1.0-object_path[0][1]*0.0025
    z=0.82+0.26
    t1,t2,t3,t4,t5,t6,t7=inverse_kinematics_1(x,y,z)
    path.append([float(t1),float(t2),float(t3),float(t4),float(t5),float(t6),float(t7)])
    x=-0.535+object_path[0][0]*0.0025
    y= 1.005-object_path[0][1]*0.0025
    print(x,y)
    z=0.745+0.26
    t1,t2,t3,t4,t5,t6,t7=inverse_kinematics_1(x,y,z)
    path.append([float(t1),float(t2),float(t3),float(t4),float(t5),float(t6),float(t7)])
    path_torso.append([0.0,float(t1)])
    for point in object_path:
        x=-0.535+point[0]*0.0025
        y=1.00-point[1]*0.0025
        z=0.825+0.26
        #print(x,y)
        t1,t2,t3,t4,t5,t6,t7=inverse_kinematics_1(x,y,z)
        #print(rad_to_deg(t1))
        path.append([float(t1),float(t2),float(t3),float(t4),float(t5),float(t6),float(t7)])
        path_torso.append([0.0,float(t1)])
    return path,path_torso

def classify_objects():
    goals_loc={}
    with open('/home/robot/Desktop/IDQN/objects_1.yaml') as file:
        objects = yaml.load(file, Loader=yaml.FullLoader)[0]
    try:
        with open('/home/robot/Desktop/IDQN/goal_1.yaml') as file:
            goal_right = yaml.load(file, Loader=yaml.FullLoader)[0]
            goals_loc['right']=goal_right['right']
    except:
        print("No right goal available")
    try:
        with open('/home/robot/Desktop/IDQN/goal_2.yaml') as file:
            goal_left = yaml.load(file, Loader=yaml.FullLoader)[0]
            goals_loc['left']=goal_left['left']
    except:
        print("No left goal available")
    if len(goals_loc.keys())==0:
        print("No goals available aborting")
        exit()
    return objects, goals_loc
def get_sub_goals(instructions,temp_objects):
    goals={}
    for ins in instructions:
        keys=[]
        for type_object in ins[0]:
            if type_object[0][2]=="*":
                for key in temp_objects.keys():
                    if type_object[0][:-1]== key[:2]:
                        keys.append(key)
            else:
                for key in temp_objects.keys():
                    if type_object[0]== key[:3]:
                        keys.append(key)
                pass
        if ins[1] is not None:
            goals[ins[1]]=keys
    return goals
def clean_objects(goals_location,temp_objects,goals):
    pop_keys=[]
    for key_o in temp_objects.keys():
        x_obj=temp_objects[key_o]['position'][0]
        y_obj=temp_objects[key_o]['position'][0]
        for key_g in goals_location.keys():
            x_g=goals_location[key_g][0]
            y_g=goals_location[key_g][0]
            if x_obj>x_g-30 and x_obj<x_g+30 and y_obj>y_g-30 and y_obj<y_g+30:
                pop_keys.append(key_o)
                break
        #print(goals_location)
    #exit()
    #print(pop_keys)
    new_objects=temp_objects.copy()
    for sub_goal in goals.keys():
        for thing in goals[sub_goal]:
            new_objects.pop(thing)
    for sub_goal in new_objects.keys():
        temp_objects.pop(sub_goal)
    for key in pop_keys:
        temp_objects.pop(key)
    return temp_objects
def get_here_location():
    box_name="right"
    return box_name

if __name__ == '__main__':
    rospy.init_node('irohms_iql_', anonymous=True)
    #exit()
    init_all()
#####################################################################
###########################Control section###############################
    actions=[0,0,0,0]
    observe_environment()
    temp_objects,goals_loc=classify_objects()
    #print(temp_objects)
    
    #print(temp_objects)
    #exit()
    ####Robot says I am ready:::
    ##############################TODO
    #Get spoken instroctions
    ##Robot confirms instruction
    ######################################
    words="put the red objects in the left box, then move the blue objects and green objects in the right box"
    instructions=recognise_instructions(words)
    #print(instructions)
    goals=get_sub_goals(instructions,temp_objects) 
    #print(goals)
    temp_objects=clean_objects(goals_loc,temp_objects,goals)  
    #print(temp_objects)
    #exit() 
    #print(goals)  
    ######################################TODO    
    # ##if there is a here:
    #####Get hand coordinates 
    #####################################
    ###Actively execute the instruction###
    if len(temp_objects.keys())==0:
        print("There are not objects to grasp on the table!")
    else:
        for i in range(0,3):
            for sub_goal in goals.keys():
                try:
                    goal=goals_loc[sub_goal]
                except:
                    print("No ",sub_goal, "box available!!")
                for objeto in goals[sub_goal]:
                    try:
                        #print(goals[sub_goal])
                        robot=IDQL(temp_objects)
                        object_path=running(objeto,goal,False,robot)
                        if  object_path is not None:
                            #print("executing path")
                            path,path_torso=get_robot_path(object_path)
                            execute(path,path_torso)
                            temp_objects.pop(objeto)
                    except Exception as e:
                        print("No ",objeto, " box available. ",e)
            
            init_all()
            objects={}
            observe_environment()
            temp_objects,goals_loc=classify_objects()
            goals=get_sub_goals(instructions,temp_objects) 
            temp_objects=clean_objects(goals_loc,temp_objects,goals)
            if len(temp_objects.keys())==0:
                break
    cv2.destroyAllWindows()
