#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import time
import yaml

RED=[0,0,255]
GREEN=[0,255,0]
BLUE=[255,0,0]
BLUE_GREEN=[255, 255, 0]
YELLOW=[0,255,255]
objects={}


def get_color(name):
    color=[255,255,255]
    if name == 'b':
        color=BLUE
    elif name =='y':
        color=YELLOW
    elif name=='g':
        color=GREEN
    elif name=='hand':
        color=RED
    return color
def tagger(img, image,name):
    original = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    ROI_number = 0
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    i=0
    types=""
    
    objective={}
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
                types="circle"
            if name=='goal_'+str(int(cX/10))+"_"+str(int(cY/10)):
                objective['goal']=[cX-5,cY+30,0]
            ## save goal position
            #if name=='hand':
            ## save hand position
            ##    object:
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
            print(objects)
            with open('/home/robot/Desktop/IDQN/objects_1.yaml', 'w') as file:
                documents = yaml.dump([objects], file)
        if len(objective)>0:
            print(objective)
            with open('/home/robot/Desktop/IDQN/goal_1.yaml', 'w') as file:
                documents = yaml.dump([objective], file)
    except:
        print("error")
        return image
    return image

def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

x1=100
y1=100
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
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
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue
        
        # crop images
        x, y, w, h = 0, 150, 350, 400
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
            if time>30:
                #PINK
                #(hMin = 146 , sMin = 65, vMin = 148), (hMax = 179 , sMax = 255, vMax = 255)
                #(hMin = 166 , sMin = 64, vMin = 170), (hMax = 174 , sMax = 255, vMax = 255)
                lower=np.array([132,70,0])
                upper=np.array([179,143,255])
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                result_pink = cv2.bitwise_and(result, result, mask = mask)
                #(hMin = 0 , sMin = 72, vMin = 0), (hMax = 179 , sMax = 255, vMax = 255)
                #lower=np.array([0,72,0])
                #upper=np.array([179,255,255])
                #hsv = cv2.cvtColor(result_pink, cv2.COLOR_BGR2HSV)
                #mask = cv2.inRange(hsv, lower, upper)
                #result_pink = cv2.bitwise_and(result_pink, result_pink, mask = mask)
                color_image=tagger(result_pink, color_image, "goal")
                #BLUE
                #(hMin = 65 , sMin = 37, vMin = 0), (hMax = 113 , sMax = 255, vMax = 255)
                lower=np.array([66,104,0])
                upper=np.array([123,255,255])
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                result_blue = cv2.bitwise_and(result, result, mask = mask)
                color_image=tagger(result_blue, color_image, "b")
                #GREEN
                #(hMin = 33 , sMin = 28, vMin = 69), (hMax = 74 , sMax = 230, vMax = 255)
                lower=np.array([29,139,33])
                upper=np.array([53,255,255])
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                result_green = cv2.bitwise_and(result, result, mask = mask)
                color_image=tagger(result_green, color_image, "g")
                #YELLOW
                #(hMin = 19 , sMin = 34, vMin = 0), (hMax = 35 , sMax = 230, vMax = 255)
                lower=np.array([19,34,0])
                upper=np.array([35,255,255])
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                result_yellow = cv2.bitwise_and(result, result, mask = mask)
                #(hMin = 20 , sMin = 37, vMin = 169), (hMax = 51 , sMax = 222, vMax = 255)
                lower=np.array([12,81,0])
                upper=np.array([32,255,255])
                hsv = cv2.cvtColor(result_yellow, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                result_yellow = cv2.bitwise_and(result_yellow, result_yellow, mask = mask)
                color_image=tagger(result_yellow, color_image, "y")
                #RED
                #(hMin = 0 , sMin = 119, vMin = 0), (hMax = 7 , sMax = 255, vMax = 255)
                lower=np.array([0,119,0])
                upper=np.array([7,255,255])
                hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                result_red = cv2.bitwise_and(result, result, mask = mask)
                color_image=tagger(result_red, color_image, "hand")
            
            try:
                images = np.hstack((color_image,result,result_pink,result_red,result_blue,result_green,result_yellow))
            except:
                images= np.hstack((color_image,result))
        else:
            images = np.hstack((color_image,result,depth_colormap))

        # Show images
        cv2.namedWindow('IDQL environment', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('IDQL environment', images)
        time+=1
        key=cv2.waitKey(1)
        if cv2.getWindowProperty('IDQL environment', cv2.WND_PROP_VISIBLE) <1:
            break
    cv2.destroyAllWindows()
finally:
    # Stop streaming
    pipeline.stop()
