#!/usr/bin/env python3

import numpy as np
import rospy
import time
import cv2
import math
import pyrealsense2 as rs
from cvzone.HandTrackingModule import HandDetector

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray,MultiArrayDimension 
from std_msgs.msg import String,Int32MultiArray
from irohms_iql.msg import Elements, Pos, Color

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

RS_CAM_X=-0.85
RS_CAM_Y=-0.22
RS_CAM_Z=1.21

class perRealsense():
    def __init__(self):
        self.objects={}
        self.objects_temp={}
        self.hands={}
        self.goals={}
        self.running=True
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #self.config.enable_device('141722074327')
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        self.device = pipeline_profile.get_device()
        self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
        self.camera_chaeckup(False)
        self.detector = HandDetector(detectionCon=0.8, maxHands=2)
        self.filters={'pink1':[[[132,70,0],[179,143,255]],None],'pink2':[[[132,70,0],[179,143,255]],"goal"],
                      'blue':[[[66,104,0],[123,255,255]],"b"],'green':[[[29,139,33],[53,255,255]],"g"],
                      'yellow':[[[19,34,0],[53,255,255]],'y'],'red':[[[0,119,0],[7,255,255]],'r']}
        rospy.init_node('irohms_perception_table', anonymous=True)
        self.pub = rospy.Publisher('/irohms_idql/irohms_perception_table_elements', Elements, queue_size=10)
        self.pub_goal = rospy.Publisher('/irohms_idql/irohms_perception_table_goals', Elements, queue_size=10)
        self.pub_hands = rospy.Publisher('/irohms_idql/irohms_perception_table_hands', Elements, queue_size=10)
        self.rate = rospy.Rate(100)
        self.align_to=rs.stream.color
        self.align=rs.align(self.align_to)
        sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        sensor.set_option(rs.option.exposure, 240.000)

    def twod_to_threed(self,x,y,z):
        #return [0,0,0]
        #frames = self.pipeline.wait_for_frames()
        #aligned_frames = self.align.process(frames)
        #depth_frame = aligned_frames.get_depth_frame()
        #self.intr = depth_frame.profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
        return rs.rs2_deproject_pixel_to_point(self.intr,[float(x),float(y)],z)    
    def camera_chaeckup(self,found_rgb):
        for s in self.device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("This requires Depth camera with Color sensor")
            exit(0)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if self.device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)
    def hand_detector(self,img,u,v,depth_frame):
        color_image=img.copy()
        hands, img = self.detector.findHands(img)  # With Draw
        hand1=None
        hand2=None
        if hands:
            hand1 = hands[0]
            if len(hands) == 2:
                hand2 = hands[1]
        if hand1 is not None:
                #lmList2 = hand2["lmList"]  # List of 21 Landmarks points
                bbox2 = hand1["bbox"]  # Bounding Box info x,y,w,h
                centerPoint2 = hand1["center"]  # center of the hand cx,cy
                handType2 = hand1["type"] 
                cv2.rectangle(color_image, (bbox2[0], bbox2[1]), (bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]), (0,255,0), 1)
                cv2.circle(color_image, (centerPoint2[0], centerPoint2[1]), 2, (255, 255, 255), -1)
                cv2.putText(color_image, str(handType2), (centerPoint2[0], centerPoint2[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                position_object=self.twod_to_threed(centerPoint2[0]+u,centerPoint2[1]+v,depth_frame.get_distance(centerPoint2[0]+v, centerPoint2[1]+u))
                self.hands[handType2]={}
                self.hands[handType2]['position']=[centerPoint2[0],centerPoint2[1]]
                self.hands[handType2]['lenght']=bbox2[2]
                self.hands[handType2]['width']=bbox2[3]
                self.hands[handType2]['pose']=[RS_CAM_X+position_object[1],RS_CAM_Y-position_object[0],RS_CAM_Z-position_object[2]]
                #self.hands[handType2]['pose']=[0.0,0.0,depth_frame.get_distance(centerPoint2[0]+v, centerPoint2[1]+u)]
                self.hands[handType2]['height']=float(depth_frame.get_distance(int(centerPoint2[0]+u), int(centerPoint2[1]+v)))


        if hand2 is not None:
                bbox2 = hand2["bbox"] 
                centerPoint2 = hand2["center"] 
                handType2 = hand2["type"] 
                cv2.rectangle(color_image, (bbox2[0], bbox2[1]), (bbox2[0]+bbox2[2], bbox2[1]+bbox2[3]), (0,255,0), 1)
                cv2.circle(color_image, (centerPoint2[0], centerPoint2[1]), 2, (255, 255, 255), -1)
                cv2.putText(color_image, str(handType2), (centerPoint2[0], centerPoint2[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                position_object=self.twod_to_threed(centerPoint2[0]+u,centerPoint2[1]+v,depth_frame.get_distance(centerPoint2[0]+v, centerPoint2[1]+u))
                self.hands[handType2]={}
                self.hands[handType2]['position']=[centerPoint2[0],centerPoint2[1]]
                self.hands[handType2]['lenght']=bbox2[2]
                self.hands[handType2]['width']=bbox2[3]
                self.hands[handType2]['pose']=[RS_CAM_X+position_object[1],RS_CAM_Y-position_object[0],RS_CAM_Z-position_object[2]]
                self.hands[handType2]['height']=float(depth_frame.get_distance(int(centerPoint2[0]+u), int(centerPoint2[1]+v)))
        return hand1,hand2,color_image
    def observe_environment(self):
        x, y, w, h = 0, 150, 350, 400
        align_to = rs.stream.color
        align = rs.align(align_to)
        fr=0
        try:
            time=0
            while not rospy.is_shutdown():
                frames = self.pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                self.intr = depth_frame.profile.as_video_stream_profile().get_intrinsics()
                if not depth_frame or not color_frame:
                    continue
                depth_image = np.asanyarray(depth_frame.get_data())[x:x+w, y:y+h]
                color_image = np.asanyarray(color_frame.get_data())[x:x+w, y:y+h]
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.06), cv2.COLORMAP_JET)
                hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
                lower = np.array([0,0,122])
                upper = np.array([179,255,255])
                mask = cv2.inRange(hsv, lower, upper)
                result = cv2.bitwise_and(color_image, color_image, mask = mask)
                result_color=None
                if time>40:
                        result_temp =None
                        for filter in self.filters.keys():
                            lower=np.array(self.filters[filter][0][0])
                            upper=np.array(self.filters[filter][0][1])
                            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
                            mask = cv2.inRange(hsv, lower, upper)
                            if self.filters[filter][1] is not None:
                                if result_temp is not None:
                                    result_color = cv2.bitwise_and(result_temp, result_temp, mask = mask)
                                    color_image=self.tagger(result_color, color_image,self.filters[filter][1],x,y,depth_frame)
                                    result_temp=None
                                else:
                                    result_color = cv2.bitwise_and(result, result, mask = mask)
                                    color_image=self.tagger(result_color, color_image,self.filters[filter][1],x,y,depth_frame)
                            else:
                                result_temp = cv2.bitwise_and(result, result, mask = mask)  
                else:
                    time+=1
                self.filter_pass(color_image,result)
                hand1,hand2,color_image=self.hand_detector(color_image,x,y,depth_frame)
                fr+=1
                if fr%20==0:
                    objects_b={}
                    if hand1 is not None or hand2 is not None:
                        objects_b=self.objects_behind(hand1,hand2)
                    self.objects={}
                    self.objects_temp={}
                    self.hands={}
                    self.goals={}
                    for obj in objects_b.keys():
                        self.objects[obj]=objects_b[obj]
                        self.objects_temp[obj]=objects_b[obj]
                self.publish_elemnts_info()
                self.publish_hands_info()
                self.publish_goal_info()
                #if result_color is not None:
                #    images = np.hstack((color_image))
                #else:
                #    images = np.hstack((color_image))  
                
                cv2.namedWindow('IDQL environment', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('IDQL environment', color_image)
                cv2.waitKey(1)
                if cv2.getWindowProperty('IDQL environment', cv2.WND_PROP_VISIBLE) <1:
                    break
                self.rate.sleep()
            cv2.destroyAllWindows()
        except Exception as e:
            rospy.logerr(e)
        self.pipeline.stop()
    def publish_elemnts_info(self):
        elements=Elements()
        position=Int32MultiArray()
        color=Int32MultiArray()
        try:
            for key in self.objects.keys():
                elements.key.append(key)
                elements.x.append(self.objects[key]['position'][0])
                elements.y.append(self.objects[key]['position'][1])
                elements.z.append(self.objects[key]['position'][2])
                elements.coorx.append(self.objects[key]['pose'][0])
                elements.coory.append(self.objects[key]['pose'][1])
                elements.coorz.append(self.objects[key]['pose'][2])
                #elements.pos=position
                elements.lenght.append(int(self.objects[key]['lenght']))
                elements.width.append(int(self.objects[key]['width']))
                elements.height.append(self.objects[key]['height'])
                elements.shape.append(self.objects[key]['shape'])
                elements.r.append(self.objects[key]['color'][0])
                elements.g.append(self.objects[key]['color'][1])
                elements.b.append(self.objects[key]['color'][2])
                #elements.path.append([0])
                #elements.grid.append([0])
                #elements.pathsmooth.append([0])
                elements.goal.append("")
            if len(elements.key)>0:
                #rospy.loginfo(elements)
                self.pub.publish(elements)
        except Exception as e:
            print(e)
            pass
    def publish_hands_info(self):
        elements=Elements()
        try:
            if len(self.hands.keys())>0:
                for key in self.hands.keys():
                    elements.key.append(key)
                    elements.x.append(self.hands[key]['position'][0])
                    elements.y.append(self.hands[key]['position'][1])
                    elements.coorx.append(self.hands[key]['pose'][0])
                    elements.coory.append(self.hands[key]['pose'][1])
                    elements.coorz.append(self.hands[key]['pose'][2])
                    #elements.z=self.hands[key]['position'][2]
                    elements.lenght.append(self.hands[key]['lenght'])
                    elements.width.append(self.hands[key]['width'])
                    elements.height.append(self.hands[key]['height'])
                    elements.shape.append("square")
                    elements.r.append(255)
                    elements.g.append(255)
                    elements.b.append(255)

            else:
                for key in ["left","right"]:
                    elements.key.append(key)
                
            #rospy.loginfo(elements)
            self.pub_hands.publish(elements)
        except Exception as e:
            pass
            #rospy.loginfo(elements)
            #self.pub_hands.publish(elements)
    def publish_goal_info(self):
        elements=Elements()
        position=Int32MultiArray()
        try:
            for key in self.goals.keys():
                elements.key.append(key)
                elements.x.append(self.goals[key]['position'][0])
                elements.y.append(self.goals[key]['position'][1])
                elements.coorx.append(self.goals[key]['pose'][0])
                elements.coory.append(self.goals[key]['pose'][1])
                elements.coorz.append(self.goals[key]['pose'][2])
                elements.height.append(self.goals[key]['height'])
                #elements.z=self.objects[key]['position'][2]
            if len(elements.key)>0:
                #rospy.loginfo(elements)
                self.pub_goal.publish(elements)
        except Exception as e:
            pass
    def objects_behind(self,hand1,hand2):
        xmin=[]
        ymax=[]
        ymin=[]
        xmax=[]
        objects_b={}
        if hand1 is not None:
                #lmList2 = hand2["lmList"]  # List of 21 Landmarks points
                bbox2 = hand1["bbox"]  # Bounding Box info x,y,w,h
                xmin.append(bbox2[0])
                ymin.append(bbox2[1])
                ymax.append(bbox2[0]+bbox2[2])
                xmax.append(bbox2[1]+bbox2[3])
        if hand2 is not None:
                bbox2 = hand2["bbox"] 
                xmin.append(bbox2[0])
                ymin.append(bbox2[1])
                ymax.append(bbox2[0]+bbox2[2])
                xmax.append(bbox2[1]+bbox2[3])
        for object in self.objects.keys():
            x=self.objects[object]['position'][0]
            y=self.objects[object]['position'][1]
            for i in range(0,len(xmin)):
                if x<xmax[i] and x>xmin[i] and y<ymax[i] and y>ymin[i]:
                    objects_b[object]=self.objects[object]
        return objects_b
                

    def tagger(self,img, image,name,u,v,depth_frame):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        types=""
        objective={}
        try:
            for c in cnts:
                w=max([xmax[0][0] for xmax in c])-min([xmax[0][0] for xmax in c])
                l=max([xmax[0][1] for xmax in c])-min([xmax[0][1] for xmax in c])
                approx = cv2.approxPolyDP(c,0.055*cv2.arcLength(c,True),True)
                if len(approx)==5 and name!='b':
                    types="pentagon"
                elif len(approx)==3:
                    types="triangle"
                elif len(approx)==4:
                    types="square"
                elif len(approx) == 6 and name!='b':
                    types="hexa"
                elif len(approx) == 7 and name!='b':
                    types="hexa"
                elif len(approx) > 7:
                    types="circle"
                elif len(approx) > 4 and name=='b':
                    types="circle"
                else:
                    types=""
                M = cv2.moments(c)
                try:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    color=[255,255,255]
                    if name=='y' and types=='square':
                        continue
                    if name=='y':
                        types="hexa"
                    if name=='g':
                        types="square"
                    if name=='r':
                        types="square"
                    if name=='b':
                        types="circle"


                    color=self.get_color(name)
                    if name!='goal' and name!='hand'  and l.astype(float)> 6 and w.astype(float)>6:          
                        position_object=self.twod_to_threed(int(cX+u),int(cY+v),depth_frame.get_distance(int(cX+v), int(cY+u)))
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]={}
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['position']=[cX,cY,0]
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['pose']=[RS_CAM_X+position_object[1],RS_CAM_Y-position_object[0],RS_CAM_Z-position_object[2]]
                        #self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['pose']=[0.0,0.0,depth_frame.get_distance(centerPoint2[0]+v, centerPoint2[1]+u)]
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['orientation']=[0,0,0]
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['lenght']=float(l.astype(float))
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['width']=float(w.astype(float))
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['height']=float(depth_frame.get_distance(int(cX+v), int(cY+u)))
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['shape']=types    
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['color']=color
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['path']=[]
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['grid']=[]
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['path_smooth']=[]
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['goal']=[]
                        self.objects_temp[name+"_"+types[0]+"_"+str(int(cX/10))+"_"+str(int(cY/10))]['contours']=[c]
                    else:
                        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                        cv2.circle(image, (cX, cY), 2, (255, 255, 255), -1)
                        if types=='square' and name=='goal':
                            #self.goal_1[name+"_1"]=[cX, cY + 40]
                            cv2.putText(image, name+"_1", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1) 
                            position_object=self.twod_to_threed(int(cX+u),int(cY+v),depth_frame.get_distance(int(cX+v), int(cY+u)))
                            self.goals['right']={}
                            self.goals['right']['position']=[cX,cY-45]
                            self.goals['right']['pose']=[RS_CAM_X+position_object[1],RS_CAM_Y-position_object[0],RS_CAM_Z-position_object[2]]
                            self.goals['right']['height']=float(depth_frame.get_distance(int(cX+v), int(cY+u)))
                            
                        elif types=='triangle' and name=='goal':
                            cv2.putText(image, name+"_2", (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            #self.goal_2[name+"_2"]=[cX, cY + 40]
                            position_object=self.twod_to_threed(int(cX+u),int(cY+v),depth_frame.get_distance(int(cX+v), int(cY+u)))
                            self.goals['left']={}
                            self.goals['left']['position']=[cX,cY-45]
                            self.goals['left']['pose']=[RS_CAM_X+position_object[1],RS_CAM_Y-position_object[0],RS_CAM_Z-position_object[2]]
                            self.goals['left']['height']=float(depth_frame.get_distance(int(cX+v), int(cY+u)))
                except Exception as e:
                    pass
        except Exception as e:
            pass
        pop_out=[]
        pop_in=[]
        for object in self.objects_temp.keys():
            if object not in self.objects.keys():
                close_obj=False
                i_tmp=self.objects_temp[object]['position'][0]
                j_tmp=self.objects_temp[object]['position'][1]                
                for obj in self.objects.keys():    
                    i_ref=self.objects[obj]['position'][0]
                    j_ref=self.objects[obj]['position'][1]
                    if (i_ref<i_tmp+3 and i_ref>i_tmp-3) and (j_ref<j_tmp+3 and j_ref>j_tmp-3):
                        close_obj=True
                        pop_out.append(obj)
                        pop_in.append(object)
                        break
                if not close_obj:
                    self.objects[object]=self.objects_temp[object]
        for p_out in pop_out:
            try:
                self.objects.pop(p_out)
            except Exception as e:
                continue
        for o_in in pop_in:
            self.objects[o_in]=self.objects_temp[o_in]
        return image
    def filter_pass(self,color_image,result):
        pop_objects=[]
        for object in self.objects.keys():
            cX=self.objects[object]['position'][0]
            cY=self.objects[object]['position'][1]
            le=self.objects[object]['lenght']
            wi=self.objects[object]['width']
            blank_space=np.asanyarray(result)[int(cY)-int(le/7):int(cY)+int(le/7),int(cX)-int(wi/7):int(cX)+int(wi/7)]
            if np.average(blank_space)==0.0:
                pop_objects.append(object)
        for pop_obj in pop_objects:
            self.objects.pop(pop_obj)
        
        pop_objects=[]

        for object in self.objects.keys():
            temp_objects=self.objects.copy()
            temp_objects.pop(object)
            cX=int(self.objects[object]['position'][0]/10)
            cY=int(self.objects[object]['position'][1]/10)
            for obj in temp_objects.keys():
                X=int(temp_objects[obj]['position'][0]/10)
                Y=int(temp_objects[obj]['position'][1]/10)
                if obj[0]==object[0]:
                    if cX+1==X or cX-1==X:
                        if cY+1==Y or cY-1==Y:
                            pop_objects.append(obj)
        for pop_obj in pop_objects:
            try:
                self.objects.pop(pop_obj)
            except:
                continue
        for object in self.objects.keys():
            c=self.objects[object]['contours']
            cX=self.objects[object]['position'][0]
            cY=self.objects[object]['position'][1]
            types=self.objects[object]['shape'][0]
            cv2.drawContours(color_image, c, -1, (0, 255, 0), 2)
            cv2.circle(color_image, (int(cX), int(cY)), 2, (255, 255, 255), -1)
            cv2.putText(color_image, object, (cX - 20, cY - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return color_image
    def get_color(self,name):
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


if __name__ == '__main__':
    camera=perRealsense()
    camera.observe_environment()
