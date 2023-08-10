#!/usr/bin/env python3

import pyttsx3
import numpy as np
import time
import cv2
import math
from scipy import interpolate
import random
import pyrealsense2 as rs
from irohms_iql.msg import Elements
import speech_recognition as sr
import subprocess
from gtts import gTTS
import rospy
import re
import os
#from geometry_msgs.msg import PoseStamped, PoseArray
from constants import *

engine = pyttsx3.init()
safe_mode=True

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
TRAINING_STEPS=10001

RS_CAM_X=-0.58
RS_CAM_Y=-0.24
RS_CAM_Z=1.21

u_default=0
v_default=150

ref_position=[0,0,0]

def make_intrinsics():
    intrinsics = rs.intrinsics()
    intrinsics.coeffs = [0,0,0,0,0]
    intrinsics.fx = 615.799 
    intrinsics.fy = 615.836
    intrinsics.height = 480
    intrinsics.ppx = 317.985
    intrinsics.ppy = 244.153
    intrinsics.width=640
    return intrinsics

intr=make_intrinsics()

##Reinforcement Learning

class IDQL():
    def __init__(self,objects):
        self.objects=objects
        self.reset()
    def reset(self):
        self.work_space = np.zeros((SH,SW))
        self.work_space = cv2.merge([self.work_space,self.work_space,self.work_space])

    def set_env(self,key,goal):
        print(key)
        self.temp_objects=self.objects.copy()
        self.temp_objects.pop(key,None)
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
            ve


rtical.append([i*x_ref-x_offset,0,i*x_ref-x_offset,SH])
        for j in range(0,m):
            horizontal.append([0,j*y_ref-y_offset,SW,j*y_ref-y_offset])
        #when offset x is positive add a column at the begining
        state=np.zeros((n, m))
        x=int(x)
        y=int(y)
        #print(x,y)
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
        for ele in self.temp_objects.keys():
            corners=self.get_corners(ele,x_offset, y_offset)
            for corner in corners:
                i=int(math.floor((corner[0])/x_ref))
                j=int(math.floor((corner[1])/y_ref))
                state[i][j]=0.2
        i=int(math.floor((goal[0]+x_offset)/x_ref))
        j=int(math.floor((goal[1]+y_offset)/y_ref))
        state[i][j]=0.3
        u,v=state.shape
        for s in range(0,u*v):
            a=(int(s/v))
            b=(((s)%v))
            c=(int((s)/(u*v)))
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
        if (a+1)<state.shape[0]:
            if state[a+1][b]==0.0 or state[a+1][b]==0.1:
                IOTA[0]=1
                q_values[0]=1
                states_[0]=['s'+str(next_s[0]),-0,False]
            elif state[a+1][b]==0.3:
                IOTA=[0,0,0,0,0,0,0,0,0]
                q_values=[0,0,0,0,0,0,0,0,0]
                IOTA[0]=1
                states_[0]=['s'+str(next_s[0]),10,False]
                q_values[0]=1.1
                return states_, IOTA, q_values
        ##LEFT
        if (a-1)>=0:
            if state[a-1][b]==0.0 or state[a-1][b]==0.1:
                IOTA[2]=1
                q_values[2]=1
                states_[2]=['s'+str(next_s[2]),-0,False]
            elif state[a-1][b]==0.3:
                IOTA=[0,0,0,0,0,0,0,0,0]
                q_values=[0,0,0,0,0,0,0,0,0]
                IOTA[2]=1
                states_[2]=['s'+str(next_s[2]),10,False]
                q_values[2]=1.1
                return states_, IOTA, q_values
        ###UP
        if (b-1)>=0:
            if state[a][b-1]==0.0 or state[a][b-1]==0.1:
                IOTA[1]=1
                q_values[1]=1
                states_[1]=['s'+str(next_s[1]),-0,False]
            elif state[a][b-1]==0.3:
                IOTA=[0,0,0,0,0,0,0,0,0]
                q_values=[0,0,0,0,0,0,0,0,0]
                IOTA[1]=1
                states_[1]=['s'+str(next_s[1]),10,False]
                q_values[1]=1.1
                return states_, IOTA, q_values
        ##DOWN
        if (b+1)<state.shape[1]:
            if state[a][b+1]==0.0 or state[a][b+1]==0.1:
                IOTA[3]=1
                q_values[3]=1
                states_[3]=['s'+str(next_s[3]),-0,False]
            elif state[a][b+1]==0.3:
                IOTA=[0,0,0,0,0,0,0,0,0]
                q_values=[0,0,0,0,0,0,0,0,0]
                IOTA[3]=1
                states_[3]=['s'+str(next_s[3]),10,False]
                q_values[3]=1.1
                return states_, IOTA, q_values
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
        return None
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
                cv2.line(self.work_space, (int(line[4]),int(line[5])), (int(line[2]),int(line[3])), [255,0,255], 2)
        
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
                            size=2
                        ##
                        if index==0:
                            cv2.circle(self.work_space, (x_offset-3+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2)+5,int(y_offset-5+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)), size, coloration, -1)
                        if index==2:
                            cv2.circle(self.work_space, (x_offset-3+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2)-5,int(y_offset-5+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)), size, coloration, -1)
                        if index==1:
                            cv2.circle(self.work_space, (x_offset-3+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2),int(y_offset-5+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)-5), size, coloration, -1)
                        if index==3:
                            cv2.circle(self.work_space, (x_offset-3+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2),int(y_offset-5+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)+5), size, coloration, -1)
                        if index==4 and cir==1:
                            cv2.circle(self.work_space, (x_offset+int(x*self.objects[obj]['width']+self.objects[obj]['width']/2),int(y_offset+y*self.objects[obj]['lenght']+self.objects[obj]['lenght']/2)), 4, BLUE_GREEN, -1)
                        index+=1
        if len(self.objects[obj]['path_smooth'])>0:
            for point in self.objects[obj]['path_smooth']:
                cv2.circle(self.work_space, (int(point[0]),int(point[1])), 4, BLUE_GREEN, -1)
                cv2.imshow('Work space',self.work_space)
                key=cv2.waitKey(int(round(25/fps)))  # We need to call cv2.waitKey after cv2.imshow
                if key == 0:  # Press Esc for exit
                    break
        cv2.imshow('Work space',self.work_space)
        #cv2.waitKey(0)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        
        #cv2.waitKey()
        #br=CvBridge()
        #pub_qtable.publish(self.br.cv2_to_imgmsg(self.work_space))

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

##Perception

class Environment():
    def __init__(self):
        self.objects={}
        self.goals={}
        self.hands={}
        self.data_objects=[]
        self.data_goals={}
        self.data_hands={}
        rospy.Subscriber('/irohms_idql/irohms_perception_table_elements', Elements, self.get_objects)
        rospy.Subscriber('/irohms_idql/irohms_perception_table_goals', Elements, self.get_goals)
        rospy.Subscriber('/irohms_idql/irohms_perception_table_hands', Elements, self.get_hands)
    def get_objects(self,data):
        self.data_objects=data
        #print(data)
    def observe_environment(self):
        self.objects={}
        self.goals={}
        try:
            for i in range(0,len(self.data_objects.key)):
                #print(self.data_objects.key[i])
                self.objects[self.data_objects.key[i]]={}
                self.objects[self.data_objects.key[i]]['position']=[self.data_objects.x[i],self.data_objects.y[i],0]
                self.objects[self.data_objects.key[i]]['orientation']=[0,0,0]
                self.objects[self.data_objects.key[i]]['lenght']=self.data_objects.width[i]
                self.objects[self.data_objects.key[i]]['width']=self.data_objects.lenght[i]
                self.objects[self.data_objects.key[i]]['height']=self.data_objects.height[i]
                self.objects[self.data_objects.key[i]]['shape']=self.data_objects.shape[i]    
                self.objects[self.data_objects.key[i]]['color']=[self.data_objects.r[i],self.data_objects.g[i],self.data_objects.b[i]]
                self.objects[self.data_objects.key[i]]['path']=[]
                self.objects[self.data_objects.key[i]]['grid']=[]
                self.objects[self.data_objects.key[i]]['path_smooth']=[]
                self.objects[self.data_objects.key[i]]['goal']=[]
        except:
            pass
     
        try:
            for i in range(0,len(self.data_goals.key)):
                #print(self.data_goals.x[i],self.data_goals.y[i])
                self.goals[self.data_goals.key[i]]={}
                self.goals[self.data_goals.key[i]]['position']=[self.data_goals.x[i],self.data_goals.y[i],0]
                self.goals[self.data_goals.key[i]]['orientation']=[0,0,0]
        except Exception as e:
            #print (e)
            pass
        try:
            for i in range(0,len(self.data_hands.key)):
                #print(self.data_goals.x[i],self.data_goals.y[i])
                self.hands[self.data_hands.key[i]]={}
                self.hands[self.data_hands.key[i]]['position']=[self.data_hands.x[i],self.data_hands.y[i],0]
                self.hands[self.data_hands.key[i]]['orientation']=[0,0,0]
                self.hands[self.data_hands.key[i]]['width']=30
                self.hands[self.data_hands.key[i]]['lenght']=20
                self.hands[self.data_hands.key[i]]['shape']='square'
                self.hands[self.data_hands.key[i]]['color']=[255,255,255]
                self.hands[self.data_hands.key[i]]['goal']=[]
                self.hands[self.data_hands.key[i]]['path']=[]
                self.hands[self.data_hands.key[i]]['grid']=[]
                self.hands[self.data_hands.key[i]]['path_smooth']=[]
                
        except Exception as e:
            #rospy.logerr(e)
            pass
        return self.objects, self.goals
    def get_goals(self,data):
        #rospy.loginfo(data.key)
        self.data_goals=data
        return
    def get_hands(self,data):
        #rospy.loginfo(data.key)
        
        self.data_hands=data
        #rospy.logerr(data)
        return

class RealTimeTracker():
    def __init__(self,objects):
        self.objects=objects
        self.reset()
    def reset(self):
        self.work_space = np.zeros((SH,SW))
        self.work_space = cv2.merge([self.work_space,self.work_space,self.work_space])
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
    def set_env(self,key,goal):
        self.temp_objects=self.objects.copy()
        self.temp_objects.pop(key,None)
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
        #when offset x is positive add a column at the begining
        state=np.zeros((n, m))
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
        for ele in self.temp_objects.keys():
            corners=self.get_corners(ele,x_offset, y_offset)
            for corner in corners:
                i=math.floor((corner[0])/x_ref)
                j=math.floor((corner[1])/y_ref)
                state[i][j]=0.2
        i=math.floor((goal[0]+x_offset)/x_ref)
        j=math.floor((goal[1]+y_offset)/y_ref)
        state[i][j]=0.3
        u,v=state.shape
        for s in range(0,u*v):
            a=(int(s/v))
            b=(((s)%v))
            c=(int((s)/(u*v)))
            states_,IOTA,q_values=self.iota(state,a,b,c,s,i,j)
            iEnv['s'+str(s)]=[states_,IOTA]
            iQtable.append(q_values)
        #print(iQtable)
        #print(len(iQtable),u*v)
        #return hh
        #env={'s1':[['sn','reward','done'],[],[],[]]}
        #iQTable=[[],[],[],..,[]]
        return vertical,horizontal,state,x,y, iQtable,actions,x_offset, y_offset, iEnv
    def testing(self,obj,goal,m_field,actions):
        vertical,horizontal,state,x,y, iQtable,acts, x_offset, y_offset,iEnv =self.set_env(obj,goal)
        msg, error, actions=self.agent(iEnv,x,y,state,actions)
        print("[ERROR]",msg,error)
        cv2.startWindowThread()
        #print(iQtable)
        robot.render(obj,vertical,horizontal,state, x_offset, y_offset,iQtable,m_field)
        if actions is not None:
            for a in actions:
                self.render(obj,vertical,horizontal,state, x_offset, y_offset,iQtable,m_field,msg,error)
                try:
                    self.step(obj,a)
                except Exception as e:
                    rospy.logerr(e)
                self.reset()
                #cv2.waitKey()  # We need to call cv2.waitKey after cv2.imshow
                #if key == 0:  # Press Esc for exit
                #    break
        #return error
        return error
    def agent(self,iEnv,x,y,s,actions):      
        state=int(s.shape[1]*x)+y
        done=False
        total_actions=[]
        msg="No problem"
        for i in range(0,len(actions)):
            action=actions[i]
            total_actions.append(action)
            #if iEnv['s'+str(state)][0][action] is None:
            #    iEnv['s'+str(state)][0][action]
            #    return msg,True, total_actions
            state_,reward,done=iEnv['s'+str(state)][0][action]
            print("Reward: ",reward)
            if reward<0:
                msg="Collision!"
                print("Collision!")
                return msg, True, total_actions
            if done:
                print("Finished")
                return msg,False, total_actions
            state_=int(state_.split('s')[1])
            state=state_
            #actions.append(action)
        #if not done:
        #    msg="Goal moved!!"
        #    print("Goal moved!!")
        #    return msg, True , total_actions
        return msg,False, total_actions
    def render(self,obj,vertical,horizontal,state,x_offset, y_offset,iQtable,m_field,msg,error):
            if error:
                cv2.putText(self.work_space, msg, (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, [0,0,255], 3, cv2.LINE_AA, False)
            else:
                cv2.putText(self.work_space, "No errors", (40,50), cv2.FONT_HERSHEY_SIMPLEX, 1.8, [0,255,0], 3, cv2.LINE_AA, False)

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
            if len(self.objects[obj]['path'])>0:
                for line in self.objects[obj]['path']:
                    cv2.line(self.work_space, (int(line[4]),int(line[5])), (int(line[2]),int(line[3])), [255,0,255], 2)
            cv2.imshow('Error tracker',self.work_space)
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
    def iota(self,state,a,b,c,current_s,i,j):
            IOTA=[0,0,0,0,0,0,0,0,0]
            q_values=[0,0,0,0,0,0,0,0,0]
            states_=[['s'+str(current_s),1,False],['s'+str(current_s),1,False],['s'+str(current_s),1,False],['s'+str(current_s),1,False],
            ['s'+str(current_s),1,False],['s'+str(current_s),1,False],['s'+str(current_s),1,False],['s'+str(current_s),1,False],['s'+str(current_s),1,False]]
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
            if (a+1)<state.shape[0]:
                if state[a+1][b]==0.0 or state[a+1][b]==0.1:
                    IOTA[0]=1
                    q_values[0]=1
                    states_[0]=['s'+str(next_s[0]),-0,False]
                if state[a+1][b]==0.2:
                    #IOTA[0]=0
                    q_values[0]=0
                    states_[0]=['s'+str(next_s[0]),-1,False]
                if state[a+1][b]==0.3:
                    IOTA=[0,0,0,0,0,0,0,0,0]
                    q_values=[0,0,0,0,0,0,0,0,0]
                    IOTA[0]=1
                    states_[0]=['s'+str(next_s[0]),10,True]
                    q_values[0]=1.1
                    return states_, IOTA, q_values
            ##LEFT
            if (a-1)>=0:
                if state[a-1][b]==0.0 or state[a-1][b]==0.1:
                    IOTA[2]=1
                    q_values[2]=1
                    states_[2]=['s'+str(next_s[2]),0,False]
                if state[a-1][b]==0.2:
                    IOTA[2]=0
                    q_values[2]=0
                    states_[2]=['s'+str(next_s[0]),-1,False]
                elif state[a-1][b]==0.3:
                    IOTA=[0,0,0,0,0,0,0,0,0]
                    q_values=[0,0,0,0,0,0,0,0,0]
                    IOTA[2]=1
                    states_[2]=['s'+str(next_s[2]),10,True]
                    q_values[2]=1.1
                    return states_, IOTA, q_values
            ###UP
            if (b-1)>=0:
                if state[a][b-1]==0.0 or state[a][b-1]==0.1:
                    IOTA[1]=1
                    q_values[1]=1
                    states_[1]=['s'+str(next_s[1]),0,False]
                elif state[a][b-1]==0.2:
                    IOTA[1]=0
                    q_values[1]=0
                    states_[1]=['s'+str(next_s[0]),-1,False]
                elif state[a][b-1]==0.3:
                    IOTA=[0,0,0,0,0,0,0,0,0]
                    q_values=[0,0,0,0,0,0,0,0,0]
                    IOTA[1]=1
                    states_[1]=['s'+str(next_s[1]),10,True]
                    q_values[1]=1.1
                    return states_, IOTA, q_values
            ##DOWN
            if (b+1)<state.shape[1]:
                if state[a][b+1]==0.0 or state[a][b+1]==0.1:
                    IOTA[3]=1
                    q_values[3]=1
                    states_[3]=['s'+str(next_s[3]),0,False]
                elif state[a][b+1]==0.2:
                    IOTA[3]=0
                    q_values[3]=0
                    states_[3]=['s'+str(next_s[0]),-1,False]
                elif state[a][b+1]==0.3:
                    IOTA=[0,0,0,0,0,0,0,0,0]
                    q_values=[0,0,0,0,0,0,0,0,0]
                    IOTA[3]=1
                    states_[3]=['s'+str(next_s[3]),10,True]
                    q_values[3]=1.1
                    return states_, IOTA, q_values
            if a<i and q_values[0]>0 and q_values[0]>0:
                q_values[0]=q_values[0]+0.1
            if a>i and q_values[2]>0 and q_values[2]>0:
                q_values[2]=q_values[2]+0.1
            if b<j and q_values[3]>0 and q_values[3]>0:
                q_values[3]=q_values[3]+0.1
            if b>j and q_values[1]>0 and q_values[1]>0:
                q_values[1]=q_values[1]+0.1
            return states_,IOTA,q_values
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

##NLP functions

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

def describe_instructions(goals):
    description="I am going to put the "
    and_the=0
    for sub_goal in goals.keys():
        blue=0
        red=0
        green=0
        yellow=0
        and_obj=0
        for objeto in goals[sub_goal]:
            color=""
            if objeto[0]=='b':
                if blue<1:
                    color="blue"
                    blue+=1
            elif objeto[0]=='r':
                color=""
                if red<1:
                    color="red"
                    red+=1
            elif objeto[0]=='g':
                if green<1:
                    color="green"
                    green+=1
            elif objeto[0]=='y':
                if yellow<1:
                    yellow+=1
                    color="yellow"
            and_obj+=1
            if and_obj==len(goals[sub_goal])+1:
                description+=" and "
            description+=str(color)+" "
        if len(goals[sub_goal])>1:
            description+="objects in the "+str(sub_goal)+" box "
        else:
            description+="object in the "+str(sub_goal)+" box "
        if len(goals.keys())>1:
            and_the+=1
            if and_the<len(goals.keys()):
                description+="and the "
    rospy.loginfo(description)
    speak_phrase(description)

def running(obj,goal,m_field,robot):
    vertical,horizontal,state,x,y, iQtable,actions, x_offset, y_offset,iEnv =robot.set_env(obj,goal)
    #solve iQTable return set of actions 
    table=np.array(iQtable)
    actionsss=robot.agent(table,iEnv,x,y,state)
    cv2.startWindowThread()
    if actionsss is not None:
        for a in actionsss:
            robot.render(obj,vertical,horizontal,state, x_offset, y_offset,iQtable,m_field)
            robot.step(obj,a)
            robot.reset()
            key=cv2.waitKey(int(round(50/fps)))  # We need to call cv2.waitKey after cv2.imshow
            if key == 0:  # Press Esc for exit
                break
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
            return list(zip(xint,yint)),actionsss
        except Exception as e:
            print("Interpolation error")
            print(e)
            return None,actionsss
    else:
        print("Not possible to handle piece")
        return None,actionsss

def get_speech_1():  
    #return
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source2: 
            r.adjust_for_ambient_noise(source2, duration=0.2)
            audio = r.listen(source2, timeout=3, phrase_time_limit=3)
            query = r.recognize_google(audio)
            print(query)
            word = query.lower()
            print(word)
            return word
            
    except Exception as e:
        rospy.logerr(e)
        rospy.loginfo("Not instruction spoken")
        return None

def get_speech():   
    #return
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source2: 
            r.adjust_for_ambient_noise(source2, duration=0.2)
            audio = r.listen(source2, timeout=5, phrase_time_limit=12)
            query = r.recognize_google(audio)
            word = query.lower()
            return word
            
    except Exception as e:
        #rospy.logerr(e)
        rospy.loginfo("Not instruction spoken")
        return None

def speak_phrase(phrase):
    engine.say(phrase)
# play the speech
    engine.runAndWait()
    #return
    #tts = gTTS(text=phrase,lang='en', slow=False)
    #tts.save("robot_phrase.mp3")
    #os.system("mpg321 robot_phrase.mp3")
    #time.sleep(3)


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

###MATH utilities

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

###Robot control





def execute(path,path_torso):
    #sss = simple_script_server()
    #handle_arm = sss.move("arm_right",[path[0]],False)
    #handle_arm.wait()
    #r = requests.get('http://10.4.13.52:5000/open', verify=False)
    #time.sleep(1)
    #handle_arm = sss.move("arm_right",[path[1]],False)
    #handle_arm.wait()
    #r = requests.get('http://10.4.13.52:5000/close', verify=False)
    #time.sleep(1)
    #handle_arm = sss.move("arm_right",path[2:],False)
    #handle_arm.wait()
    #r = requests.get('http://10.4.13.52:5000/open', verify=False)
    #TODO kuka robot and gripper
    return
def twod_to_threed(x,y,z):
    return rs.rs2_deproject_pixel_to_point(intr,[float(x),float(y)],z)
def get_robot_path(object_path,dist):
    path=[]
    path_torso=[]
    positions_realsense=twod_to_threed(object_path[0][0]+u_default,object_path[0][1]+v_default,dist)
    x=RS_CAM_X-positions_realsense[1]
    y=RS_CAM_Y-positions_realsense[0]
    z=RS_CAM_Z-positions_realsense[2]+0.2
    path.append([float(x),float(y),float(z)])
    #t1,t2,t3,t4,t5,t6,t7=inverse_kinematics_1(x,y,z)
    #path.append([float(t1),float(t2),float(t3),float(t4),float(t5),float(t6),float(t7)])
    #x=-0.535+object_path[0][0]*0.0025
    #y= 1.005-object_path[0][1]*0.0025
    print(x,y,z)
    z=RS_CAM_Z-1.17+0.14
    path.append([float(x),float(y),float(z)])
    #z=RS_CAM_Z-positions_realsense[2]+0.2
    #path.append([float(x),float(y),float(z)])
    #t1,t2,t3,t4,t5,t6,t7=inverse_kinematics_1(x,y,z)
    #path.append([float(t1),float(t2),float(t3),float(t4),float(t5),float(t6),float(t7)])
    path_torso.append([0.0,0.0])
    for point in object_path:
        positions_realsense=twod_to_threed(point[0]+u_default,point[1]+v_default,1.18)
        x=RS_CAM_X-positions_realsense[1]
        y=RS_CAM_Y-positions_realsense[0]
        z=RS_CAM_Z-positions_realsense[2]+0.25
        #print(x,y)
        #t1,t2,t3,t4,t5,t6,t7=inverse_kinematics_1(x,y,z)
        #print(rad_to_deg(t1))
        path.append([float(x),float(y),float(z)])
        path_torso.append([0.0,0.0])
    #TODO get kuka path
    return path,path_torso

def clean_objects(goals_location,temp_objects,goals):
    pop_keys=[]
    objects_2=temp_objects.copy()
    for key_o in temp_objects.keys():
        x_obj=temp_objects[key_o]['position'][0]
        y_obj=temp_objects[key_o]['position'][1]
        #print(x_obj,y_obj)
        for key_g in goals_location.keys():
            x_g=goals_location[key_g]['position'][0]
            y_g=goals_location[key_g]['position'][1]
            if x_obj>x_g-60 and x_obj<x_g+60 and y_obj>y_g-60 and y_obj<y_g+60:
                pop_keys.append(key_o)
                break
    new_objects=temp_objects.copy()
    for sub_goal in goals.keys():
        for thing in goals[sub_goal]:
            new_objects.pop(thing)
    for sub_goal in new_objects.keys():
        temp_objects.pop(sub_goal)
    for key in pop_keys:
        try:
            temp_objects.pop(key)
            objects_2.pop(key)
        except Exception as e:
            rospy.logerr(e)
    return temp_objects, objects_2

def copy_dict(dictionary):
    new_dict={}
    for key, value in dictionary.items():
        new_dict[key]=value
    return new_dict
def get_hand_dict(hand):
    hand_dict={}
    return hand_dict

def get_angles(pos,wrist):
    euler=[0,0,0]
    euler[0]=3.14159
    euler[2]=math.atan(float(pos[1])/float(pos[0]))-1.5707+wrist
    euler[2]=-1.5707+wrist
    print(euler[2]*360/(2*3.1416))
    return euler
def get_message_pose(pos,euler,height):
    grasping_pose = PoseStamped()
    grasping_pose.pose.position.x = pos[0]
    grasping_pose.pose.position.y = pos[1]
    grasping_pose.pose.position.z = pos[2]+height
    q=quaternion_from_euler(euler[0],euler[1],euler[2])
    grasping_pose.pose.orientation.w = q[3]
    grasping_pose.pose.orientation.x = q[0]
    grasping_pose.pose.orientation.y = q[1]
    grasping_pose.pose.orientation.z = q[2]
    return grasping_pose

def open_gripper():
    kuka.publish_grip_cmd(gripper_pre_1)
    return True
def close_gripper():
    kuka.publish_grip_cmd(gripper_close)
    return True
def move_arm(move):
    #kuka.publish_pose(waiting_pose)
    #print(move[0],move[1],move[2])
    euler=get_angles(move,0.0)
    grasping_pose=get_message_pose(move,euler,0.1)
    kuka.publish_pose(grasping_pose)
    return True

def execute_step(path,path_torso,instructions,objeto,goal,objects_3,actions_set,x,y,sub_goal):
    #TODO init kuka
    time.sleep(0.5)
    move_arm(path[0])
    open_gripper()
    move_arm(path[1])
    close_gripper()
    for move in path[2:]:
        handle_arm = move_arm(move)
        #move kuka to position
        ###look for changes
        if safe_mode:
            _,goals_loc=env.observe_environment()
            goal=goals_loc[sub_goal]['position']
            hands=env.hands.copy()
            env.hands={}
            try:
                for hand,item in hands.items():
                    if len(item.keys())>0:
                        objects_3[hand]=item
            except Exception as e:
                rospy.logerr(e)
            try:
                objects_3[objeto]['position']=[x,y,0]
                objects_3[objeto]['path']=[]
                changes=RealTimeTracker(objects_3)
                error=changes.testing(objeto,goal,False,actions_set)
                print(error)
                if error:
                    speak_phrase("Becareful with your hand")
                    time.sleep(2)
                    while error:
                        
                        _,goals_loc=env.observe_environment()
                        goal=goals_loc[sub_goal]['position']
                        hands=env.hands.copy()
                        env.hands={}
                        try:
                            for hand,item in hands.items():
                                if len(item.keys())>0:
                                    objects_3[hand]=item
                        except Exception as e:
                            rospy.logerr(e)
                        try:
                            objects_3[objeto]['position']=[x,y,0]
                            objects_3[objeto]['path']=[]
                            changes=RealTimeTracker(objects_3)
                            error=changes.testing(objeto,goal,False,actions_set)
                        except Exception as e:
                            rospy.logerr(e)
                            continue
                    speak_phrase("Thank you")

                    
                #print(objects_3)
            except Exception as e:
                rospy.logerr(e)
                continue
        ##objects_3[objeto]['position']=[x,y,0]
        ##
        ##
        ##print(error)
        ####TODO if error  then plan from current position to goal 
    #handle_arm.wait()
    open_gripper()
    kuka.home()
    return True


if __name__ == '__main__':
    #rospy.init_node('irohms_planner', anonymous=True)
    #while not rospy.is_shutdown():
    #    words=get_speech_1()
    #    exit()
    kuka.init_robot()
    time.sleep(7)
    env=Environment()
    actions=[0,0,0,0]
    objects_1={}
    manual=True #Set this to False if a spoken entry is required, otherwise write the instruction manually.
    ####Robot says I am ready#####
    rospy.loginfo("Robot ready!!")
    speak_phrase("Robot ready")
    speak_phrase("Waiting for instructions")
    while not rospy.is_shutdown():
        if not manual:
            while not rospy.is_shutdown():
                words=get_speech_1()
                if words is None:
                    continue
                if words=="finish":
                    speak_phrase("My job is done")
                    exit()
                if words=="get ready":
                    speak_phrase("ok")
                    rospy.loginfo("Ok!")
                    break
            while not rospy.is_shutdown():
                words=get_speech()
                print(words)
                words="put the green objects here"
                if words is None:
                    continue
                if words=="finish":
                    speak_phrase("My job is done")
                    exit()
                #rospy.loginfo("Instruction: ",str(words))
                if len(words.split("here"))>1:
                    #get hands position to the closest point
                    #change here for the goal
                    objects_1,goals_loc=env.observe_environment() 
                    not_hand=True
                    while not_hand:
                        objects_1,goals_loc=env.observe_environment() 
                        hands=env.hands.copy()
                        #print(hands)
                        for hand in hands.keys():
                            try:
                                x_h=hands[hand]['position'][0]
                                y_h=hands[hand]['position'][1]
                                for goal in goals_loc.keys():
                                    #print(hands[hand],goals_loc[goal])
                                    x_g=goals_loc[goal]['position'][0]
                                    y_g=goals_loc[goal]['position'][1]
                                    #rospy.loginfo(x_h,x_g,y_h,y_g)
                                    if x_h<x_g+40 and x_h>x_g-40 and y_h<y_g+40 and y_h>y_g-40:
                                        speak_phrase(str(goal)+" box")
                                        not_hand=False
                                        goal=str(goal)+" box"
                                        break
                            except Exception as e:
                                #pass
                                print(e)
                    string_here=words.split("here")
                    print(string_here)
                    words=' '.join([string_here[0],"in the "+goal])
                    speak_phrase("I got it")
                    time.sleep(2)
                    break
        else:
            words="put the blue objects to the left box"
        objects_1,goals_loc=env.observe_environment()  
        instructions=recognise_instructions(words)
        goals=get_sub_goals(instructions,objects_1) 
        temp_objects, objects_1=clean_objects(goals_loc,objects_1,goals)
        if len(temp_objects.keys())==0:
            rospy.loginfo("There are not objects to grasp on the table!")
            #rospy.loginfo("Please repeat the instruction.")
        else:
            describe_instructions(goals)
            #time.sleep(2)    
            for i in range(0,2):
                for sub_goal in goals.keys():
                    try:
                        goal=goals_loc[sub_goal]['position']
                    except:
                        print("No ",sub_goal, "box available!!")
                    for objeto in goals[sub_goal]:
                        try:
                            #print(goals[sub_goal])
                            x=int(objects_1[objeto]['position'][0])
                            y=int(objects_1[objeto]['position'][1])
                            objects_2=objects_1.copy()
                            #print(objects_1)
                            robot=IDQL(objects_1.copy())
                            robot.reset()
                            #print(objects_1.keys())
                            ##TODO get position before training
                            object_path,actions_set=running(objeto,goal,True,robot)
                            if  object_path is not None:
                                #print("executing path")
                                path,path_torso=get_robot_path(object_path,float(objects_2[objeto]['height']))
                                execute_step(path,path_torso,instructions,objeto,goal,objects_2,actions_set,x,y,sub_goal)
                                temp_objects.pop(objeto)
                                objects_1.pop(objeto)
                        except Exception as e:
                            rospy.logerr(e)
                        kuka.home()
                        objects_1,goals_loc=env.observe_environment()
                        goals=get_sub_goals(instructions,objects_1) 
                        temp_objects,objects_1=clean_objects(goals_loc,objects_1,goals)
                if len(temp_objects.keys())==0:
                    break
            if len(temp_objects.keys())==0:
                speak_phrase("Task finished")
            else:
                speak_phrase("Sorry I could not handle all the objects")
        cv2.destroyAllWindows()
        
