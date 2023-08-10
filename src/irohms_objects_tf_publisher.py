#!/usr/bin/env python2


import rospy
from std_msgs.msg import String
from irohms_iql.msg import Elements
from tf.transformations import euler_from_quaternion,quaternion_from_euler
import tf
import numpy as np




class EnvObjects():
    def __init__(self):
        self.trans=tf.TransformBroadcaster()
        rospy.init_node('irohms_tf_objects_publisher', anonymous=True)
        rospy.Subscriber("/irohms_idql/irohms_perception_table_elements", Elements, self.callback_elements)
        rospy.Subscriber("/irohms_idql/irohms_perception_table_goals", Elements, self.callback_goals)
        rospy.Subscriber("/irohms_idql/irohms_perception_table_hands", Elements, self.callback_hands)
        
    def callback_elements(self,data):
        #rospy.loginfo("Elements %s", data.key)
        try:
            for index in range(0,len(data.key)):
                q=quaternion_from_euler(0.0,0.0,0.0)
                #self.trans.sendTransform((data.coorx[index],data.coory[index],data.coorz[index]),(q[0],q[1],q[2],q[3]),rospy.Time.now(),str(data.key[index]),"iiwa_link_0")
        except Exception as e:
            pass
            #rospy.logerr(e)
    def callback_hands(self,data):
        #rospy.loginfo( "Hands %s", data.key)
        try:
            for index in range(0,len(data.key)):
                q=quaternion_from_euler(0.0,0.0,0.0)
                self.trans.sendTransform((data.coorx[index],data.coory[index],data.coorz[index]),(q[0],q[1],q[2],q[3]),rospy.Time.now(),str(data.key[index]+"_hand"),"iiwa_link_0")
        except Exception as e:
            pass
            #rospy.logerr(e)
    def callback_goals(self,data):
        #rospy.loginfo( "Goals %s", data.key)
        try:
            for index in range(0,len(data.key)):
                q=quaternion_from_euler(0.0,0.0,0.0)
                self.trans.sendTransform((data.coorx[index],data.coory[index],data.coorz[index]),(q[0]+0.05,q[1],q[2],q[3]),rospy.Time.now(),str(data.key[index]+"_box"),"iiwa_link_0")
        except Exception as e:
            #pass
            rospy.logerr(e)
        
        

if __name__ == '__main__':
    EnvObjects()
    rospy.spin()