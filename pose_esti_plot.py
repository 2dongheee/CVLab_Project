import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.animation as animation

import time
import rospy
from std_msgs.msg import Float64MultiArray 
from sensor_msgs.msg import Imu

import signal, sys
from SM_OpticalFlow import Opticlaflow

class plot_pose_estimation():
    def __init__(self):
     
        self.getNewRosMsg = False
        self.getNewRosMsg_est = False

        self.idx = 0
        self.idx_est = 0

        self.fig = plt.figure()   
        self.ax = self.fig.gca(xlim=(-200,500), ylim=(0, -800), zlim=(30,70), projection='3d')
        
        self.ax.set_xlabel('X(m)')
        self.ax.set_ylabel('Y(m)')
        self.ax.set_zlabel('Z(m)')
        self.ax.set_title('Pose estimation')
        self.ax.legend(loc='upper right', fontsize=8, shadow=True)

        rospy.init_node('pose_plot')
        rospy.Subscriber('/imu_sensor/pose', Imu, self.gt_CallBack)
        rospy.Subscriber('/camera_sensor/pose', Imu, self.est_CallBack)
        # rospy.Subscriber('/imu_sensor/pose', Imu, self.gt_CallBack)

        # ax.grid(False)

        self.position = np.array([[1.], 
                                [1.], 
                                [1.],
                                [1.]])

        self.position_est = np.array([[-137.02], 
                                [70.26], 
                                [38.9],
                                [1.]])

        self.P_mat = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])

        self.P_mat_est = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [0, 0, 0, 0]])

        self.Gt_dataset = np.array([[0],
                                    [0],
                                    [0]])
        
        self.Est_dataset = np.array([[0],
                                    [0],
                                    [0]])

        self.gt_xdata, self.gt_ydata, self.gt_zdata = [-137.02], [70.26], [38.90]
        self.est_xdata, self.est_ydata, self.est_zdata = [-137.02], [70.26], [38.90]

        self.redDots_GT = self.ax.plot(self.gt_xdata, self.gt_ydata, self.gt_zdata, ms=1, c='r', marker='o')[0]
        self.redDots_Est = self.ax.plot(self.est_xdata, self.est_ydata, self.est_zdata, ms=1, c='b', marker='o')[0]
        
        ani_GT = animation.FuncAnimation(self.fig, self.animate, init_func= self.init, blit=False, repeat=False)
        ani_Est = animation.FuncAnimation(self.fig, self.animate_Est, init_func= self.init_Est, blit=False, repeat=False)
        self.fig.tight_layout()
        plt.show()


    def init(self):
        return self.redDots_GT

    def init_Est(self):
        return self.redDots_Est

    def est_CallBack(self, msg):
        euler_x, euler_y, euler_z = msg.orientation.x, msg.orientation.y, msg.orientation.z
        t_x, t_y, t_z =  msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z

        R_mat = self.euler_to_rotMat(euler_x, euler_y, euler_z)
        self.P_mat_est = np.array([[R_mat[0][0], R_mat[0][1], R_mat[0][2], t_z],
                                [R_mat[1][0], R_mat[1][1], R_mat[1][2], t_y],
                                [R_mat[2][0], R_mat[2][1], R_mat[2][2], t_x]])

        self.getNewRosMsg_est = True

    def gt_CallBack(self, msg):
        euler_x, euler_y, euler_z = msg.orientation.x, msg.orientation.y, msg.orientation.z
        t_x, t_y, t_z =  msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z
    
        R_mat = self.euler_to_rotMat(euler_x, euler_y, euler_z)
        self.P_mat = np.array([[R_mat[0][0], R_mat[0][1], R_mat[0][2], t_x],
                                [R_mat[1][0], R_mat[1][1], R_mat[1][2], t_y],
                                [R_mat[2][0], R_mat[2][1], R_mat[2][2], t_z]])

        self.getNewRosMsg = True

    def euler_to_rotMat(self, roll, pitch, yaw):
        Rz_yaw = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [          0,            0, 1]])

        Ry_pitch = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [             0, 1,             0],
            [-np.sin(pitch), 0, np.cos(pitch)]])

        Rx_roll = np.array([
            [1,            0,             0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]])
        
        rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
        return rotMat

    def getDataset(self):
        if self.getNewRosMsg :
            gt_data = np.dot(self.P_mat, self.position)
            gt_data[0][0] = round(gt_data[0][0], 2)
            gt_data[1][0] = round(gt_data[1][0], 2)
            gt_data[2][0] = round(gt_data[2][0], 2)
            print("GT",gt_data)
            
            absolutescale = math.sqrt((gt_data[0][0]-self.gt_xdata[self.idx-1])**2+(gt_data[1][0]-self.gt_ydata[self.idx-1])**2+(gt_data[2][0]-self.gt_zdata[self.idx-1])**2)
 
            if gt_data[0][0] != 0 and absolutescale > 0.1:
                self.gt_xdata.append(gt_data[0][0])
                self.gt_ydata.append(gt_data[1][0])
                self.gt_zdata.append(gt_data[2][0])
            
                self.Gt_dataset = np.array([gt_data[0][0], gt_data[1][0], gt_data[2][0]])
                self.idx += 1
                self.getNewRosMsg = False
                if self.idx == 1: print("Init Position(GT) : x={%0.2f}, y={%0.2f}, z={%0.2f}"%(gt_data[0][0], gt_data[1][0], gt_data[2][0]))
    
    def getDataset_Est(self):
        if self.getNewRosMsg_est :
            est_data = np.dot(self.P_mat, self.position_est)
            est_data[0][0] = round(est_data[0][0], 2)
            est_data[1][0] = round(est_data[1][0], 2)
            est_data[2][0] = round(est_data[2][0], 2)
            print("Est",est_data)
            
            absolutescale = math.sqrt((est_data[0][0]-self.est_xdata[self.idx_est-1])**2+(est_data[1][0]-self.est_ydata[self.idx_est-1])**2+(est_data[2][0]-self.est_zdata[self.idx_est-1])**2)
 
            if est_data[0][0] != 0 and absolutescale > 0.1:
                self.est_xdata.append(est_data[2][0])
                self.est_ydata.append(est_data[1][0])
                self.est_zdata.append(est_data[0][0])
            
                self.est_dataset = np.array([est_data[0][0], est_data[1][0], est_data[2][0]])
                self.idx_est += 1
                self.getNewRosMsg_est = False
                if self.idx_est == 1: print("Init Position(EST) : x={%0.2f}, y={%0.2f}, z={%0.2f}"%(est_data[0][0], est_data[1][0], est_data[2][0]))


    def animate(self, i):
        self.getDataset()
    
        self.redDots_GT.set_data(self.gt_xdata[:i], self.gt_ydata[:i] ) 
        self.redDots_GT.set_3d_properties(self.gt_zdata[:i]) 

        return self.redDots_GT

    def animate_Est(self, i):
        self.getDataset_Est()
    
        self.redDots_Est.set_data(self.est_xdata[:i], self.est_ydata[:i] ) 
        self.redDots_Est.set_3d_properties(self.est_zdata[:i]) 

        return self.redDots_Est
             
def signal_handler(signal, frame):
        print("Exit")
        exit(0)

if __name__=="__main__":
    # Opt = Opticlaflow()
    signal.signal(signal.SIGINT, signal_handler)
    plot = plot_pose_estimation()
    rospy.spin()
    
        
    

