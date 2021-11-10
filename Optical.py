import numpy as np
import cv2
import math

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from sensor_msgs.msg import Imu

from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Bool
import tf

from calibration import calib
from scipy.spatial.transform import Rotation

import time

class Opticlaflow:
    def __init__(self):
        self.K = np.array([[1108.5520500997836, 0.00000000e+00, 640.0], 
                        [0.00000000e+00, 1108.5520500997836, 403.0], 
                        [0.0, 0.0, 1.0]])
        self.dist = np.array([0, 0, 0, 0, 0])
        
        self.cur_img = {'img':None, 'header':None}
        self.get_new_img_msg = False
        self.get_new_imu_msg = False

        self.preproc_ct = 0

        self.Imu_Data = []
    
        self.cur_R, self.cur_t = None, None
        self.R_matrices, self.T_vectors = [], []

        self.pre_pose, self.cur_pose = None, None
      
        self.est_pose = None
        self.pre_est_euler = None

        self.bridge = CvBridge()
        
        rospy.init_node('opticalflow')
        rospy.Subscriber('/vds_node_localhost_2211/image_raw/compressed', CompressedImage, self.SM_IMGcallback)
        # rospy.Subscriber('/gmsl_camera/port_0/cam_0/image_raw/compressed', CompressedImage, self.SM_IMGcallback)

        rospy.Subscriber('/imu_sensor/pose', Imu, self.IMU2Roation, queue_size=10)
        
        self.cam_pose = rospy.Publisher('camera_pose', Float64MultiArray, queue_size=10)
        self.est_pose = rospy.Publisher('/camera_sensor/pose', Imu, queue_size=10)
        self.pub_od = rospy.Publisher('/od_result', Image, queue_size=1)


    def SM_IMGcallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (640, 403), interpolation=cv2.INTER_AREA)
        self.cur_img['img'] = self.undistort(img)
        self.cur_img['header'] = msg.header
        self.get_new_img_msg = True

    def undistort(self, img):
        w,h = (img.shape[1], img.shape[0])
        newcameraMtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist, (w,h),1,(w,h))
        dst = cv2.undistort(img, self.K, self.dist, None, newcameraMtx)
        x,y,w,h = roi
        undistorted_img = dst[y:y+h,x:x+w]

        return undistorted_img

    def IMU2Roation(self, msg):
        self.Imu_Data = msg.orientation.x, msg.orientation.y, msg.orientation.z ## Euler angle (radian)
        self.Position = msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z ## absolute Translation (m)

        # ## R_z
        # mat_yaw = np.array([[math.cos(self.Imu_Data[2]), -math.sin(self.Imu_Data[2]), 0],
        #                     [math.sin(self.Imu_Data[2]), math.cos(self.Imu_Data[2]), 0],
        #                     [0, 0, 1]])
        # ## R_y
        # mat_pitch = np.array([[math.cos(self.Imu_Data[1]),0,math.sin(self.Imu_Data[1])],
        #                     [0,1,0],
        #                     [-math.sin(self.Imu_Data[1]),0,math.cos(self.Imu_Data[1])]])
        # ## R_x
        # mat_roll = np.array([[1, 0, 0],
        #                     [0, math.cos(self.Imu_Data[0]), -math.sin(self.Imu_Data[0])],
        #                     [0, math.sin(self.Imu_Data[0]), math.cos(self.Imu_Data[0])]])

        # self.init_R = np.dot(mat_yaw, np.dot(mat_pitch, mat_roll))
        self.get_new_imu_msg = True
    
    ''' Feature detection Filtering '''
    def distance_filter(self, f1,f2):
        x1,y1 = f1.pt
        x2,y2 = f2.pt
        return np.sqrt((x2-x1)**2+(y2-y1)**2)

    def filteringByDistance(self, block_kp, distE=50):
        size = len(block_kp)
        if size < 50 : return block_kp
        else:
            mask = np.arange(1,size+1).astype(np.bool8)
            for i,f1 in enumerate(block_kp):
                if not mask[i]:
                    continue
                else:
                    for j,f2 in enumerate(block_kp):
                        if i==j:
                            continue
                        if self.distance_filter(f1,f2) < distE:
                            mask[j] = False
                        
            block_kp = np.array(block_kp)
            return block_kp[mask]


    def filtering_Block(self, pre_kp, first_frame, roi_block=4):
        h, w = first_frame.shape
        
        block_h = h / (roi_block/2)
        block_w = w / (roi_block/2)
        # --------------
        #   1   |   2  |
        # --------------
        #   3   |   4  |
        # --------------
        block_1, block_2, block_3, block_4 = [], [], [], []
        
        for point in pre_kp:
            point_x, point_y = point.pt
            ### Add conditions to increase the number of blocks.
            if (point_x >= 0 and point_x < block_w) and (point_y >= 0 and point_y < block_h):
                block_1.append(point)

            elif (point_x >= block_w and point_x < block_w*2) and (point_y >= 0 and point_y < block_h):
                block_2.append(point)

            elif (point_x >= 0 and point_x < block_w) and (point_y >= block_h and point_y < block_h*2):
                block_3.append(point)

            else:
                block_4.append(point)

        block_1 = self.filteringByDistance(block_1)
        block_2 = self.filteringByDistance(block_2)
        block_3 = self.filteringByDistance(block_3)
        block_4 = self.filteringByDistance(block_4)

        pre_kp = np.concatenate([block_1, block_2, block_3, block_4])

        return pre_kp

    def featureTracking(self, image_ref, image_cur, ref_kp):
        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize  = (23,23), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))
        cur_kp, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, ref_kp, None, **lk_params) 
        st = st.reshape(st.shape[0])
        pre_kp = ref_kp[st == 1]
        cur_kp = cur_kp[st == 1]

        return pre_kp, cur_kp

    def featureDetection(self, first_frame, second_frame):
        det = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)
        pre_kp = det.detect(first_frame)
        pre_kp = self.filtering_Block(pre_kp, first_frame)

        pre_kp = np.array([x.pt for x in pre_kp], dtype=np.float32)

        pre_kp, cur_kp = self.featureTracking(first_frame, second_frame, pre_kp)

        if self.preproc_ct == 1:
            self.cur_R, self.cur_t = self.getRTMatrix(pre_kp, cur_kp)
            
        return pre_kp

    def getRTMatrix(self, pre_kp, cur_kp):
        # E, mask = cv2.findEssentialMat(cur_kp, pre_kp, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)  
        F, mask = cv2.findFundamentalMat(cur_kp, pre_kp, cv2.FM_RANSAC, param1=0.1, param2=0.999)
        E = np.dot(self.K.T, np.dot(F, self.K))
        _, R, t, mask = cv2.recoverPose(E, cur_kp, pre_kp, self.K)

        return R, t

    def posemsg2ROS(self, est_euler):
        est_pose_msg = Imu()
        est_pose_msg.orientation.x = est_euler[0]
        est_pose_msg.orientation.y = est_euler[1]
        est_pose_msg.orientation.z = est_euler[2]
        est_pose_msg.angular_velocity.x = self.cur_t[0][0]
        est_pose_msg.angular_velocity.y = self.cur_t[1][0]
        est_pose_msg.angular_velocity.z = self.cur_t[2][0]

        return est_pose_msg

    def triangulatePoints(self, R, t, pre_kp, cur_kp):
        """Triangulates the feature correspondence points with
        the camera intrinsic matrix, rotation matrix, and translation vector.
        It creates projection matrices for the triangulation process."""
        # The canonical matrix (set as the origin)
        P0 = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0]])
        P0 = self.K.dot(P0)
        # Rotated and translated using P0 as the reference point
        P1 = np.hstack((R, t))
        P1 = self.K.dot(P1)
        # Reshaped the point correspondence arrays to cv2.triangulatePoints's format
        point1 = pre_kp.reshape(2, -1)
        point2 = cur_kp.reshape(2, -1)

        return cv2.triangulatePoints(P0, P1, point1, point2).reshape(-1, 4)[:, :3]

    def getAbsoluteScale(self): 
        x_prev = self.pre_pose[0]
        y_prev = self.pre_pose[1]
        z_prev = self.pre_pose[2]

        x_curv = self.cur_pose[0]
        y_curv = self.cur_pose[1]
        z_curv = self.cur_pose[2]
 
        return np.sqrt((x_curv - x_prev) * (x_curv - x_prev) + (y_curv - y_prev) * (y_curv - y_prev) + (z_curv - z_prev) * (z_curv - z_prev))

    def getEulerAngle(self, R, t, state, pre_kp, cur_kp):

        absolute_scale = self.getAbsoluteScale()
        # relative_scale = self.getRelativeScale()

        # Accepts only dominant forward motion
        if (t[2] > t[0] and t[2] > t[1]):
            self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
            
            self.cur_R = R.dot(self.cur_R)
  
            self.T_vectors.append(tuple(self.cur_R.dot(self.cur_t)))
            self.R_matrices.append(tuple(self.cur_R))
            self.new_cloud = self.triangulatePoints(self.cur_R, self.cur_t,  pre_kp, cur_kp)
            self.pre_pose = self.cur_pose
            
        if state > 0.25:
            x = math.atan2(self.cur_R[2,1] , self.cur_R[2,2])
            y = math.atan2(self.cur_R[2,0], math.sqrt(self.cur_R[0,0] * self.cur_R[0,0] + self.cur_R[1,0] * self.cur_R[1,0]))
            z = math.atan2(self.cur_R[1,0], self.cur_R[0,0])
            self.pre_est_euler = x, y, z
            return np.array([x, y, z])
      
        else:
            return np.array([self.pre_est_euler[0], self.pre_est_euler[1], self.pre_est_euler[2]])

    def getCameraRtMsg(self, position):
        camera_pose = Float64MultiArray()
        camera_pose.data = position
        return camera_pose

    def main(self):
        try:
            frame_ct = 0

            while not rospy.is_shutdown():
                if self.get_new_img_msg and self.preproc_ct <= 1:
                    if self.preproc_ct == 0 :
                        first_frame = cv2.cvtColor(self.cur_img['img'], cv2.COLOR_BGR2GRAY) 
                        img_mask = np.zeros_like(self.cur_img['img'])

                    else :
                        second_frame = cv2.cvtColor(self.cur_img['img'], cv2.COLOR_BGR2GRAY)
                        pre_kp = self.featureDetection(first_frame, second_frame)
                        self.pre_pose = self.Position
                        last_frame = second_frame    

                    self.preproc_ct += 1
                    self.get_new_img_msg = False

                if self.get_new_img_msg and self.preproc_ct > 1:
                    new_frame = cv2.cvtColor(self.cur_img['img'], cv2.COLOR_BGR2GRAY)     
                    self.cur_pose = self.Position
                    pre_kp, cur_kp = self.featureTracking(last_frame, new_frame, pre_kp)

                    state = np.mean(np.abs(cur_kp - pre_kp)) ## detect vehicle movement
                
                    ## Update feature (At least number of point 6)
                    if  state < 0.25 or pre_kp.shape[0] < 6 or frame_ct % 5 == 0:
                        pre_kp = self.featureDetection(last_frame, new_frame)
                        pre_kp, cur_kp = self.featureTracking(last_frame, new_frame, pre_kp)
                        img_mask = np.zeros_like(self.cur_img['img'])

                    R, t = self.getRTMatrix(pre_kp, cur_kp)
                    est_euler = self.getEulerAngle(R, t, state, pre_kp, cur_kp)

                    ### Publish
                    self.cam_pose.publish(self.getCameraRtMsg(est_euler))  ### Qt_visualize
                    self.est_pose.publish(self.posemsg2ROS(est_euler))  ### Compare Simulator GT

                    ### Draw
                    for i,(new,old) in enumerate(zip(cur_kp, pre_kp)):
                        a,b = new.ravel()
                        c,d = old.ravel()
                        img_mask = cv2.line(img_mask, (int(a),int(b)),(int(c),int(d)), (0,255,0), 2)
                        frame = cv2.circle(self.cur_img['img'],(int(a),int(b)),3,(0,255,0),-1)
                    
                    if self.pub_od.get_num_connections() > 0:
                        msg = None
                        try:
                            msg = self.bridge.cv2_to_imgmsg(cv2.add(frame, img_mask), "bgr8")
                            msg.header = self.cur_img['header']
                        except CvBridgeError as e:
                            print(e)
                        self.pub_od.publish(msg)

                    frame_ct += 1
                    pre_kp = cur_kp
                    last_frame = new_frame
                    self.get_new_img_msg = False
                    self.get_new_imu_msg = False
    
        except rospy.ROSInterruptException:
            rospy.logfatal("{object_detection} is dead.")

if __name__ == "__main__":
    optical = Opticlaflow()
    optical.main()
    rospy.spin()
    