import torch
from tool.torch_utils import *
from tool.utils import *
from models.py2_Yolov4_model import Yolov4

import cv2
import time
import copy
import argparse
import math

import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from std_msgs.msg import Float64MultiArray 
from sensor_msgs.msg import Imu
from cv_bridge import CvBridge, CvBridgeError

from API.tracker import Tracker
from API.drawer import Drawer
from calibration.calib import Calibration

class DistCorrection:
    '''Correct the ground with a pose from the IMU sensor.'''
    def __init__(self, calibration):
        self.calibration = calibration
        self.Imu_Data = []
        self.roll = 0
        self.pitch = 0
        # self.yaw = 0
        self.init_translation_z = 1.6152
        self.translation_Z = 0
        self.get_new_img_msg = False
        
    def SIMUcallback(self, msg):
        self.Imu_Data = -msg.orientation.x, -msg.orientation.y, msg.orientation.z, msg.orientation.w
        self.roll = math.degrees(self.Imu_Data[0])
        self.pitch = math.degrees(self.Imu_Data[1])
        # self.yaw = math.degrees(self.Imu_Data[2])
        self.translation_Z = msg.angular_velocity.z # Temporary variable.
        self.get_new_img_msg = True

    def getRotationMatrix(self):
        if self.get_new_img_msg:
            ''' Real-World = Quaternion to Rotation '''
            ''' Simulator = Euler to Rotation '''
            ## R_z
            mat_yaw = np.array([[math.cos(self.Imu_Data[2]), -math.sin(self.Imu_Data[2]), 0],
                                [math.sin(self.Imu_Data[2]), math.cos(self.Imu_Data[2]), 0],
                                [0, 0, 1]])
            ## R_y
            mat_pitch = np.array([[math.cos(self.Imu_Data[1]),0,math.sin(self.Imu_Data[1])],
                                [0,1,0],
                                [-math.sin(self.Imu_Data[1]),0,math.cos(self.Imu_Data[1])]])
            ## R_x
            mat_roll = np.array([[1, 0, 0],
                                [0, math.cos(self.Imu_Data[0]), -math.sin(self.Imu_Data[0])],
                                [0, math.sin(self.Imu_Data[0]), math.cos(self.Imu_Data[0])]])

            new_R = np.dot(mat_yaw, np.dot(mat_pitch, mat_roll))
            new_M2, ground_plane = self.getNewGroundPlane(new_R)

            return new_M2, ground_plane


    def getNewGroundPlane(self, new_R):
        pts_3d = np.array([self.calibration.pt1_2, self.calibration.pt2_2, self.calibration.pt3_2, self.calibration.pt4_2])

        for i, pt_3d in enumerate(pts_3d):
            pts_3d[i][2] = pts_3d[i][2] - (self.translation_Z - self.init_translation_z)
            pts_3d[i] = np.dot(new_R, pt_3d.transpose())
        
        new_pts_2d = self.calibration.project_3d_to_2d(pts_3d.transpose())
    
        new_src_pt = np.float32([[int(np.round(new_pts_2d[0][0])), int(np.round(new_pts_2d[1][0]))],
                                [int(np.round(new_pts_2d[0][1])), int(np.round(new_pts_2d[1][1]))],
                                [int(np.round(new_pts_2d[0][2])), int(np.round(new_pts_2d[1][2]))],
                                [int(np.round(new_pts_2d[0][3])), int(np.round(new_pts_2d[1][3]))]])

        new_M2 = cv2.getPerspectiveTransform(new_src_pt, self.calibration.dst_pt)
        draw_plane = np.array(new_src_pt, np.int32)

        return new_M2, draw_plane

class Detection:
    def __init__(self, args, Correction, calibration, image_shape=(1280, 806)):
        self.img_shape = image_shape
        self.args = args
        self.calibration = calibration
        self.tracker = Tracker((self.img_shape[1], self.img_shape[0]), min_hits=1, num_classes=args.n_classes, interval=args.interval) # Initialize tracker
        self.drawer = Drawer(args.namesfile)  # Inidialize drawer class
        self.bridge = CvBridge()
        
        self.model = self.LoadModel()
        self.cur_img = {'img':None, 'header':None}

        rospy.init_node('esti_dist')
        rospy.Subscriber('/vds_node_localhost_2211/image_raw/compressed', CompressedImage, self.IMGcallback)
        rospy.Subscriber('/imu_sensor/pose', Imu, Correction.SIMUcallback, queue_size=1)

        self.pub_od = rospy.Publisher('/od_result', Image, queue_size=1)
        self.pub_dist = rospy.Publisher("/distance_arr", Float64MultiArray, queue_size=1)

        self.get_new_img_msg = False

    def IMGcallback(self, msg):
        np_arr = np.fromstring(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if self.img_shape != (img.shape[1], img.shape[0]): img = cv2.resize(img, self.img_shape)
        self.cur_img['img'] = img # self.calibration.undistort(img)
        self.cur_img['header'] = msg.header
        self.get_new_img_msg = True

    def LoadModel(self):
        model = Yolov4(n_classes=self.args.n_classes, inference=True)
        pretrained_dict = model.load_state_dict(torch.load(self.args.weightfile, map_location="cuda:%s"%(self.args.gpu_num)))
        print("pre_weight : %s load done!!!" %(self.args.weightfile.split('/')[-1]))
        torch.cuda.set_device(self.args.gpu_num)
        model.cuda()
        model.eval()
        torch.backends.cudnn.benchmark = True
        print ('Current cuda device : %s'%(torch.cuda.current_device()))
        return model

    def get_dist_array_msg(self, dist_arr):
        ''' For the comparison plot.'''
        dist_array_msg = Float64MultiArray()
        for dist in dist_arr:
            dist_array_msg.data.append(dist[0])

        return dist_array_msg

    def main(self):
        try:
            moving_tra, moving_det = 0., 0.
            frame_ind = 0
            dist_arr =[]
            new_groud_detection_flag = False

            ground_plane = np.array(calibration.src_pt, np.int32)
            inti_ground_plane = ground_plane

            while not rospy.is_shutdown():
                if self.get_new_img_msg:
                    start = time.time()
                    dets_arr, labels_arr, is_dect = None, None, None
                    if np.mod(frame_ind, self.args.interval) == 0:
                        orig_im = copy.copy(self.cur_img['img'])
                        orig_im = cv2.resize(orig_im, (self.img_shape))

                        img = cv2.resize(self.cur_img['img'], (320, 320))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        bbox = do_detect(self.model, img, 0.7, 0.4)[0]

                        if len(bbox) > 0: 
                            bbox = np.vstack(bbox)
                            output = copy.copy(bbox)

                            data_list = []
                            pts_2d = []
                            
                            output[:,0] = (bbox[:,0] - bbox[:,2] / 2.0) * self.img_shape[0]
                            output[:,1] = (bbox[:,1] - bbox[:,3] / 2.0) * self.img_shape[1]
                            output[:,2] = (bbox[:,0] + bbox[:,2] / 2.0) * self.img_shape[0]
                            output[:,3] = (bbox[:,1] + bbox[:,3] / 2.0) * self.img_shape[1]
            
                            for box in output[:,:4]:   
                                center_pt_x = (box[0] + ((box[2] - box[0]) / 2))
                                center_pt_y = box[3] + ((box[3] - box[1])*self.args.pixel_offset) #pixel_offset
                                orig_im = cv2.line(orig_im, (int(center_pt_x), int(center_pt_y)),(int(center_pt_x), int(center_pt_y)), (255,255,0), 10)
                                pts_2d.append(np.float32([[center_pt_x], [center_pt_y], [1.0]]))

                            if  abs(Correction.pitch) > 0.29 or abs(Correction.roll) > 0.02: 
                                new_groud_detection_flag = True
                                mew_M2, ground_plane = Correction.getRotationMatrix()
                                
                                new_pts_3d = np.dot(mew_M2, pts_2d)
                                new_pts_3d = np.squeeze(new_pts_3d, axis=2)
                                
                                new_pts_3d = new_pts_3d / new_pts_3d[2,:]
                                pts_3d = new_pts_3d.T[:,:2]
                
                            else:
                                pts_3d = np.dot(self.calibration.M2, pts_2d)
                                pts_3d = np.squeeze(pts_3d, axis=2)
                                pts_3d = pts_3d / pts_3d[2,:]
                                pts_3d = pts_3d.T[:,:2]
                                ground_plane = np.array(self.calibration.src_pt, np.int32)

                            dist_arr = []
                            for pt_3d in pts_3d:
                                dist = np.sqrt(pt_3d[0]**2 + pt_3d[1]**2)
                                dist_arr.append([round(dist, 2)])
                            
                            dets_arr, labels_arr = output[:,0:4], output[:,-1].astype(int)
        
                        else:
                            dets_arr, labels_arr = np.array([]), np.array([])

                        is_dect = True
                        
                    elif np.mod(frame_ind, interval) != 0:
                        dets_arr, labels_arr = np.array([]), np.array([])
                        is_dect = False

                    pt_det = (time.time() - start)
                    # tracker_arr = tracker.update(dets_arr, labels_arr, is_dect=is_dect)
                    # pt_tra = (time.time() - start)
                    
                    if frame_ind != 0:
                        #moving_tra = (frame_ind / float(frame_ind + 1) * moving_tra) + (1. / float(frame_ind + 1) * pt_tra)
                        moving_det = (frame_ind / float(frame_ind + 1) * moving_det) + (1. / float(frame_ind + 1) * pt_det)

                    dist_array_msg = self.get_dist_array_msg(dist_arr)
                    self.pub_dist.publish(dist_array_msg)
                    
                    #show_frame = drawer.draw(orig_im, tracker_arr, labels_arr, dist_arr, (1. / (moving_tra + 1e-8)), is_tracker=True)
                    show_frame = self.drawer.draw(orig_im, dets_arr, labels_arr, dist_arr, (1. / (moving_det + 1e-8)), is_tracker=False)
                    
                    # ground plane
                    if new_groud_detection_flag:
                        show_frame = cv2.polylines(show_frame, [ground_plane], True, (0,0,255), thickness=2)
                        new_groud_detection_flag = False
                    show_frame = cv2.polylines(show_frame, [inti_ground_plane], True, (0,255,0), thickness=2)
                                            
                    if self.pub_od.get_num_connections() > 0:
                        msg = None
                        try:
                            msg = self.bridge.cv2_to_imgmsg(show_frame, "bgr8")
                            msg.header = self.cur_img['header']
                        except CvBridgeError as e:
                            print(e)
                        self.pub_od.publish(msg)

                    frame_ind += 1
                    get_new_img_msg = False

            rospy.spin()
        except rospy.ROSInterruptException:
            rospy.logfatal("{object_detection} is dead.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters that need to be changed')
    parser.add_argument('--weightfile', default="./weight/yolov4_SM.pth")
    parser.add_argument('--n_classes', default=80, help="Number of classes")
    parser.add_argument('--namesfile', default="data/2021_coco.names", help="Label name of classes")
    parser.add_argument('--gpu_num', default=2, help="Use number gpu")
    parser.add_argument('--interval', default=1, help="Tracking interval")
    parser.add_argument('--pixel_offset', default=0.1, help="Bounding Box Pixel offset rate")

    args = parser.parse_args()

    calibration = Calibration('calibration/SM_camera.txt', 'calibration/SM_camera_lidar.txt')

    Correction = DistCorrection(calibration)
    Detection = Detection(args, Correction, calibration)
    Detection.main()
    
    

    