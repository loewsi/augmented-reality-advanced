#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
from renderClass import Renderer
from dt_apriltags import Detector
from image_geometry import PinholeCameraModel

import rospy
import yaml
import sys
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

import rospkg 


"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""

class ARNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ARNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")

        rospack = rospkg.RosPack()
        # Initialize an instance of Renderer giving the model in input.
        self.renderer = Renderer(rospack.get_path('augmented_reality_apriltag') + '/src/models/duckie.obj')
        self.bridge = CvBridge()
        #
        #   Write your code here
        #
        self.homography = self.load_extrinsics()
        self.H = np.array(self.homography).reshape((3, 3))
        self.Hinv = np.linalg.inv(self.H)
        self.at_detector = Detector(searchpath=['apriltags'],
                       families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

        self.sub_compressed_img = rospy.Subscriber(
            "/robot_name/camera_node/image/compressed",
            CompressedImage,
            self.callback,
            queue_size=1
        )

        self.sub_camera_info = rospy.Subscriber(
            "/robot_name/camera_node/camera_info",
            CameraInfo,
            self.cb_camera_info,
            queue_size=1
        )

        self.pub_map_img = rospy.Publisher(
            "~april_tag_duckie/image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
        )


    def callback(self, msg):
        header = msg.header
        cv_img = self.readImage(msg)

        cameraMatrix = np.array(self.camera_model.K)
        camera_params = (cameraMatrix[0, 0], cameraMatrix[1, 1], cameraMatrix[0, 2], cameraMatrix[1, 2])
        gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray_img, True, camera_params, 0.065)
        Ho = np.array(tags[0].homography)
        print("first", Ho)
        Ho_new = self.projection_matrix(cameraMatrix, Ho)
        print("second", Ho_new)
        img_out = self.renderer.render(cv_img, Ho_new)
        msg_out = self.bridge.cv2_to_compressed_imgmsg(img_out, dst_format='jpeg')
        msg_out.header = header
        self.pub_map_img.publish(msg_out)




    def cb_camera_info(self, msg):
        # unsubscribe from camera_info
        self.loginfo('Camera info message received. Unsubscribing from camera_info topic.')
        # noinspection PyBroadException
        try:
            self.sub_camera_info.shutdown()
        except BaseException:
            self.loginfo('BaseException')
            pass
        # ---
        self.H_camera, self.W_camera = msg.height, msg.width
        # create new camera info
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)



    
    def projection_matrix(self, intrinsic, homography):
        """
            Write here the compuatation for the projection matrix, namely the matrix
            that maps the camera reference frame to the AprilTag reference frame.
        """

        #
        # Write your code here
        #

        kinv = np.linalg.inv(intrinsic)
        rt = np.dot(kinv, homography)


        r1 = rt[:, 0]
        r2 = rt[:, 1]
        t = rt[:, 2]

        r1_norm = r1
        print(rt)
        r2_new = r2 - np.dot(r1_norm, r2) * r1_norm
        r2_norm = r2_new / np.linalg.norm(r2_new) * np.linalg.norm(r1)
        r3_new = np.cross(r1_norm, r2_norm)
        r3_norm = r3_new / np.linalg.norm(r3_new) * np.linalg.norm(r1)


        matrix = np.zeros((3, 4))
        matrix[:, 0] = r1_norm
        matrix[:, 1] = r2_norm
        matrix[:, 2] = r3_norm
        matrix[:, 3] = t

        homo = np.dot(intrinsic, matrix)
        return homo


    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg_image)
            return cv_image
        except CvBridgeError as e:
            self.log(e)
            return []

    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


    def onShutdown(self):
        super(ARNode, self).onShutdown()

    def load_extrinsics(self):
        """
        Loads the homography matrix from the extrinsic calibration file.
        Returns:
            :obj:`numpy array`: the loaded homography matrix
        """
        # load intrinsic calibration
        cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        cali_file = cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(cali_file):
            self.log("Can't find calibration file: %s.\n Using default calibration instead."
                     % cali_file, 'warn')
            cali_file = (cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(cali_file):
            msg = 'Found no calibration file ... aborting'
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        try:
            with open(cali_file, 'r') as stream:
                calib_data = yaml.load(stream)
        except yaml.YAMLError:
            msg = 'Error in parsing calibration file %s ... aborting' % cali_file
            self.log(msg, 'err')
            rospy.signal_shutdown(msg)

        return calib_data['homography']



if __name__ == '__main__':
    # Initialize the node
    camera_node = ARNode(node_name='augmented_reality_apriltag_node')
    # Keep it spinning to keep the node alive
    rospy.spin()