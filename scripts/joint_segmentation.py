#!usr/bin/env python3.7

# System imports
import sys
import threading

import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# ROS imports
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String, Header
from ros_joint_segmentation.msg import HandlerPrediction, FrontPrediction

# ROS package specific imports
from segmentation_models import get_preprocessing
from segmentation_models.metrics import FScore
from segmentation_models.losses import dice_loss


class JointSegmentator:
    DETECTION_THRESHOLD = 0.5
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    INPUT_CHANNELS = 3
    GRAYSCALE_CHANNEL = 0
    HANDLER_CHANNEL = 1
    ROT_FRONT_CHANNEL = 2

    def __init__(self, rgb_image_topic):
        self.backbone = "efficientnetb0"

        # Topic name of 640x480 image to subscribe
        self.rgb_image_topic = rgb_image_topic

        # Topic name where prediction will be published
        joint_prediction_topic = rospy.get_param("joint_prediction_topic", default="joint_prediction")
        # self.prediction_pub = rospy.Publisher(joint_prediction_topic, None, queue_size=1)

        self.cv_bridge = CvBridge()

        # Load trained weights
        model_path = rospy.get_param("joint_seg_model", None)
        try:
            self.model = load_model(model_path, custom_objects={"f1-score": FScore, "dice_loss": dice_loss})
            print(self.model.summary())
        except TypeError as e:
            raise BaseException("Path to model not given! Set joint_seg_model param")

        # Preprocess image function according to backbone specific
        self.preprocess_input = get_preprocessing(self.backbone)

        # rgb and grayscale image
        rgb_sub = rospy.Subscriber(self.rgb_image_topic, data_class=Image, callback=self.rgb_image_callback,
                                   queue_size=1, buff_size=2 ** 24)
        self.grayscale_image = None

        # handler mask
        handler_sub = rospy.Subscriber("/handler_prediction",
                                       data_class=HandlerPrediction,
                                       callback=self.handler_mask_callback,
                                       queue_size=1, buff_size=2 ** 24)
        self.handler_mask = None

        # rot_front mask
        rot_front_sub = rospy.Subscriber("/front_prediction",
                                         data_class=FrontPrediction,
                                         callback=self.rot_front_mask_callback,
                                         queue_size=1, buff_size=2 ** 24)
        self.rot_front_mask = None

        # Last input message and message lock
        self.last_msg = None
        self.msg_lock = threading.Lock()

        # Should publish visualization image
        self.should_publish_visualization = rospy.get_param("visualize_joint_prediction", True)
        if self.should_publish_visualization:
            self.vis_pub = rospy.Publisher("joint_visualization", Image, queue_size=1)

    def rgb_image_callback(self, data):
        rospy.logdebug("Get input image")

        if self.grayscale_image is None:
            rgb_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            self.grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)  # TODO: Check the conversion
            self.grayscale_image = self.grayscale_image.astype(np.float32) / 255

            if self.are_all_inputs_ready():
                self.run_inference()

    def handler_mask_callback(self, data):
        if self.handler_mask is None:
            handler_mask = self.merge_to_single_mask(data)
            self.handler_mask = handler_mask.astype(np.float32) / 255

            if self.are_all_inputs_ready():
                self.run_inference()

    def rot_front_mask_callback(self, data):
        if self.rot_front_mask is None:
            rot_front_mask = self.merge_to_single_mask(data, class_to_merge="rot_front")
            self.rot_front_mask = rot_front_mask.astype(np.float32) / 255

            if self.are_all_inputs_ready():
                self.run_inference()

    def merge_to_single_mask(self, data: HandlerPrediction, class_to_merge: str = None):
        """
        Merge list of masks to one common mask

        :param class_to_merge: class to merge. For example create mask only from rotational fronts
        :param data: input class containing list of masks and list of class_names
        :return: one common mask
        """

        single_mask = np.zeros((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.uint8)

        for mask, class_name in zip(data.masks, data.class_names):
            if class_to_merge is not None:
                if class_name != class_to_merge:
                    continue
            current_mask_cv2 = self.cv_bridge.imgmsg_to_cv2(mask, "mono8")
            single_mask += current_mask_cv2

        single_mask[single_mask >= 255] = 255  # prune the value to 255 if mask was greater than 255
        return single_mask

    def are_all_inputs_ready(self):
        if (self.grayscale_image is not None) and \
                (self.handler_mask is not None) and \
                (self.rot_front_mask is not None):
            return True
        else:
            return False

    def reset_all_inputs(self):
        self.handler_mask = None
        self.grayscale_image = None
        self.rot_front_mask = None

    def prepare_input_data(self):
        """
        Prepares input data for model. Channels order based on model training:
        1) Grayscale image
        2) Handler mask
        3) Rotational front mask

        :return:
        """
        input_data = np.empty((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.INPUT_CHANNELS), dtype=np.float32)

        input_data[:, :, self.GRAYSCALE_CHANNEL] = self.grayscale_image
        input_data[:, :, self.HANDLER_CHANNEL] = self.handler_mask
        input_data[:, :, self.ROT_FRONT_CHANNEL] = self.rot_front_mask

        return input_data

    def run_inference(self):

        if self.model is not None:
            input_data = self.prepare_input_data()
            # Predict
            prediction = self.model.predict(input_data[np.newaxis, :])[0]
            # Convert prediction mask from 0-1 scale to 0-255
            prediction = np.where(prediction > self.DETECTION_THRESHOLD, np.uint8(255), np.uint8(0))

            if self.should_publish_visualization:
                image_msg = self.cv_bridge.cv2_to_imgmsg(prediction, encoding="mono8")
                self.vis_pub.publish(image_msg)

        self.reset_all_inputs()


def main(args):
    rospy.init_node("joint_segmentation")
    if rospy.has_param("rgb_image_topic"):
        rgb_image_topic = rospy.get_param("rgb_image_topic")
        print(rgb_image_topic)

    joint_segmentator = JointSegmentator(rgb_image_topic)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
