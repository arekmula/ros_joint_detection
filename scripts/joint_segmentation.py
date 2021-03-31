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
from ros_joint_segmentation.msg import HandlerPrediction

# ROS package specific imports
from segmentation_models import get_preprocessing
from segmentation_models.metrics import FScore
from segmentation_models.losses import dice_loss


class JointSegmentator:
    DETECTION_THRESHOLD = 0.5

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

            if self.are_all_inputs_ready():
                self.run_inference()

    def handler_mask_callback(self, data: HandlerPrediction):
        print("received handler")
        if self.handler_mask is None:
            handler_mask = np.zeros((480, 640), dtype=np.uint8)

            for mask in data.masks:
                mask_cv = self.cv_bridge.imgmsg_to_cv2(mask, "mono8")
                handler_mask += mask_cv

            print(np.max(handler_mask))
            handler_mask[handler_mask >= 255] = 255  # prune the value to 255 if mask was greater than 255
            # Publish visualization image
            if self.should_publish_visualization:
                image_msg = self.cv_bridge.cv2_to_imgmsg(handler_mask, encoding="mono8")
                self.vis_pub.publish(image_msg)

            self.handler_mask = handler_mask.astype(np.float32) / 255

            if self.are_all_inputs_ready():
                self.run_inference()

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

    def run_inference(self):

        # Predict
        prediction = self.model.predict(tf.expand_dims(None, axis=0))[0]
        # Convert image from 0-1 scale to 0-255
        prediction = np.where(prediction > self.DETECTION_THRESHOLD, np.uint8(255), np.uint8(0))

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
