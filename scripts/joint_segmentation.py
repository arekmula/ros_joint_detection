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
    MINIMUM_JOINT_PREDICTION_HEIGHT = 100

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
        self.rgb_image = None

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
            self.rgb_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            self.grayscale_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)  # TODO: Check the conversion
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

        # TODO: If there's no handler in the front mask skip it in merging, so the network doesn't generate false
        #  positives
        # TODO: Or if there's no handler give some probability then?

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

    def method_no_1(self, contours: np.ndarray):
        """
        Using found contours to find the corners of predicted joint

        :param contours:
        :return:
        """

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if h < self.MINIMUM_JOINT_PREDICTION_HEIGHT:
                continue

            pts = contour.reshape(contour.shape[0], 2)
            # top left point has smallest sum and bottom right has smallest sum
            s = np.sum(pts, axis=1)
            top_left = pts[np.argmin(s)]
            bottom_right = pts[np.argmax(s)]
            # top right has minimum difference and bottom left has maximum difference
            diff = np.diff(pts, axis=1)
            top_right = pts[np.argmin(diff)]
            bottom_left = pts[np.argmax(diff)]

            x_top = int((top_left[0] + top_right[0]) / 2)
            y_top = int((top_left[1] + top_right[1]) / 2)

            x_bot = int((bottom_left[0] + bottom_right[0]) / 2)
            y_bot = int((bottom_left[1] + bottom_right[1]) / 2)

            cv2.line(self.rgb_image, (x_top, y_top), (x_bot, y_bot), (255, 255, 0), 2)

    def method_no_2(self, contours: np.ndarray):
        """
        Finds joint based on bounding box of contour

        :param contours:
        :return:
        """
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if h < self.MINIMUM_JOINT_PREDICTION_HEIGHT:
                continue

            x = int(x + w/2)
            cv2.line(self.rgb_image, (x, y), (x, y+h), (0, 255, 255), 2)

    def post_process_prediction(self, img):
        """
        Post process the prediction to get better predictions

        :param img:
        :return:
        """
        img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)  # Blur the image
        edges = cv2.Canny(img, 100, 200)  # Find edges on the image
        # Find external contours on the image and gather or corners
        contours, hierarchy = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

        self.method_no_1(contours)
        self.method_no_2(contours)

        if self.should_publish_visualization:
            image_msg = self.cv_bridge.cv2_to_imgmsg(self.rgb_image, encoding="bgr8")
            self.vis_pub.publish(image_msg)

    def run_inference(self):
        if self.model is not None:
            input_data = self.prepare_input_data()
            # Predict
            prediction = self.model.predict(input_data[np.newaxis, :])[0]
            # Convert prediction mask from 0-1 scale to 0-255
            prediction = np.where(prediction > self.DETECTION_THRESHOLD, np.uint8(255), np.uint8(0))

            # Remove noise
            prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, (3, 3))
            # TODO: Post process the prediction with HoughLines
            # TODO: Find lines on original grayscale image and like compare prediction with those lines
            self.post_process_prediction(prediction)

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
