#!usr/bin/env python3.7

# System imports
import sys
import threading

import cv2
import numpy as np
from keras.models import load_model

# ROS imports
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ros_joint_segmentation.msg import HandlerPrediction, FrontPrediction, JointPrediction

# ROS package specific imports
from segmentation_models import get_preprocessing
from segmentation_models.metrics import FScore
from segmentation_models.losses import dice_loss
from postprocessing_helpers.line_operations import find_vertices_of_prediction_joints, get_middle_point_of_lines,\
    get_general_line_coeffs, get_closest_lines_indexes


class JointSegmentator:
    DETECTION_THRESHOLD = 0.5
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 480
    INPUT_CHANNELS = 3
    GRAYSCALE_CHANNEL = 0
    HANDLER_CHANNEL = 1
    ROT_FRONT_CHANNEL = 2
    CANNY_THRESHOLD1 = 47
    CANNY_THRESHOLD2 = 255
    HOUGH_THRESHOLD = 150
    HOUGH_MIN_LINE_LENGTH = 30
    HOUGH_MAX_LINE_GAP = 18
    ROT_JOINT_90ANGLE_TILT = 10

    def __init__(self, rgb_image_topic):
        self.backbone = "efficientnetb0"

        # Topic name of 640x480 image to subscribe
        self.rgb_image_topic = rgb_image_topic

        # Topic name where prediction will be published
        joint_prediction_topic = rospy.get_param("joint_prediction_topic", default="joint_prediction")
        self.prediction_pub = rospy.Publisher(joint_prediction_topic, JointPrediction, queue_size=1)

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
        self.grayscale_image_cv2 = None
        self.rgb_image = None
        self.header = None

        # handler mask
        handler_sub = rospy.Subscriber("/handler_prediction",
                                       data_class=HandlerPrediction,
                                       callback=self.handler_mask_callback,
                                       queue_size=1, buff_size=2 ** 24)
        self.handler_mask = None
        self.handler_data: HandlerPrediction = None

        # rot_front mask
        rot_front_sub = rospy.Subscriber("/front_prediction",
                                         data_class=FrontPrediction,
                                         callback=self.rot_front_mask_callback,
                                         queue_size=1, buff_size=2 ** 24)
        self.rot_front_mask = None
        self.rot_front_data: FrontPrediction = None

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
            self.header = data.header
            self.rgb_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            self.grayscale_image_cv2 = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
            self.grayscale_image = self.grayscale_image_cv2.astype(np.float32) / 255

            if self.are_all_inputs_ready():
                self.delete_rot_front_without_handler()
                self.run_inference()

    def handler_mask_callback(self, data: HandlerPrediction):
        if self.handler_data is None:
            self.handler_data = data

            if self.are_all_inputs_ready():
                self.delete_rot_front_without_handler()
                self.run_inference()

    def rot_front_mask_callback(self, data: FrontPrediction):
        if self.rot_front_data is None:
            self.rot_front_data = data

            if self.are_all_inputs_ready():
                self.delete_rot_front_without_handler()
                self.run_inference()

        # TODO: Or if there's no handler give some probability then?

    def delete_rot_front_without_handler(self):
        """
        Deletes rotational fronts that has no detected handlers in it.

        :return:
        """
        rot_front_data = self.rot_front_data
        prediction_indexes_to_delete = []

        for prediction_index, rot_front_box in enumerate(self.rot_front_data.boxes):
            rot_front_has_handler = False

            for handler_box in self.handler_data.boxes:
                handler_x = (handler_box.x_offset + handler_box.x_offset + handler_box.width) / 2
                handler_y = (handler_box.y_offset + handler_box.y_offset + handler_box.height) / 2
                # Check if handler box is in rotation front
                if ((handler_x >= rot_front_box.x_offset)
                        and handler_x <= (rot_front_box.x_offset + rot_front_box.width)
                        and (handler_y >= rot_front_box.y_offset)
                        and handler_y <= (rot_front_box.y_offset + rot_front_box.height)):
                    rot_front_has_handler = True
                    break

            if not rot_front_has_handler:
                # If rot front has no handler skip this rotational front prediction
                prediction_indexes_to_delete.append(prediction_index)

        for prediction_index in sorted(prediction_indexes_to_delete, reverse=True):
            # Delete prediction with no handler
            rot_front_data.boxes.pop(prediction_index)
            rot_front_data.class_names.pop(prediction_index)
            rot_front_data.masks.pop(prediction_index)

            try:
                rot_front_data.class_ids.pop(prediction_index)
            except AttributeError as e:
                rot_front_data.class_ids = list(rot_front_data.class_ids)
                rot_front_data.class_ids.pop(prediction_index)
            try:
                rot_front_data.scores.pop(prediction_index)
            except AttributeError as e:
                rot_front_data.scores = list(rot_front_data.scores)
                rot_front_data.scores.pop(prediction_index)

        self.rot_front_data = rot_front_data

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
                (self.handler_data is not None) and \
                (self.rot_front_data is not None):
            return True
        else:
            return False

    def reset_all_inputs(self):
        self.grayscale_image = None
        self.grayscale_image_cv2 = None
        self.handler_data = None
        self.handler_mask = None
        self.rot_front_data = None
        self.rot_front_mask = None
        self.header = None

    def prepare_input_data(self):
        """
        Prepares input data for model. Channels order based on model training:
        1) Grayscale image
        2) Handler mask
        3) Rotational front mask

        :return:
        """
        rot_front_mask = self.merge_to_single_mask(self.rot_front_data, class_to_merge="rot_front")
        self.rot_front_mask = rot_front_mask.astype(np.float32) / 255

        handler_mask = self.merge_to_single_mask(self.handler_data)
        self.handler_mask = handler_mask.astype(np.float32) / 255

        input_data = np.empty((self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.INPUT_CHANNELS), dtype=np.float32)

        input_data[:, :, self.GRAYSCALE_CHANNEL] = self.grayscale_image
        input_data[:, :, self.HANDLER_CHANNEL] = self.handler_mask
        input_data[:, :, self.ROT_FRONT_CHANNEL] = self.rot_front_mask

        return input_data

    def find_lines_on_grayscale_image(self):
        """
        Finds vertical lines on input grayscale image.

        :return: List of tuples containing top and bottom vertices of vertical lines. Tuple has vertices in following
         order x1, y1, x2, y2
        """
        edges = cv2.Canny(self.grayscale_image_cv2, threshold1=self.CANNY_THRESHOLD1, threshold2=self.CANNY_THRESHOLD2)
        linesP = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=self.HOUGH_THRESHOLD,
                                 minLineLength=self.HOUGH_MIN_LINE_LENGTH, maxLineGap=self.HOUGH_MAX_LINE_GAP)

        lines_vertices_list = []  # List of vertices for each line
        if linesP is not None:
            for i in range(0, len(linesP)):
                x1, y1, x2, y2 = linesP[i][0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                if (np.abs(angle) > (90 + self.ROT_JOINT_90ANGLE_TILT)) or\
                        (np.abs(angle) < (90 - self.ROT_JOINT_90ANGLE_TILT)):
                    continue  # Skip lines that are not vertical
                lines_vertices_list.append((x1, y1, x2, y2))
        return lines_vertices_list

    def post_process_prediction(self, img):
        """
        Post process the prediction to get better predictions

        :param img:
        :return:
        """
        prediction_mask = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0)  # Blur the prediction mask
        # Find edges on the image
        prediction_edges = cv2.Canny(prediction_mask, threshold1=self.CANNY_THRESHOLD1,
                                     threshold2=self.CANNY_THRESHOLD2)
        # Find vertices of prediction joints based on canny edge
        prediction_vertices = find_vertices_of_prediction_joints(prediction_edges)
        # Find middle point of predicted joint
        prediction_middle_points = get_middle_point_of_lines(prediction_vertices)

        # Find vertices on grayscale img
        grayscale_vertices = self.find_lines_on_grayscale_image()
        # Find general line coeffs of grayscale lines
        grayscale_line_coeffs = []
        for vertice in grayscale_vertices:
            coeffs_vertices = get_general_line_coeffs(vertice[0], vertice[1], vertice[2], vertice[3])
            grayscale_line_coeffs.append(coeffs_vertices)

        # For each middle point from prediction get distance to each line and find index of closest one
        closest_line_indexes = get_closest_lines_indexes(prediction_middle_points, grayscale_line_coeffs)

        if self.should_publish_visualization:
            for line_index in closest_line_indexes:
                vertices_to_draw = grayscale_vertices[line_index]
                cv2.line(self.rgb_image, (vertices_to_draw[0], vertices_to_draw[1]),
                         (vertices_to_draw[2], vertices_to_draw[3]),
                         (0, 255, 255), 3)

            image_msg = self.cv_bridge.cv2_to_imgmsg(self.rgb_image, encoding="bgr8")
            self.vis_pub.publish(image_msg)

        self.publish_prediction(closest_line_indexes, grayscale_vertices)

    def publish_prediction(self, closest_line_indexes, vertices):
        prediction_msg = JointPrediction()
        prediction_msg.header = self.header

        for line_index in closest_line_indexes:
            x1, y1, x2, y2 = vertices[line_index]
            prediction_msg.x1.append(x1)
            prediction_msg.y1.append(y1)
            prediction_msg.x2.append(x2)
            prediction_msg.y2.append(y2)

        self.prediction_pub.publish(prediction_msg)

    def run_inference(self):
        if self.model is not None:
            input_data = self.prepare_input_data()
            # Predict
            prediction = self.model.predict(input_data[np.newaxis, :])[0]
            # Convert prediction mask from 0-1 scale to 0-255
            prediction = np.where(prediction > self.DETECTION_THRESHOLD, np.uint8(255), np.uint8(0))

            # Remove noise
            prediction = cv2.morphologyEx(prediction, cv2.MORPH_OPEN, (3, 3))

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
