#!usr/bin/env python3.7

# System imports
import sys
import threading
from operator import itemgetter

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
    get_general_line_coeffs, get_closest_lines_indexes, get_point_on_opposite_edge_of_handler, get_point_line_distance


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

        self.rot_front_bboxes = []  # List of bounding boxes of rotational fronts
        self.rot_front_info_list = []  # List of dictionaries corresponding to above bounding boxes lists. Each
        # dictionary have info about index of handler in rotational front and index of rotational front prediction

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
        Saves corresponding handler index for each rotational front

        :return:
        """
        rot_front_data = self.rot_front_data
        prediction_indexes_to_delete = []

        for prediction_index, rot_front_box in enumerate(self.rot_front_data.boxes):

            prediction_dictionary = {}
            rot_front_has_handler = False

            for handler_idx, handler_box in enumerate(self.handler_data.boxes):
                handler_x = (handler_box.x_offset + handler_box.x_offset + handler_box.width) / 2
                handler_y = (handler_box.y_offset + handler_box.y_offset + handler_box.height) / 2
                # Check if handler box is in rotation front
                if ((handler_x >= rot_front_box.x_offset)
                        and handler_x <= (rot_front_box.x_offset + rot_front_box.width)
                        and (handler_y >= rot_front_box.y_offset)
                        and handler_y <= (rot_front_box.y_offset + rot_front_box.height)):
                    rot_front_has_handler = True
                    # Save index of handler corresponding to the rotational front
                    prediction_dictionary["handler_index"] = handler_idx
                    break

            if not rot_front_has_handler:
                # If rot front has no handler skip this rotational front prediction
                prediction_indexes_to_delete.append(prediction_index)
            else:
                if rot_front_data.class_names[prediction_index] == "rot_front":
                    # Save prediction index of rotational front
                    prediction_dictionary["original_prediction_index"] = prediction_index
                    self.rot_front_info_list.append(prediction_dictionary)

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

    def merge_to_single_mask(self, data: HandlerPrediction,
                             class_to_merge: str = None,
                             class_to_distuinguish: str = "rot_front"):
        """
        Merge list of masks to one common mask

        :param class_to_distuinguish: class for which you want to distuinguish bboxes
        :param class_to_merge: class to merge. For example create mask only from rotational fronts
        :param data: input class containing list of masks and list of class_names
        :return: one common mask
        """

        single_mask = np.zeros((self.IMAGE_HEIGHT, self.IMAGE_WIDTH), dtype=np.uint8)

        for mask, class_name, box in zip(data.masks, data.class_names, data.boxes):
            if class_to_merge is not None:
                if class_name != class_to_merge:
                    continue
            current_mask_cv2 = self.cv_bridge.imgmsg_to_cv2(mask, "mono8")
            # Save rotational bounding boxes for later usage
            if class_name == class_to_distuinguish:
                self.rot_front_bboxes.append(box)

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
        self.rot_front_bboxes = []
        self.rot_front_info_list = []

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

        # For each middle point from prediction get distance to each line from lines on found on grayscale image
        # and find index of closest one
        closest_line_indexes = get_closest_lines_indexes(prediction_middle_points, grayscale_line_coeffs)

        # Get rid of lines found on grayscale image that are not close to predicted ones
        grayscale_line_coeffs_closest = [grayscale_line_coeffs[i]["general_coeffs"] for i in closest_line_indexes]
        grayscale_vertices_closest = [grayscale_vertices[i] for i in closest_line_indexes]

        # Attach each left joint to rotational front
        joints_coeffs, joints_vertices, joint_front_indexes = self.attach_joint_to_front(grayscale_line_coeffs_closest,
                                                                                         grayscale_vertices_closest)

        # Publish visualization of joints
        if self.should_publish_visualization:
            for vertices_to_draw in joints_vertices:
                cv2.line(self.rgb_image, (vertices_to_draw[0], vertices_to_draw[1]),
                         (vertices_to_draw[2], vertices_to_draw[3]),
                         (0, 255, 255), 3)
            image_msg = self.cv_bridge.cv2_to_imgmsg(self.rgb_image, encoding="bgr8")
            self.vis_pub.publish(image_msg)

        self.publish_prediction(joints_vertices)

    def remove_joints_without_fronts(self, indexes_to_keep, joint_coeffs, joint_vertices):
        """
        Creates joint coefficients and joint vertices lists based on indexes_to_keep

        :param indexes_to_keep: list indexes of joints that should be left
        :param joint_coeffs: input list of joint coefficients
        :param joint_vertices: input list of joint vertices
        :return: joint coefficients and joint vertices lists with indexes marked as indexes_to_stay
        """

        joint_coeffs_new = [joint_coeffs[i] for i in indexes_to_keep]
        joint_vertices_new = [joint_vertices[i] for i in indexes_to_keep]

        return joint_coeffs_new, joint_vertices_new

    def attach_joint_to_front(self, joints_coeffs, joints_vertices, distance_threshold=15):
        """
        For each joint, function finds closest rotational front.
         - If distance between joint and rotational front is greater than distance_threshold the joint is removed.
         - If one rotational front has more than 1 predicted joint, the one with minimum distance will be kept. Other
          will be removed
        
        :param distance_threshold: distance threshold in px to mark joint as correct
        :param joints_coeffs: List of joints general line coefficients
        :param joints_vertices: List of joints top and bottom vertices

        :return joints_coeffs: List of joints general line coefficients that have attached fronts to it
        :return joints_vertices: List of joints top and bottom vertices that have attached fronts to it
        :return joint_front_indexes: List of fronts indexes from input prediction that corresponds to joints
        """

        joint_front_indexes = []  # Indexes of fronts corresponding to the joint
        joint_front_distances = []  # Distances from joint to fronts
        joint_indexes_to_keep = []  # Indexes of joints to keep

        for joint_idx, joint in enumerate(joints_coeffs):
            distances_to_fronts = []

            # For each rotational front calculate which edge of front is rotational based on handler position and then
            # calculate the distance between front and predicted joint
            for rot_front_bbox, rot_front_info in zip(self.rot_front_bboxes, self.rot_front_info_list):
                # Get bounding box of handler that is in rotational front
                handler_mask_box = self.handler_data.boxes[rot_front_info["handler_index"]]
                # Get point that lands on the middle of rotational edge of the front
                front_edge_x, front_edge_y = get_point_on_opposite_edge_of_handler(rot_front_bbox, handler_mask_box)
                # Calculate the distance between predicted joint and front
                distance = get_point_line_distance(joint, (front_edge_x, front_edge_y))
                distances_to_fronts.append(distance)

            # Choose front with minimum distance
            distance_to_front = np.min(distances_to_fronts)
            closest_front = np.argmin(distances_to_fronts)

            # Get rid of joints that are predicted way of closest rotational front
            if distance_to_front > distance_threshold:
                continue

            # Save index of front from original front prediction that matches current joint
            front_index = self.rot_front_info_list[closest_front]["original_prediction_index"]

            # Check if current front has attached any joint to it
            if front_index in joint_front_indexes:
                idx = int(np.where(np.array(joint_front_indexes) == front_index)[0])
                distance = joint_front_distances[idx]

                # If it does, then overwrite the front's attached joint only if the new joint is closer than the
                # previous one. If not skip the joint
                if distance_to_front <= distance:
                    joint_front_distances[idx] = distance
                    joint_front_indexes[idx] = front_index
                    joint_indexes_to_keep[idx] = joint_idx
            # Save new joint to the new front
            else:
                joint_front_distances.append(distance_to_front)
                joint_front_indexes.append(front_index)
                joint_indexes_to_keep.append(joint_idx)

        # Keep joint vertices and coefficients only if it has attached front to it
        joints_coeffs, joints_vertices = self.remove_joints_without_fronts(joint_indexes_to_keep,
                                                                           joints_coeffs,
                                                                           joints_vertices)

        return joints_coeffs, joints_vertices, joint_front_indexes

    def publish_prediction(self, vertices):
        prediction_msg = JointPrediction()
        prediction_msg.header = self.header

        for vertice in vertices:
            x1, y1, x2, y2 = vertice
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
