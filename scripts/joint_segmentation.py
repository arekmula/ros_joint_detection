#!usr/bin/env python3.7

# System imports
import sys
import threading

# ROS imports
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String, Header


class JointSegmentator:

    def __init__(self, rgb_image_topic):
        # Topic name of 640x480 image to subscribe
        self.rgb_image_topic = rgb_image_topic

        # Topic name where prediction will be published
        joint_prediction_topic = rospy.get_param("joint_prediction_topic", default="joint_prediction")
        # self.prediction_pub = rospy.Publisher(joint_prediction_topic, None, queue_size=1)

        self.cv_bridge = CvBridge()

        # Last input message and message lock
        self.last_msg = None
        self.msg_lock = threading.Lock()

    def image_callback(self, data):
        rospy.logdebug("Get input image")

        if self.msg_lock.acquire(False):
            self.last_msg = data
            self.msg_lock.release()

    def run_inference(self):
        image_sub = rospy.Subscriber(self.rgb_image_topic, data_class=Image, callback=self.image_callback,
                                     queue_size=1, buff_size=2 ** 24)

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self.msg_lock.acquire(False):
                msg = self.last_msg
                self.last_msg = None
                self.msg_lock.release()

            if msg is not None:
                try:
                    cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                except CvBridgeError as e:
                    print(e)


def main(args):
    rospy.init_node("joint_segmentation")
    if rospy.has_param("rgb_image_topic"):
        rgb_image_topic = rospy.get_param("rgb_image_topic")
        print(rgb_image_topic)

    joint_segmentator = JointSegmentator(rgb_image_topic)
    joint_segmentator.run()


if __name__ == '__main__':
    main(sys.argv)