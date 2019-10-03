#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from smach import State, StateMachine
from smach_ros import IntrospectionServer
import cv2
import cv_bridge
import numpy as np
import time





class StopState(State):
    def __init__(self, delay_time=.5):
        State.__init__(self, outcomes=['ok'])
        self.delay_time = delay_time
        self.rate = rospy.Rate(10)
        self.angle_error_pub = rospy.Publisher('angle_error', Float64, queue_size=1)
        self.linear_error_pub = rospy.Publisher('linear_error', Float64, queue_size=1)

    def execute(self, ud):
        start_time = time.time()
        while time.time() - start_time < self.delay_time:
            self.angle_error_pub.publish(0.)
            self.linear_error_pub.publish(0.)
            self.rate.sleep()

        return 'ok'


class LineFollowState(State):
    class StopError(Exception):
        pass

    def __init__(self, forward_speed, line_filter, stop_filter):
        State.__init__(self, outcomes=['ok', 'stop'])

        self.forward_speed = forward_speed
        self.line_filter = line_filter
        self.stop_filter = stop_filter

        self.bridge = cv_bridge.CvBridge()
        self.twist = Twist()
        self.image_sub = rospy.Subscriber('camera/rgb/image_raw', Image, self.image_callback)
        self.angle_error_pub = rospy.Publisher('angle_error', Float64, queue_size=1)
        self.linear_error_pub = rospy.Publisher('linear_error', Float64, queue_size=1)

        self.rate = rospy.Rate(10)
        self.image = None
        self.saw_stopline = None

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def wait_for_image(self):
        while self.image is None:
            self.rate.sleep()

    def execute(self, ud):
        self.wait_for_image()

        self.saw_stopline = False
        try:
            while True:
                self.tick()
                self.rate.sleep()
        except self.StopError:
            return 'stop'

    def tick(self):
        image = np.copy(self.image)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        eye_mask = self.get_eye_mask(hsv)
        line_mask = self.line_filter(hsv) & eye_mask
        stop_mask = self.stop_filter(hsv) & eye_mask

        # Check for stopping condition
        stop_mask_empty = np.sum(stop_mask) < 50
        if self.saw_stopline and stop_mask_empty:
            raise self.StopError()
        else:
            self.saw_stopline = not stop_mask_empty

        # Line following
        if not self.is_mask_empty(line_mask):
            cx, cy = self.get_mask_center(line_mask)
            cv2.circle(image, (cx, cy), 20, (0, 0, 255), -1)
            err = cx - image.shape[1] / 2

            self.linear_error_pub.publish(-self.forward_speed)
            self.angle_error_pub.publish(-err)

        # Display
        cv2.imshow('main_camera', image)
        cv2.imshow('line_mask',line_mask * 255)
        cv2.imshow('stop_mask', stop_mask * 255)
        cv2.waitKey(3)

    @staticmethod
    def get_eye_mask(image):
        h, w, d = image.shape
        search_top = 1 * h / 4 - 20
        search_bot = 1 * h / 4
        eye_mask = np.ones((h, w), dtype=bool)
        eye_mask[0:search_top, 0:w] = False
        eye_mask[search_bot:h, 0:w] = False
        return eye_mask

    @staticmethod
    def get_mask_center(mask):
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy

    @staticmethod
    def is_mask_empty(mask):
        return np.all(mask == 0)


def make_course2_follow_state():
    return LineFollowState(
        forward_speed=0.8,
        line_filter=lambda hsv: cv2.inRange(hsv, np.asarray([0, 0, 200]), np.asarray([179, 10, 250])),
        stop_filter=lambda hsv: cv2.inRange(hsv, np.asarray([0, 70, 50]), np.asarray([10, 255, 250])) |
                                cv2.inRange(hsv, np.asarray([170, 70, 50]), np.asarray([180, 255, 250])),
    )


def make_realcourse_follow_state():
    return LineFollowState(
        forward_speed=0.8,
        line_filter=lambda hsv: cv2.inRange(hsv, np.array([0,  0,  225]), np.array([255, 50, 255])),
        stop_filter=lambda hsv: cv2.inRange(hsv, np.asarray([0, 70, 50]), np.asarray([10, 255, 250])) |
                                cv2.inRange(hsv, np.asarray([170, 70, 50]), np.asarray([180, 255, 250]))
    )


rospy.init_node('follower')

sm = StateMachine(outcomes=['ok', 'stop'])
with sm:
    StateMachine.add('FOLLOW', make_realcourse_follow_state(), transitions={'stop': 'STOP'})
    StateMachine.add('STOP', StopState(), transitions={'ok': 'FOLLOW'})

sis = IntrospectionServer('smach_server', sm, '/SM_ROOT')
sis.start()

sm.execute()
