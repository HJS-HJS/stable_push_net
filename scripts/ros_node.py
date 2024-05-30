#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import collision
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import rospy
import tf
import tf.transformations as tft
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


from stable_push_net_ros.srv import GetStablePushPath, GetStablePushPathRequest, GetStablePushPathResponse
from stable_pushing.stable_push_planner import HybridAstarPushPlanner
from stable_pushing.stable_determinator import StablePushNetDeterminator
from scripts.stable_pushing.stable_push_utils.contact_point_sampler import ContactPointSampler
from stable_pushing.map_interface import MapInterface

    
class StablePushNetServer(object):
    
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.tf = tf.TransformerROS()
        self.map_interface = MapInterface()
        
        # param
        self.planner_config = rospy.get_param("~planner")
        self.hybrid_config = rospy.get_param("~hybrid")
        self.depth_based_config = rospy.get_param("~depth_based")

        # print param to terminal
        rospy.loginfo("planner_config: {}".format(self.planner_config))
        rospy.loginfo("hybrid_config: {}".format(self.hybrid_config))
        rospy.loginfo("depth_based_config: {}".format(self.depth_based_config))

        # initialize ros service
        rospy.Service(
            '/stable_push_planner/get_stable_push_path',
            GetStablePushPath,
            self.get_stable_push_path_handler)

        # stable_determinator with trained model
        stable_determinator = StablePushNetDeterminator()
            
        self.planner = HybridAstarPushPlanner(
            stable_determinator=stable_determinator,
            grid_size=self.hybrid_config['grid_size'],
            dtheta=np.radians(self.hybrid_config['dtheta']))

        # print when server is ready
        rospy.loginfo('StablePushNetServer is ready to serve.')
    
    @staticmethod
    def collision_circles_to_obstacles(circles):
        """Conver Circle msg to collision.Circle object.

        Args:
            circles (List[Circle]): Circle msg.

        Returns:
            obstacles (List[collision.Circle]): List of collision.Circle objects.
        """
        obstacles = []
        for circle in circles:
            obstacles.append(collision.Circle(collision.Vector(circle.x, circle.y), circle.r))
        return obstacles
    
    def get_stable_push_path_handler(self, request):
        """response to ROS service. make push path and gripper pose by using trained model(push net)

        Args:
            request (GetStablePushPathRequest): ROS service from stable task

        Returns:
            GetStablePushPathResponse: generated nav_msgs::Path(), and gripper pose(angle, width)
        """
        assert isinstance(request, GetStablePushPathRequest)
        # save request data
        depth_img_msg = request.depth_image
        segmask_msg = request.segmask
        camera_info_msg = request.cam_info 
        camera_pose_msg = request.cam_pose
        map_info_msg = request.map_info
        goal_pose_msg = request.goal_pose

        # convert data to proper type
        # img to cv2
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
        segmask_img = self.cv_bridge.imgmsg_to_cv2(segmask_msg, desired_encoding='passthrough')
        # camera intrinsic to matrix
        cam_intr = np.array(camera_info_msg.K).reshape(3, 3)
        # camera extrinsic to tf
        cam_pos_tran = [camera_pose_msg.pose.position.x, camera_pose_msg.pose.position.y, camera_pose_msg.pose.position.z]
        cam_pos_quat = [camera_pose_msg.pose.orientation.x, camera_pose_msg.pose.orientation.y, camera_pose_msg.pose.orientation.z, camera_pose_msg.pose.orientation.w]
        cam_pos = self.tf.fromTranslationRotation(cam_pos_tran, cam_pos_quat)

        # map size
        map_corners = [
            map_info_msg.corners[0].x,
            map_info_msg.corners[1].x,
            map_info_msg.corners[0].y,
            map_info_msg.corners[1].y]
        # obstacle position, size
        map_obstacles = self.collision_circles_to_obstacles(map_info_msg.collision_circles)
        
        rospy.loginfo("Received request.")
        
        # Sample push contacts
        cps = ContactPointSampler(cam_intr, cam_pos, 
                                gripper_width = self.planner_config['gripper_width'],
                                num_push_dirs = self.planner_config['num_push_directions'])
        contact_points = cps.sample(depth_img, segmask_img)

        # set goal        
        goal = (goal_pose_msg[0], goal_pose_msg[1], np.pi )

        # Derive stable push path
        self.planner.update_map(map_corners, map_obstacles)
        image = np.multiply(depth_img, segmask_img)

        # Generate push path
        best_path, _, best_pose = self.planner.plan(
            image, contact_points, goal, #depth_img * segmask_img is for masked depth image
            learning_base=True,
            visualize=self.planner_config['visualize'])
        print("[", np.rad2deg(best_pose[0]), best_pose[1], "]")

        # Make ros msg
        res = GetStablePushPathResponse()   
        path_msg = Path()
        path_msg.header.frame_id = camera_pose_msg.header.frame_id
        path_msg.header.stamp = rospy.Time.now()
        for each_point in best_path:
            pose_stamped = PoseStamped()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = camera_pose_msg.header.frame_id
            pose_stamped.pose.position.x, pose_stamped.pose.position.y, pose_stamped.pose.position.z = each_point[0], each_point[1], self.planner_config['height']
            pose_stamped.pose.orientation.x, pose_stamped.pose.orientation.y, pose_stamped.pose.orientation.z, pose_stamped.pose.orientation.w = tft.quaternion_from_euler(each_point[2], 0-np.pi, np.pi/2 - best_pose[0], axes='rzxy')
            path_msg.poses.append(pose_stamped)
        
        # Response the ROS service
        rospy.loginfo('Successfully generate path path')
        res.path = path_msg
        res.gripper_pose = [best_pose[0], best_pose[1]]
        return res

if __name__ == '__main__':
    rospy.init_node('stable_push_net_server')
    server = StablePushNetServer()
    
    rospy.spin()
