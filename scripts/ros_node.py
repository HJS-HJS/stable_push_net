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
from stable_push_net_ros.srv import GetStablePushPathTest, GetStablePushPathTestRequest, GetStablePushPathTestResponse
from stable_push_net_ros.msg import PushTarget
from stable_pushing.stable_push_planner import HybridAstarPushPlanner
from stable_pushing.stable_determinator import StablePushNetDeterminator
from stable_pushing.stable_push_utils.contact_point_sampler import ContactPointSampler
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
            GetStablePushPathTest,
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
            request (GetStablePushPathTestRequest): ROS service from stable task

        Returns:
            GetStablePushPathTestResponse: generated nav_msgs::Path(), and gripper pose(angle, width)
        """
        assert isinstance(request, GetStablePushPathTestRequest)
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
        res = GetStablePushPathTestResponse()   
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

class StablePushNetModuleServer(object):
    
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
    def collision_circles_to_obstacles(dishes):
        """Conver Circle msg to collision.Circle object.

        Args:
            circles (List[Circle]): Circle msg.

        Returns:
            obstacles (List[collision.Circle]): List of collision.Circle objects.
        """
        obstacles = []
        for dish in dishes:
            _r = dish.bbox.size_x if dish.bbox.size_x > dish.bbox.size_y else dish.bbox.size_y
            obstacles.append(collision.Circle(collision.Vector(dish.bbox.center.x, dish.bbox.center.y), _r))
            print("obs pose: ", dish.bbox.center.x, ", ", dish.bbox.center.y, "obs radius: ", _r)
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
        dish_seg_msg = request.dish_segmentation
        table_det_msg = request.table_detection
        depth_img_msg = request.depth_image
        camera_info_msg = request.cam_info
        camera_pose_msg = request.camera_pose
        push_target_array_msg = request.push_targets
        
        rospy.loginfo("Received request.")
        
        # Parse push target msg
        target_id_list, goal_pose_list, push_direction_range_list = self.parse_push_target_msg(push_target_array_msg)

        # camera extrinsic to tf
        cam_pos_tran = [camera_pose_msg.pose.position.x, camera_pose_msg.pose.position.y, camera_pose_msg.pose.position.z]
        cam_pos_quat = [camera_pose_msg.pose.orientation.x, camera_pose_msg.pose.orientation.y, camera_pose_msg.pose.orientation.z, camera_pose_msg.pose.orientation.w]
        cam_pos = self.tf.fromTranslationRotation(cam_pos_tran, cam_pos_quat)

        # convert data to proper type
        # img to cv2
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
        # camera intrinsic to matrix
        cam_intr = np.array(camera_info_msg.K).reshape(3, 3)

        segmask_list, id_list = self.parse_dish_segmentation_msg(dish_seg_msg)
        dish_shape_list = self.map_interface.get_dish_shapes(id_list, segmask_list, depth_img, cam_pos, cam_intr)
        rospy.loginfo("{} Dishes are segmented".format(len(dish_shape_list)))


        # map size
        # map_corners = [
        #     map_info_msg.corners[0].x,
        #     map_info_msg.corners[1].x,
        #     map_info_msg.corners[0].y,
        #     map_info_msg.corners[1].y]
        map_corners = self.parse_table_detection_msg(table_det_msg) # min_x, max_x, min_y, max_y
        
        # obstacle position, size
        # map_obstacles = self.collision_circles_to_obstacles(dish_seg_msg.detections, target_id_list)
        map_obstacles = self.collision_circles_to_obstacles([], target_id_list)
        
        # Loop through push path planning until we find the push path
        for i in range(len(target_id_list)):
            # Get current target ID
            target_id = target_id_list[i]
            goal_pose = goal_pose_list[i]
            push_direction_range = push_direction_range_list[i]

            # Get corresponding data
            segmask_img = segmask_list[target_id]

            # Sample push contacts
            cps = ContactPointSampler(cam_intr, cam_pos, 
                                    gripper_width = self.planner_config['gripper_width'],
                                    num_push_dirs = self.planner_config['num_push_directions'])
            contact_points = cps.sample(depth_img, segmask_img)

            # set goal        
            goal = goal_pose

            # Derive stable push path
            self.planner.update_map(map_corners, map_obstacles)
            image = np.multiply(depth_img, segmask_img)

            # Generate push path
            best_path, is_success, best_pose = self.planner.plan(
                image, contact_points, goal, #depth_img * segmask_img is for masked depth image
                learning_base=True,
                visualize=self.planner_config['visualize'])
            print("[", np.rad2deg(best_pose[0]), best_pose[1], "]")

            if not is_success:
                best_path = []


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
            res.plan_successful = is_success
            res.gripper_pose = [best_pose[0], best_pose[1]]
            return res

    def parse_push_target_msg(self, push_target_array_msg):
        ''' Parse push target array msg to target ids and push directions.'''
        priority_list = []
        target_id_list = []
        goal_pose_list = []
        push_direction_range_list = []
        
        
        for target in push_target_array_msg.push_targets:
            assert isinstance(target, PushTarget)
            priority_list.append(target.priority)
            target_id_list.append(target.push_target_id)
            goal_pose_list.append([target.goal_pose.x, target.goal_pose.y, target.goal_pose.theta])
            push_direction_range_list.append([target.start_pose_min_theta.theta, target.start_pose_max_theta.theta])
            
        priority_array = np.array(priority_list)
        target_id_array = np.array(target_id_list)
        goal_pose_array = np.array(goal_pose_list)
        push_direction_range_array = np.array(push_direction_range_list)
        
        # Sort target goals by priority
        sorted_indices = np.argsort(priority_array)
        
        target_id_list = target_id_array[sorted_indices]
        goal_pose_list = goal_pose_array[sorted_indices]
        push_direction_range_list = push_direction_range_array[sorted_indices]
            
        return target_id_list, goal_pose_list, push_direction_range_list

    def parse_dish_segmentation_msg(self, dish_segmentation_msg):
        ''' Parse dish segmentation msg to segmasks and ids.'''
        
        segmasks = []
        
        for detection in dish_segmentation_msg.detections:
            # Get segmask
            segmask_msg = detection.source_img
            segmask = self.cv_bridge.imgmsg_to_cv2(segmask_msg, desired_encoding='passthrough')
            segmasks.append(segmask)
            
        # Since all dishes have same segmentation id, we have to manually assign ids
        ids = [i for i in range(len(segmasks))]
        
        return segmasks, ids
    
    def parse_table_detection_msg(self, table_det_msg):
        ''' Parse table detection msg to table pose.'''
        
        position_msg = table_det_msg.center.position
        orientation_msg = table_det_msg.center.orientation
        size_msg = table_det_msg.size
        
        position = np.array([position_msg.x, position_msg.y, position_msg.z])
        orientation = np.array([orientation_msg.x, orientation_msg.y, orientation_msg.z, orientation_msg.w])
        
        rot_mat = tft.quaternion_matrix(orientation)[:3,:3]
        
        # Get local positions of vertices 
        vertices_loc = []
        for x in [-size_msg.x/2, size_msg.x/2]:
            for y in [-size_msg.y/2, size_msg.y/2]:
                for z in [-size_msg.z/2, size_msg.z/2]:
                    vertices_loc.append([x,y,z])
        vertices_loc = np.array(vertices_loc)
        
        # Convert to world frame
        vertices_world = np.matmul(rot_mat, vertices_loc.T).T + position
        
        x_max, x_min = np.max(vertices_world[:,0]), np.min(vertices_world[:,0])
        y_max, y_min = np.max(vertices_world[:,1]), np.min(vertices_world[:,1])
        # z_max, z_min = np.max(vertices_world[:,2]), np.min(vertices_world[:,2])
            
        return [x_min, x_max, y_min, y_max]

if __name__ == '__main__':
    rospy.init_node('stable_push_net_server')
    # server = StablePushNetServer()
    server = StablePushNetModuleServer()
    
    rospy.spin()
