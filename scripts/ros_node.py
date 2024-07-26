#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import collision
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from typing import List, Tuple

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
from stable_pushing.utils.contact_point_sampler import ContactPointSampler
from stable_pushing.map_interface import MapInterface

# temp
from moveit_msgs.msg import CartesianTrajectory, CartesianTrajectoryPoint
from stable_push_net_ros.msg import CollisionCircle


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
        
        # Get parameters.
        self.planner_config     = rospy.get_param("~planner")
        self.hybrid_config      = rospy.get_param("~hybrid")
        self.depth_based_config = rospy.get_param("~depth_based")
        self.gripper_config     = rospy.get_param("~gripper")[self.planner_config["gripper"]]

        # Print param to terminal.
        rospy.loginfo("planner_config: {}".format(self.planner_config))
        rospy.loginfo("hybrid_config: {}".format(self.hybrid_config))
        rospy.loginfo("depth_based_config: {}".format(self.depth_based_config))
        rospy.loginfo("depth_based_config: {}".format(self.gripper_config))

        # Initialize ros service.
        rospy.Service(
            '/stable_push_planner/get_stable_push_path',
            GetStablePushPath,
            self.get_stable_push_path_handler
            )

        # Stable_determinator with trained model.
        stable_determinator = StablePushNetDeterminator()

        #################################################  
        # temp Publisher for visualization
        self.push_path_interpolated_pub = rospy.Publisher(
            '/stable_push_server/moveit_msgs/cartersian/interpolated', CartesianTrajectory, queue_size=2)
        self.push_path_origin_pub = rospy.Publisher(
            '/stable_push_server/push_path', Path, queue_size=2)
        #################################################  

        # Hybrid A* Planner
        self.planner = HybridAstarPushPlanner(
            stable_determinator=stable_determinator,
            grid_size=          self.hybrid_config['grid_size'],
            dtheta=             np.radians(self.hybrid_config['dtheta'])
            )

        # Print info message to terminal when push server is ready.
        rospy.loginfo('StablePushNetServer is ready to serve.')
    
    def get_stable_push_path_handler(self, request:GetStablePushPathRequest) -> GetStablePushPathResponse:
        """Response to ROS service. make push path and gripper pose by using trained model(push net).

        Args:
            request (GetStablePushPathRequest): ROS service from stable task

        Returns:
            GetStablePushPathResponse: generated push_path(moveit_msgs::CartesianTrajectory()), plan_successful(bool), gripper pose(float32[angle, width])
        """

        assert isinstance(request, GetStablePushPathRequest)
        # Save service request data.
        dish_seg_msg          = request.dish_segmentation  # vision_msgs/Detection2DArray
        table_det_msg         = request.table_detection    # vision_msgs/BoundingBox3D
        depth_img_msg         = request.depth_image        # sensor_msgs/Image
        camera_info_msg       = request.camera_info        # sensor_msgs/CameraInfo
        camera_pose_msg       = request.camera_pose        # geometry_msgs/PoseStamped
        push_target_array_msg = request.push_targets       # PushTargetArray
        rospy.loginfo("Received request.")
        
        # Parse service request data.
        # Parse segmentation image data.
        # Convert segmentation image list from vision_msgs/Detection2DArray to segmask list and id list.
        segmask_list, id_list = self.parse_dish_segmentation_msg(dish_seg_msg)
        #######################################################
        # 지워도 되는지 확인
        # dish_shape_list = self.map_interface.get_dish_shapes(id_list, segmask_list, depth_img, cam_pos, cam_intr)
        # rospy.loginfo("{} Dishes are segmented".format(len(dish_shape_list)))
        #######################################################

        # Parse table (map) data.
        # Convert table_detection from vision_msgs/BoundingBox3D to map corner and table normal vector matrix.
        map_corners, rot_matrix = self.parse_table_detection_msg(table_det_msg) # min_x, max_x, min_y, max_y

        # Parse camera data.
        # Convert camera extrinsic type from geometry_msgs/PoseStamped to extrinsic tf.
        cam_pos_tran = [camera_pose_msg.pose.position.x, camera_pose_msg.pose.position.y, camera_pose_msg.pose.position.z]
        cam_pos_quat = [camera_pose_msg.pose.orientation.x, camera_pose_msg.pose.orientation.y, camera_pose_msg.pose.orientation.z, camera_pose_msg.pose.orientation.w]
        cam_pos = self.tf.fromTranslationRotation(cam_pos_tran, cam_pos_quat)
        # Convert depth image type from sensor_msgs/Image to cv2.
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_img_msg, desired_encoding='passthrough')
        # Convert camera intrinsic type from sensor_msgs/CameraInfo to matrix.
        cam_intr = np.array(camera_info_msg.K).reshape(3, 3)

        # Parse push_targets data.
        # Convert the dish ID, target point, and pushable range of each recognized plate into a list according to push priority.
        target_id_list, goal_pose_list, push_range_list = self.parse_push_target_msg(push_target_array_msg)
        # Convert a pushable angle range to a non-pushable angle range.
        push_range_list = np.where(push_range_list < 0, push_range_list + 2 * np.pi, push_range_list)
        no_push_range_list = np.fliplr(push_range_list)

        # Set obstacles that must not collide.
        map_obstacles = self.collision_circles_to_obstacles(dish_seg_msg.detections, target_id_list)
        
        # Loop through push path planning until we find the push path
        for i in range(len(target_id_list)):
            # Get target id, goal, and push range according to push priority.
            target_id     = target_id_list[i]
            goal          = goal_pose_list[i]
            no_push_range = no_push_range_list[i]
            segmask_img   = segmask_list[target_id]

            # Convert depth image to masked depth image (to reduce computational amount).
            masked_depth_image = np.multiply(depth_img, segmask_img)

            # Sample the push contact points where the dishes can be pushed.
            cps = ContactPointSampler(cam_intr, 
                                      cam_pos,
                                      gripper_width = self.planner_config['gripper_width'],
                                      num_push_dirs = self.planner_config['num_push_directions']
                                      )
            contact_points = cps.sample(masked_depth_image, no_push_range)

            # Update push map size, obstacles.
            self.planner.update_map(map_corners, map_obstacles)

            # Visualization for Debugging.
            if self.planner_config['visualize']:
                # Show the depth, mask, and masked_depth of the plate.
                fig = plt.figure()
                ax1 = fig.add_subplot(221)
                ax1.imshow(depth_img)
                ax2 = fig.add_subplot(222)
                ax2.imshow(segmask_img)
                ax3 = fig.add_subplot(223)
                ax3.imshow(masked_depth_image)
                plt.show()

                # Show calculated contact points and push points.
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(111)
                ax.imshow(masked_depth_image)
                ax.scatter(contact_points[0].edge_uv[:,0], contact_points[0].edge_uv[:,1], c='k', marker='o')
                for contact_point in contact_points:
                    position = contact_point.contact_points_uv.mean(0)
                    ax.scatter(position[0], position[1], c='r', marker='o')
                ax.set_aspect('equal')
                plt.show()

            # Generate push path
            best_path, is_success, best_pose = self.planner.plan(
                masked_depth_image,
                contact_points, 
                goal,
                visualize=self.planner_config['visualize'])
            
            # Compensate push gripper width.
            if best_pose[1] > 0.1189: best_pose[1] -=0.015 

            # If creating a push path fails, try creating the next plate push path.
            if not is_success: continue

            # Offset between push point and eef according to gripper width.
            _offset = best_pose[1] - 0.114
            # _offset = np.array([-0.038990381 + _offset / 2 / np.tan(np.deg2rad(120)), 0, 0, 0])
            # _offset = np.array([-0.038990381 + _offset / 2 / np.tan(np.deg2rad(120)), 0, 0, 0]) / 2
            _offset = np.array([_offset / 2 / np.tan(np.deg2rad(120)), 0, 0, 0])

            # Make path ros msg
            path_msg = Path()
            path_msg.header.frame_id = camera_pose_msg.header.frame_id
            path_msg.header.stamp = rospy.Time.now()
            for each_point in best_path:
                _pose_stamped = PoseStamped()
                _pose_stamped.header.stamp = rospy.Time.now()
                _pose_stamped.header.frame_id = camera_pose_msg.header.frame_id
                _pose_stamped.pose.position.x, _pose_stamped.pose.position.y = each_point[0], each_point[1]
                _pose_stamped.pose.position.z = self.gripper_config['height'] + self.cal_path_height(each_point[0], each_point[1])
                path_rot_matrix = np.dot(rot_matrix, tft.euler_matrix(each_point[2] + np.deg2rad(self.gripper_config["z_angle"]), 0-np.pi, np.pi/2 - best_pose[0], axes='rzxy'))
                _pose_stamped.pose.position.x += np.dot(path_rot_matrix, _offset)[0]
                _pose_stamped.pose.position.y += np.dot(path_rot_matrix, _offset)[1]
                _pose_stamped.pose.position.z += np.dot(path_rot_matrix, _offset)[2]
                _pose_stamped.pose.orientation.x, _pose_stamped.pose.orientation.y, _pose_stamped.pose.orientation.z, _pose_stamped.pose.orientation.w = tft.quaternion_from_matrix(path_rot_matrix)
                path_msg.poses.append(_pose_stamped)


            print("\n\nIs",camera_pose_msg.header.frame_id, "and \"base_0\" same?")
            print("if so, convert base_0 to msg.frame_id\n\n")

            # Make path ros msg as moveit_msgs::CartesianTrajectory()
            _interpolated_traj = CartesianTrajectory()
            _interpolated_traj.header.stamp = rospy.Time.now()
            _interpolated_traj.header.frame_id = "base_0" # base link of doosan m1013
            _interpolated_traj.tracked_frame = "grasp_point" # end effector of gripper
            _interpolated_traj.points =[]

            # Set pushing velocity
            _vel = 0.1 # m/s
            # Calculate push spent time
            _spent_time = 0
            for i, each_point in enumerate(best_path):
                # calculate spent time between push points
                if i is not (len(best_path) - 1):
                    x = i + 1
                else:
                    x = i
                _lengh = np.linalg.norm(np.array([each_point[0] - best_path[x][0], each_point[1] - best_path[x][1]]))
                _spent_time += rospy.Duration.from_sec(_lengh / _vel)
                print(_spent_time)
            print(_spent_time)

            # Convert each push point to CartesianTrajectory()
            for each_point in best_path:
                # set each CartesianTrajectoryPoint()
                _point = CartesianTrajectoryPoint()
                # whole spent time
                _point.time_from_start = _spent_time
                # point position
                _point.point.pose.position.x, _point.point.pose.position.y = each_point[0], each_point[1]
                _point.point.pose.position.z = self.gripper_config['height'] + self.cal_path_height(each_point[0], each_point[1])
                # apply gripper tilt angle (table angle, gripper push tilt angle)
                path_rot_matrix = np.dot(rot_matrix, tft.euler_matrix(each_point[2] + np.deg2rad(self.gripper_config["z_angle"]), 0-np.pi, np.pi/2 - best_pose[0], axes='rzxy'))
                # apply offset position generated by gripper shape (difference from eef and push point)
                _point.point.pose.position.x += np.dot(path_rot_matrix, _offset)[0]
                _point.point.pose.position.y += np.dot(path_rot_matrix, _offset)[1]
                _point.point.pose.position.z += np.dot(path_rot_matrix, _offset)[2]
                # gripper orientation
                _point.point.pose.orientation.x, _point.point.pose.orientation.y, _point.point.pose.orientation.z, _point.point.pose.orientation.w = tft.quaternion_from_matrix(path_rot_matrix)
                _interpolated_traj.points.append(_point)

            #####################같은지 확인
            # temp
            self.push_path_interpolated_pub.publish(_interpolated_traj)
            self.push_path_origin_pub.publish(path_msg)
            #####################같은지 확인

            # Response the ROS service
            # rospy.loginfo('Successfully generate path')
            # res.path = path_msg
            # res.plan_successful = is_success
            # res.gripper_pose = [best_pose[0], best_pose[1]]
            # return res
        
            # Response the ROS service
            rospy.loginfo('Successfully generate path')
            res = GetStablePushPathResponse()
            res.path = _interpolated_traj
            res.plan_successful = is_success
            res.gripper_pose = [best_pose[0], best_pose[1]]
            return res
        
        # path_msg = Path()
        # path_msg.header.frame_id = camera_pose_msg.header.frame_id
        # path_msg.header.stamp = rospy.Time.now()
        # path_msg.poses = []

        # res = GetStablePushPathResponse()   
        # rospy.loginfo('Path generation failed')
        # res.path = path_msg
        # res.plan_successful = False
        # res.gripper_pose = [90, 0]
        # return res

        # generate empty path msg when path generation failed
        path_msg = CartesianTrajectory()
        path_msg.header.frame_id = camera_pose_msg.header.frame_id
        path_msg.header.stamp = rospy.Time.now()
        path_msg.points = []

        rospy.loginfo('Path generation failed')
        res = GetStablePushPathResponse()   
        res.path = path_msg
        res.plan_successful = False
        res.gripper_pose = [90, 0]
        return res

    @staticmethod
    def collision_circles_to_obstacles(dishes: GetStablePushPathRequest.dish_segmentation.detections, target_id_list: List[int]) -> List[collision.Circle]:
        """Apply recognized dishes as obstacles.

        Args:
            dishes (GetStablePushPathRequest.dish_segmentation.detections): Dishes received through service request.
            target_id_list (List[int]): ID list of recognized dishes.

        Returns:
            obstacles (List[collision.Circle]): List of collision.Circle objects.
        """
        
        # obstacle list
        obstacles = []

        #####################temptemp
        # circles = []
        # obs1 = CollisionCircle()
        # obs2 = CollisionCircle()
        # obs3 = CollisionCircle()
        # obs4 = CollisionCircle()
        # obs5 = CollisionCircle()
        # obs1.x, obs1.y, obs1.r = -0.65, 0.1, 0.1
        # obs2.x, obs2.y, obs2.r = -0.65, 0.2, 0.1
        # obs3.x, obs3.y, obs3.r = -0.15, 0.05, 0.05
        # obs4.x, obs4.y, obs4.r = -0.70, 0.05, 0.05
        # obs5.x, obs5.y, obs5.r = -0.40, 0.05, 0.05
        # circles.append(obs1)
        # circles.append(obs2)
        # circles.append(obs3)
        # circles.append(obs4)
        # circles.append(obs5)
        # for circle in circles:
        #     obstacles.append(collision.Circle(collision.Vector(circle.x, circle.y), circle.r))
        #####################temptemp

        # for dish in dishes:
        #     _r = dish.bbox.size_x if dish.bbox.size_x > dish.bbox.size_y else dish.bbox.size_y
        #     obstacles.append(collision.Circle(collision.Vector(dish.bbox.center.x, dish.bbox.center.y), _r))
        #     print("obs pose: ", dish.bbox.center.x, ", ", dish.bbox.center.y, "obs radius: ", _r)

        return obstacles
    
    def parse_push_target_msg(self, push_target_array_msg:GetStablePushPathRequest.push_targets):
        ''' Parse push target array msg to target ids, push directions and push_range. Each list is arranged according to push priority.
        
        Args:
            push_target_array_msg (GetStablePushPathRequest.PushTargetArray): ROS service from stable task

        Returns:
            tuple containing
            - target_id_list (np.array[int]):
            - goal_pose_list (List[geometry_msgs/Pose2D]):
            - push_range_list (List[Float, Float]):
        '''

        # Create list. Each list has the same index value for the same dish.
        priority_list   = [] # Push priority according to dishes.
        target_id_list  = [] # Target id of the dishes.
        goal_pose_list  = [] # Push target(goal) point along the plate.
        push_range_list = [] # List of possible push angles according to dishes.
        
        # Fill out the list according to each dish.
        for target in push_target_array_msg.push_targets:
            assert isinstance(target, PushTarget)
            priority_list.append(target.priority)
            target_id_list.append(target.push_target_id)
            goal_pose_list.append([target.goal_pose.x, target.goal_pose.y, target.goal_pose.theta])
            push_range_list.append([target.start_pose_min_theta.theta, target.start_pose_max_theta.theta])

        # Convert each list to numpy array.
        priority_array             = np.array(priority_list)
        target_id_array            = np.array(target_id_list)
        goal_pose_array            = np.array(goal_pose_list)
        push_direction_range_array = np.array(push_range_list)
        
        # Sort array by priority.
        sorted_indices = np.argsort(priority_array)
        
        # Rearrange each array according to priority.
        target_id_list  = target_id_array[sorted_indices]
        goal_pose_list  = goal_pose_array[sorted_indices]
        push_range_list = push_direction_range_array[sorted_indices]
            
        return target_id_list, goal_pose_list, push_range_list

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
        
        self.position_msg = table_det_msg.center.position
        orientation_msg = table_det_msg.center.orientation
        self.size_msg = table_det_msg.size
        
        position = np.array([self.position_msg.x, self.position_msg.y, self.position_msg.z])
        orientation = np.array([orientation_msg.x, orientation_msg.y, orientation_msg.z, orientation_msg.w])
        
        rot_mat = tft.quaternion_matrix(orientation)[:3,:3]
        self.n_vector = rot_mat[:,2]
        
        # Get local positions of vertices 
        vertices_loc = []
        for x in [-self.size_msg.x/2, self.size_msg.x/2]:
            for y in [-self.size_msg.y/2, self.size_msg.y/2]:
                for z in [-self.size_msg.z/2, self.size_msg.z/2]:
                    vertices_loc.append([x,y,z])
        vertices_loc = np.array(vertices_loc)
        
        # Convert to world frame
        vertices_world = np.matmul(rot_mat, vertices_loc.T).T + position
        
        x_max, x_min = np.max(vertices_world[:,0]), np.min(vertices_world[:,0])
        y_max, y_min = np.max(vertices_world[:,1]), np.min(vertices_world[:,1])
        # z_max, z_min = np.max(vertices_world[:,2]), np.min(vertices_world[:,2])

        return [x_min, x_max, y_min, y_max], tft.quaternion_matrix(orientation)

    def cal_path_height(self, x, y):
        ''' Parse table detection msg to table pose.'''
        
        _z = self.position_msg.z - self.n_vector[0] / self.n_vector[2] * (x - self.position_msg.x) - self.n_vector[1] / self.n_vector[2] * (y - self.position_msg.y) + self.size_msg.z/2

        return _z

if __name__ == '__main__':
    rospy.init_node('stable_push_net_server')
    server = StablePushNetModuleServer()
    
    rospy.spin()
