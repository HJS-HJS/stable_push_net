import copy
from itertools import product
import numpy as np
from typing import List, Union
from scipy.spatial.transform import Rotation as R
from scipy import ndimage
import cv2

def tmat(pose):
    ''' Pose datatype conversion
    
    gymapi.Transform -> Homogeneous transformation matrix (4 x 4)
    
    '''
    t = np.eye(4)
    t[0, 3], t[1, 3], t[2, 3] = pose.p.x, pose.p.y, pose.p.z
    quat = np.array([pose.r.x, pose.r.y, pose.r.z, pose.r.w])
    t[:3,:3] = R.from_quat(quat).as_matrix()
    return t

def fibonacci_lattice(samples: int=2000) -> List[Union[float, float]]:
    """generate ICR in fibonacci lattice

    Args:
        samples (int, optional): Number to make lattice velocity samples. Defaults to 2000.

    Returns:
        List[Union[float, float]]: Point list with [x, y]. x=[-1,1], y=[0,1]
    """
    points = []
    phi = (1. + np.sqrt(5.)) / 2.  # golden angle in radians

    for i in range(samples):
        x = 1. - 2. * ((i / phi) % 1.)
        y = i / (samples - 1)
        points.append((x, y))
        
    return np.vstack((np.array(points).T,np.random.rand(samples)))

def square_board(samples: int=2000) -> List[Union[float, float]]:
    """generate ICR in square board

    Args:
        samples (int, optional): Number to make lattice velocity samples. Defaults to 2000.

    Returns:
        List[Union[float, float]]: Point list with [x, y]. x=[-1,1], y=[0,1]
    """
    _z_num = 4
    # _root = int(np.sqrt(samples / _z_num))
    # while True:
    #     if (samples / _z_num)%_root==0:
    #         break
    #     else:
    #         _root += 1
    _root = 100

    _x = np.linspace(-1, 1, _root)
    _x = np.sign(_x) * np.power(_x, 2)
    _y = np.linspace(0, 1, int(samples / _z_num /_root))
    _z = np.linspace(0, 1, _z_num)

    _model_input = np.array(list(product(_x, _y, _z)))

    return _model_input.T, np.array([_root, samples / _z_num /_root, _z_num]).astype(np.uint)

def model_input(samples: int=2000, mode: List=[None, None, None]) -> Union[Union[float, float, float], Union[float, float, float]]:
    """Created model input and real value list

    Args:
        samples (int, optional): Number of model inputs(points). Defaults to 2000.
        mode (List, optional): Fix input value. Defaults to [None, None, None]. Each list means icr[m], gripper angle[deg], and gripper width[m] in that order. Set to none if you do not want to change it.

    Returns:
        Union[Union[float, float, float], Union[float, float, float]]: Returns model input and real value. The model value is a value directly inserted into the learned model, and the real value indicates what each value actually means.
    """
    
    # Parameters
    MAX_R=0.5
    MIN_R=-1.5
    MAX_A=np.pi/2
    MIN_A=0
    MAX_L=0.08
    MIN_L=0.04

    _model_input = (fibonacci_lattice(samples=samples))
    _model_input[2,:] = MAX_L - (MAX_L - MIN_L) * _model_input[2,:]
                    
    _real_value = copy.deepcopy(_model_input)
    # ICR biased
    # _real_value[0,:] = (np.sign(_model_input[0,:]) * np.power(10, MIN_R + np.abs(_model_input[0,:]) * (MAX_R - MIN_R)))
    # ICR uniform distribution
    _real_value[0,:] = np.sign(_model_input[0,:]) * (np.power(10, MIN_R) + np.abs(_model_input[0,:]) * (np.power(10, MAX_R) - np.power(10, MIN_R)))
    _model_input[0,:] = np.sign(_real_value[0,:]) * (np.log10(np.abs(_real_value[0,:])) - MIN_R) / (MAX_R - MIN_R)
    _real_value[1,:] = 90 - np.rad2deg( MIN_A + _model_input[1,:] * (MAX_A - MIN_A))

    for i, _num in enumerate(mode):
        if _num is None:
            continue
        else:
            if i == 0:
                _real_value[0,:] = _num
                _model_input[0,:] = np.sign(_real_value[0,:]) * (np.log10(np.abs(_real_value[0,:])) - MIN_R) / (MAX_R - MIN_R)
                pass

            elif i == 1:
                _model_input[1,:] = (np.pi/2 - _num - MIN_A) / (MAX_A - MIN_A)
                _real_value[1,:] = np.rad2deg(_num)
                pass

            elif i == 2:
                _model_input[2,:] = _num
                _real_value[2,:] = _num
                pass
            
    return _model_input.T, _real_value.T

def checker_input(samples: int=2000, mode: List=[None, None, None]) -> Union[Union[float, float, float], Union[float, float, float]]:
    """Created model input and real value list

    Args:
        samples (int, optional): Number of model inputs(points). Defaults to 2000.
        mode (List, optional): Fix input value. Defaults to [None, None, None]. Each list means icr[m], gripper angle[deg], and gripper width[m] in that order. Set to none if you do not want to change it.

    Returns:
        Union[Union[float, float, float], Union[float, float, float]]: Returns model input and real value. The model value is a value directly inserted into the learned model, and the real value indicates what each value actually means.
    """
    
    # Parameters
    MAX_R=0.5
    MIN_R=-1.5
    MAX_A=np.pi/2
    MIN_A=0
    MAX_L=0.08
    MIN_L=0.04

    _model_input, _shape = square_board(samples=samples)
    _model_input[2,:] = MAX_L - (MAX_L - MIN_L) * _model_input[2,:]

    _real_value = copy.deepcopy(_model_input)
    # ICR uniform distribution
    _real_value[0,:] = np.sign(_model_input[0,:]) * (np.power(10, MIN_R) + np.abs(_model_input[0,:]) * (np.power(10, MAX_R) - np.power(10, MIN_R)))
    _model_input[0,:] = np.sign(_real_value[0,:]) * (np.log10(np.abs(_real_value[0,:])) - MIN_R) / (MAX_R - MIN_R)
    _real_value[1,:] = 90 - np.rad2deg( MIN_A + _model_input[1,:] * (MAX_A - MIN_A))

    for i, _num in enumerate(mode):
        if _num is None:
            continue
        else:
            if i == 0:
                _real_value[0,:] = _num
                _model_input[0,:] = np.sign(_real_value[0,:]) * (np.log10(np.abs(_real_value[0,:])) - MIN_R) / (MAX_R - MIN_R)
                pass

            elif i == 1:
                _model_input[1,:] = (np.pi/2 - _num - MIN_A) / (MAX_A - MIN_A)
                _real_value[1,:] = np.rad2deg(_num)
                pass

            elif i == 2:
                _model_input[2,:] = _num
                _real_value[2,:] = _num
                pass
    return _model_input.T, _real_value.T, _shape

def crop_image(depth_image, push_contact):
        ''' Convert the given depth image to the network image input
        
        1. Set the contact point to the center of the image
        2. Rotate the image so that the push direction is aligned with the x-axis
        3. Crop the image so that the object can be fit into the entire image
        4. Resize the image to the network input size (96 x 96)
        
        '''
        
        image_height, image_width = 96, 96

        H,W = depth_image.shape
        contact_points_uv = push_contact.contact_points_uv
        edge_uv = push_contact.edge_uv
        edge_center_uv = edge_uv.mean(axis=0)
        
        '''
        contact_points_uv, edge_uv: [row, col] = [u,v]
        Image coordinate:           [row, col] = [v,u]
        '''
        
        contact_center_uv = contact_points_uv.mean(0).astype(int)
        contact_center_vu = np.array([contact_center_uv[1], contact_center_uv[0]])
        
        ########################################################
        # Modify pushing direction to head to the -v direction #
        ########################################################
        u1, v1 = contact_points_uv[0]
        u2, v2 = contact_points_uv[1]
        push_dir = np.array([u1-u2,v2-v1])
        
        # Center of the rotated edge center should be in -v direction
        rot_rad = np.pi - np.arctan2(push_dir[1],push_dir[0])  # push direction should head to the -v direction (up in the image)
        R = np.array([[np.cos(rot_rad), -np.sin(rot_rad)], 
                      [np.sin(rot_rad),  np.cos(rot_rad)]])
        edge_center_vu = np.array([edge_center_uv[1], edge_center_uv[0]])
        rotated_edge_center = R @ (edge_center_vu - contact_center_vu)
        
        if rotated_edge_center[0] > 0:
            rot_angle = 180 + np.rad2deg(rot_rad)
        else:
            rot_angle = np.rad2deg(rot_rad)
            
        ###################################
        # Rotate and crop the depth image #
        ###################################
        
        # Shift the image so that the contact point is at the center
        shifted_img = ndimage.shift(depth_image, (np.round(H/2-contact_center_vu[0]).astype(int), np.round(W/2-contact_center_vu[1]).astype(int)), mode='nearest')
        
        # Rotate the image so that the pushing direction heads to -v direction
        rotated_img = ndimage.rotate(shifted_img, rot_angle, mode='nearest', reshape=False)
        
        
        # Crop the image so that the object can be fit into the entire image
        center_y, center_x = np.round(H/2).astype(int), np.round(W/2).astype(int)
        crop_size_unit = 75
        cropped_img = rotated_img[center_y - 3*crop_size_unit : center_y + crop_size_unit, center_x  - 2*crop_size_unit : center_x  + 2*crop_size_unit]
        
        # Resize the image to the network input size (96 x 96)
        cropped_img = cv2.resize(cropped_img, (image_width,image_height))
        
        return cropped_img
    
    
    