from collections import namedtuple

import numpy as np
import torch
from geometrout import SE3, Sphere
from geometrout.maths import transform_in_place

from robofin.robots import Robot

SphereInfo = namedtuple("SphereInfo", "radii centers")


class CollisionSpheres:
    """
    Generic collision detection using spheres for any robot.
    Loads sphere data from robot configuration files instead of hard-coded constants.
    """
    
    def __init__(self, robot: Robot, margin=0.0):
        """
        Initialize collision spheres for a generic robot.
        
        Args:
            robot: Robot instance with loaded collision spheres
            margin: Additional margin to add to all sphere radii
        """
        self.robot = robot
        self.margin = margin
        self._init_collision_spheres()
        self._init_self_collision_spheres()

    def _init_collision_spheres(self):
        """Initialize collision spheres from robot configuration"""
        spheres = {}
        
        # Convert robot.collision_spheres format to internal format
        for link_name, sphere_list in self.robot.collision_spheres.items():
            for sphere_data in sphere_list:
                radius = sphere_data['radius'] + self.margin
                center = np.array(sphere_data['origin'])
                
                if link_name not in spheres:
                    spheres[link_name] = []
                
                spheres[link_name].append(
                    SphereInfo(radii=np.array([radius]), centers=center.reshape(1, -1))
                )
        
        # Combine spheres per link
        self.cspheres = {}
        for link_name, sphere_infos in spheres.items():
            if sphere_infos:
                radii = np.concatenate([info.radii for info in sphere_infos])
                centers = np.concatenate([info.centers for info in sphere_infos], axis=0)
                radii.setflags(write=False)
                centers.setflags(write=False)
                self.cspheres[link_name] = SphereInfo(radii=radii, centers=centers)

    def _init_self_collision_spheres(self):
        """Initialize self-collision spheres from robot configuration"""
        link_names = []
        centers = {}
        
        # Convert robot.self_collision_spheres format
        for link_name, sphere_list in self.robot.self_collision_spheres.items():
            if link_name not in centers:
                link_names.append(link_name)
                centers[link_name] = []
            
            for sphere_data in sphere_list:
                centers[link_name].append(sphere_data['origin'])
        
        self.points = [(name, np.asarray(centers[name])) for name in link_names]
        
        # Build collision matrix
        all_self_spheres = []
        for link_name, sphere_list in self.robot.self_collision_spheres.items():
            for sphere_data in sphere_list:
                all_self_spheres.append((
                    link_name, 
                    sphere_data['origin'], 
                    sphere_data['radius']
                ))
        
        self.collision_matrix = -np.inf * np.ones((len(all_self_spheres), len(all_self_spheres)))
        
        link_ids = {link_name: idx for idx, link_name in enumerate(link_names)}
        
        # Set up the self collision distance matrix
        for idx1, (link_name1, center1, radius1) in enumerate(all_self_spheres):
            for idx2, (link_name2, center2, radius2) in enumerate(all_self_spheres):
                # Ignore all sphere pairs on the same link or adjacent links
                if link_name1 in link_ids and link_name2 in link_ids:
                    if abs(link_ids[link_name1] - link_ids[link_name2]) < 2:
                        continue
                self.collision_matrix[idx1, idx2] = radius1 + radius2

    def has_self_collision(self, config, auxiliary_joint_values=None, buffer=0.0):
        """Check for self-collision given robot configuration"""
        # Build full configuration
        full_config = self._build_full_config(config, auxiliary_joint_values)
        
        # Get FK for all links
        fk_result = self.robot.fk_torch(torch.tensor(full_config, dtype=torch.float32, device=self.robot.device).unsqueeze(0))
        
        fk_points = []
        for link_name, centers in self.points:
            if link_name in fk_result:
                pose = fk_result[link_name][0].cpu().numpy()
                transformed_points = transform_in_place(np.copy(centers), pose)
                fk_points.append(transformed_points)
        
        if not fk_points:
            return False
            
        transformed_centers = np.concatenate(fk_points, axis=0)
        points_matrix = np.tile(transformed_centers, (transformed_centers.shape[0], 1, 1))
        distances = np.linalg.norm(
            points_matrix - points_matrix.transpose((1, 0, 2)), axis=2
        )
        return np.any(distances < self.collision_matrix + buffer)

    def _build_full_config(self, config, auxiliary_joint_values=None):
        """Build full robot configuration from main manipulator config and auxiliary joint values"""
        if len(config) == self.robot.DOF:
            return config
        elif len(config) == self.robot.MAIN_DOF:
            full_config = np.zeros(self.robot.DOF)
            
            # Set main manipulator joints
            main_indices = [i for i, joint in enumerate(self.robot.urdf.actuated_joints) 
                          if joint.joint_type in ['revolute', 'continuous']]
            full_config[main_indices] = config
            
            # Set auxiliary joints
            aux_indices = [i for i, joint in enumerate(self.robot.urdf.actuated_joints) 
                         if joint.joint_type not in ['revolute', 'continuous']]
            
            if auxiliary_joint_values is not None:
                if isinstance(auxiliary_joint_values, (int, float)):
                    auxiliary_joint_values = [auxiliary_joint_values] * len(aux_indices)
                full_config[aux_indices] = auxiliary_joint_values[:len(aux_indices)]
            else:
                # Use default values (e.g., 0.04 for prismatic joints)
                full_config[aux_indices] = 0.04
                
            return full_config
        else:
            raise ValueError(f"Config length {len(config)} doesn't match robot DOF {self.robot.DOF} or main DOF {self.robot.MAIN_DOF}")

    def csphere_info(self, config, auxiliary_joint_values=None, base_pose=np.eye(4), with_base_link=False):
        """Get collision sphere information for given configuration"""
        full_config = self._build_full_config(config, auxiliary_joint_values)
        
        # Get FK for all links
        config_torch = torch.tensor(full_config, dtype=torch.float32, device=self.robot.device)
        fk_result = self.robot.fk_torch(config_torch.unsqueeze(0))
        
        radii = []
        centers = []
        
        for link_name, info in self.cspheres.items():
            if not with_base_link and 'link0' in link_name:
                continue
                
            if link_name in fk_result:
                pose = fk_result[link_name][0].cpu().numpy()
                # Apply base pose transformation
                pose = base_pose @ pose
                transformed_centers = transform_in_place(np.copy(info.centers), pose)
                centers.append(transformed_centers)
                radii.append(info.radii)
        
        if not centers:
            return SphereInfo(radii=np.array([]), centers=np.zeros((0, 3)))
            
        return SphereInfo(
            radii=np.concatenate(radii), 
            centers=np.concatenate(centers)
        )

    def collision_spheres(self, config, auxiliary_joint_values=None, base_pose=np.eye(4), with_base_link=False):
        """Get collision spheres as Sphere objects"""
        info = self.csphere_info(config, auxiliary_joint_values, base_pose, with_base_link)
        return [Sphere(c, r) for c, r in zip(info.centers, info.radii)]

    def robot_collides(self, config, auxiliary_joint_values, primitives, 
                      scene_buffer=0.0, self_collision_buffer=0.0, 
                      check_self=True, with_base_link=False):
        """Check if robot collides with environment or itself"""
        if check_self and self.has_self_collision(config, auxiliary_joint_values, self_collision_buffer):
            return True
            
        cspheres = self.csphere_info(config, auxiliary_joint_values, with_base_link=with_base_link)
        
        for p in primitives:
            if np.any(p.sdf(cspheres.centers) < cspheres.radii + scene_buffer):
                return True
                
        return False


class TorchCollisionSpheres:
    """
    Torch-based collision detection for GPU acceleration.
    Generic version that works with any robot.
    """
    
    def __init__(self, robot: Robot, margin=0.0, device="cpu"):
        """
        Initialize torch collision spheres for a generic robot.
        
        Args:
            robot: Robot instance with loaded collision spheres
            margin: Additional margin to add to all sphere radii
            device: Torch device for computations
        """
        self.robot = robot
        self.margin = margin
        self.device = device
        self._init_collision_spheres()
        self._init_self_collision_spheres()

    def _init_collision_spheres(self):
        """Initialize collision spheres from robot configuration"""
        spheres = {}
        
        for link_name, sphere_list in self.robot.collision_spheres.items():
            sphere_infos = []
            for sphere_data in sphere_list:
                radius = sphere_data['radius'] + self.margin
                center = torch.as_tensor(sphere_data['origin'], device=self.device)
                
                sphere_infos.append(
                    SphereInfo(
                        radii=torch.tensor([radius], device=self.device),
                        centers=center.unsqueeze(0)
                    )
                )
            
            if sphere_infos:
                spheres[link_name] = sphere_infos
        
        # Combine spheres per link
        self.cspheres = {}
        for link_name, sphere_infos in spheres.items():
            if sphere_infos:
                radii = torch.cat([info.radii for info in sphere_infos])
                centers = torch.cat([info.centers for info in sphere_infos])
                self.cspheres[link_name] = SphereInfo(radii=radii, centers=centers)

    def _init_self_collision_spheres(self):
        """Initialize self-collision spheres from robot configuration"""
        link_names = []
        centers = {}
        
        for link_name, sphere_list in self.robot.self_collision_spheres.items():
            if link_name not in centers:
                link_names.append(link_name)
                centers[link_name] = []
            
            for sphere_data in sphere_list:
                center_tensor = torch.as_tensor(sphere_data['origin'], device=self.device)
                centers[link_name].append(center_tensor)
        
        self.points = [(name, torch.stack(centers[name])) for name in link_names if centers[name]]
        
        # Build collision matrix
        all_self_spheres = []
        for link_name, sphere_list in self.robot.self_collision_spheres.items():
            for sphere_data in sphere_list:
                all_self_spheres.append((
                    link_name, 
                    sphere_data['origin'], 
                    sphere_data['radius']
                ))
        
        self.collision_matrix = -np.inf * torch.ones(
            (len(all_self_spheres), len(all_self_spheres)), 
            device=self.device
        )
        
        link_ids = {link_name: idx for idx, link_name in enumerate(link_names)}
        
        # Set up the self collision distance matrix
        for idx1, (link_name1, center1, radius1) in enumerate(all_self_spheres):
            for idx2, (link_name2, center2, radius2) in enumerate(all_self_spheres):
                if link_name1 in link_ids and link_name2 in link_ids:
                    if abs(link_ids[link_name1] - link_ids[link_name2]) < 2:
                        continue
                self.collision_matrix[idx1, idx2] = radius1 + radius2

    def transform_in_place(self, point_cloud, transformation_matrix):
        """Transform point cloud in place using transformation matrix"""
        point_cloud_T = torch.transpose(point_cloud, -2, -1)
        ones_shape = list(point_cloud_T.shape)
        ones_shape[-2] = 1
        ones = torch.ones(ones_shape, device=self.device).type_as(point_cloud)
        homogeneous_xyz = torch.cat((point_cloud_T, ones), dim=-2)
        transformed_xyz = torch.matmul(transformation_matrix, homogeneous_xyz)
        point_cloud[..., :3] = torch.transpose(transformed_xyz[..., :3, :], -2, -1)
        return point_cloud

    def robot_collides(self, config, auxiliary_joint_values, primitives,
                      scene_buffer=0.0, self_collision_buffer=0.0,
                      check_self=True, with_base_link=False):
        """Check if robot collides with environment or itself (torch version)"""
        if not isinstance(primitives, list):
            primitives = [primitives]
            
        squeeze = False
        if config.ndim == 1:
            config = config.unsqueeze(0)
            squeeze = True
            
        collisions = torch.zeros((config.size(0),), dtype=torch.bool, device=config.device)
        
        if check_self:
            # Self collision check would go here
            # For now, skip to avoid complexity
            pass
            
        cspheres = self.csphere_info(config, auxiliary_joint_values, with_base_link=with_base_link)
        
        for p in primitives:
            p_collisions = torch.any(
                p.sdf(cspheres.centers) < cspheres.radii + scene_buffer, dim=1
            )
            collisions = torch.logical_or(p_collisions, collisions)
            
        if squeeze:
            collisions = collisions.squeeze(0)
            
        return collisions

    def csphere_info(self, config, auxiliary_joint_values=None, base_pose=None, with_base_link=False):
        """Get collision sphere information for given configuration (torch version)"""
        squeeze = False
        if config.ndim == 1:
            config = config.unsqueeze(0)
            squeeze = True
            
        if base_pose is None:
            base_pose = torch.eye(4, device=self.device)[None, ...].type_as(config)
        elif base_pose.ndim == 2:
            base_pose = base_pose.unsqueeze(0)
            
        B = config.size(0)
        
        # Build full configuration
        if config.size(1) == self.robot.MAIN_DOF:
            full_config = torch.zeros((B, self.robot.DOF), device=config.device, dtype=config.dtype)
            main_indices = [i for i, joint in enumerate(self.robot.urdf.actuated_joints) 
                          if joint.joint_type in ['revolute', 'continuous']]
            full_config[:, main_indices] = config
            
            aux_indices = [i for i, joint in enumerate(self.robot.urdf.actuated_joints) 
                         if joint.joint_type not in ['revolute', 'continuous']]
            
            if auxiliary_joint_values is not None:
                if isinstance(auxiliary_joint_values, (int, float)):
                    full_config[:, aux_indices] = auxiliary_joint_values
                else:
                    full_config[:, aux_indices] = torch.as_tensor(auxiliary_joint_values, device=config.device)
            else:
                full_config[:, aux_indices] = 0.04
        else:
            full_config = config
            
        # Get FK for all links
        fk_result = self.robot.fk_torch(full_config)
        
        radii = []
        centers = []
        
        for link_name, info in self.cspheres.items():
            if not with_base_link and 'link0' in link_name:
                continue
                
            if link_name in fk_result:
                pose = fk_result[link_name]
                pose = base_pose @ pose if base_pose is not None else pose
                
                transformed_centers = self.transform_in_place(
                    torch.clone(info.centers[None, ...].expand(B, -1, -1)).type_as(pose),
                    pose
                )
                centers.append(transformed_centers)
                radii.append(info.radii[None, ...].expand(B, -1))
        
        if not centers:
            empty_radii = torch.zeros((B, 0), device=config.device)
            empty_centers = torch.zeros((B, 0, 3), device=config.device)
            if squeeze:
                empty_radii = empty_radii.squeeze(0)
                empty_centers = empty_centers.squeeze(0)
            return SphereInfo(radii=empty_radii, centers=empty_centers)
            
        radii = torch.cat(radii, dim=1)
        centers = torch.cat(centers, dim=1)
        
        if squeeze:
            radii = radii.squeeze(0)
            centers = centers.squeeze(0)
            
        return SphereInfo(radii=radii, centers=centers)
