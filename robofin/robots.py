#!/usr/bin/env python3
"""
Generic Robot class to replace all robot-specific classes.
Uses urchin URDF for robot-agnostic kinematics and dynamics.
"""

from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
from collections import OrderedDict, deque

import json
import yaml
import numpy as np
import torch
import urchin
from termcolor import cprint
from scipy.spatial.transform import Rotation

from robofin.point_cloud_tools import transform_point_cloud
from .torch_urdf import TorchURDF


class Robot:
    """
    Stateless class that provides helpful methods for a robot defined in a URDF.
    """
    def __init__(self,
                 urdf_path: Union[str, Path],
                 device: str | torch.device = 'cpu'):
        """
        Initialize robot from directory containing URDF and config files.
        
        Expected structure:
        robot_directory/
        ├── *.urdf                    # URDF file  
        ├── robot_config.yaml          # Link configuration
        ├── collision_spheres/        # Collision spheres
        └── meshes/                   # Mesh files
        """
        self.urdf_path = urdf_path
        self.robot_directory = Path(urdf_path).parent
        self.device = device
        self.urdf = urchin.URDF.load(str(self.urdf_path), lazy_load_meshes=True)
        self.torch_urdf = TorchURDF.load(str(self.urdf_path),
                                         lazy_load_meshes=True,
                                         device=self.device)
        self.include_base_link_spheres = False  # influences what compute_spheres() returns

        # Load robot configuration from robot_config.yaml
        self._load_robot_config()

        # Load collision spheres from collision_spheres/
        self._load_collision_spheres()

        # Extract robot properties from URDF
        self._extract_robot_properties()

    def _load_robot_config(self):
        """Load robot configuration from robot_config.yaml"""
        robot_config_path = self.robot_directory / "robot_config.yaml"
        if not robot_config_path.exists():
            raise FileNotFoundError(f"robot_config.yaml not found in {self.robot_directory}")
            
        with open(robot_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        robot_cfg = config['robot_config']
        self.tcp_link_name = robot_cfg['tcp_link_name']
        self.base_link_name = robot_cfg['base_link_name']
        # self.eef_base_link_name = robot_cfg['eef_base_link_name']
        self.eef_link_names = robot_cfg['eef_links']
        self.eef_visual_link_names = robot_cfg['eef_visual_links']
        self.arm_visual_link_names = robot_cfg['arm_visual_links']
        self.arm_link_names = robot_cfg['arm_links']
        self.auxiliary_joint_names: List[str] = robot_cfg['auxiliary_joint_names']
        auxiliary_joints_values: List[float] = robot_cfg['auxiliary_joints_values']
        self.auxiliary_joint_defaults = {joint_name: joint_value for joint_name, joint_value in zip(self.auxiliary_joint_names, auxiliary_joints_values)}
        # NOTE: mimic joints are not included in self.urdf.actuated_joints
        self.main_joint_names: List[str] = [j.name for j in self.urdf.actuated_joints if j.name not in self.auxiliary_joint_names] # same order as given in urdf
            
        self.neutral_config = np.array(robot_cfg['neutral_config']) # only main joints included (not auxiliary joints, not mimic joints)
        self.neutral_config_dict = {joint_name: joint_value for joint_name, joint_value in zip(self.main_joint_names, self.neutral_config)} 
        for joint_name in self.auxiliary_joint_names:
            if joint_name not in self.neutral_config_dict:
                self.neutral_config_dict[joint_name] = self.auxiliary_joint_defaults[joint_name]

        self.MAIN_DOF = len(self.main_joint_names) # controlled DOF
        self.DOF = len(self.urdf.actuated_joints) # total DOF (incl. constant auxiliary joints, excluding mimic joints)
    
    def _load_collision_spheres(self):
        """Load collision spheres from collision_spheres/"""
        collision_spheres_path = self.robot_directory / "collision_spheres"
        if not collision_spheres_path.exists():
            raise FileNotFoundError(f"collision_spheres/ not found in {self.robot_directory}")

        self.collision_spheres = {}
        self.self_collision_spheres = {}

        collision_spheres_file = collision_spheres_path / "collision_spheres.json"
        self_collision_spheres_file = collision_spheres_path / "self_collision_spheres.json"

        if not collision_spheres_file.exists():
            raise FileNotFoundError(f"collision_spheres.json or self_collision_spheres.json not found in {collision_spheres_path}")
        with open(collision_spheres_file, "r") as f:
            self.collision_spheres: Dict = json.load(f)

        if not self_collision_spheres_file.exists():
            cprint(f"No self_collision_spheres.json found in {collision_spheres_path}, using collision_spheres.json", "yellow")
            self.self_collision_spheres = self.collision_spheres
        else:
            with open(self_collision_spheres_file, "r") as f:
                self.self_collision_spheres: Dict = json.load(f)

        self._prep_collision_spheres()

    def _prep_collision_spheres(self):
        """
        Prepare collision spheres for fast access during batched sdf evaluation. 
        Store a list of tulpes: List[Tuple[radius: float, Dict[link_name: str, sphere_xyz: np.ndarray]]] 
        to group spheres by radius. All radii are rounded to 3 decimal places.
        """

        # map radii to sphere origins (grouped by link)
        reversed_dict: Dict[float, Dict[str, np.ndarray]] = {}
        for link, spheres in self.collision_spheres.items():
            for sphere in spheres:
                radius = round(sphere['radius'], 3)
                origin = np.array(sphere['origin'])

                if radius not in reversed_dict:
                    reversed_dict[radius] = {}
                if link not in reversed_dict[radius]:
                    reversed_dict[radius][link] = np.array([]).reshape(0, 3)
                reversed_dict[radius][link] = np.vstack([reversed_dict[radius][link], origin])


        # Convert the reversed_dict to a list of tuples
        reversed_spheres: List[Tuple[float, Dict[str, np.ndarray]]] = [] # identical format to FrankaConstants.SPHERES
        for radius, sphere_origins in reversed_dict.items():
            if len(sphere_origins) > 0:
                for link, spheres in sphere_origins.items():
                    sphere_origins[link] = np.array(spheres)
                reversed_spheres.append((radius, sphere_origins))
        
        self.spheres: List[Tuple[float, Dict[str, torch.Tensor]]] = []
        for radius, point_set in reversed_spheres:
            sphere_centers = {
                k: torch.as_tensor(v).to(self.device) for k, v in point_set.items()
            }
            if not self.include_base_link_spheres:
                sphere_centers = {
                    k: v for k, v in sphere_centers.items() if k != self.base_link_name
                }
            if len(sphere_centers) == 0:
                continue
            self.spheres.append(
                (
                    radius,
                    sphere_centers,
                )
            )


    def _extract_robot_properties(self):
        """Extract robot properties from URDF"""
        self.name = self.urdf.name
        self.actuated_joints: List[urchin.Joint] = self.urdf.actuated_joints
        self.main_joints: List[urchin.Joint] = [j for j in self.urdf.actuated_joints if j.name in self.main_joint_names]
        self.auxiliary_joints: List[urchin.Joint] = [j for j in self.urdf.actuated_joints if j.name in self.auxiliary_joint_names]

        self.main_joint_limits = np.array([
            [j.limit.lower if j.limit and j.limit.lower is not None else -np.inf,
             j.limit.upper if j.limit and j.limit.upper is not None else np.inf]
            for j in self.main_joints
        ])

        self.joint_limits = np.array([
            [j.limit.lower if j.limit and j.limit.lower is not None else -np.inf,
             j.limit.upper if j.limit and j.limit.upper is not None else np.inf]
            for j in (self.urdf.actuated_joints + self.auxiliary_joints)
        ])
        
        self.velocity_limits = np.array([
            j.limit.velocity if j.limit and j.limit.velocity is not None else np.inf
            for j in self.urdf.actuated_joints
        ])
        
        # Gather links with static transforms to the tcp link
        # Also gather auxiliary joints that are connected to the end effector, e.g "finger joints"
        # NOTE: This is not a complete solution, as it does not account for nested auxiliary joints, e.g. "finger joints" of a "finger" link
        self.fixed_eef_link_transforms = {}
        self.eef_aux_joints = {}  # parent link name -> list of (joint, child link name)
        queue = deque()
        tcp = self.tcp_link_name
        self.fixed_eef_link_transforms[tcp] = np.eye(4)
        queue.append((tcp, np.eye(4)))
        visited = set([tcp])
        while queue:
            current, current_transform = queue.popleft()
            adjacent_joints = [j for j in self.urdf.joints if j.parent == current or j.child == current]
            for j in adjacent_joints:
                if j.joint_type != "fixed":
                    if j.name in self.auxiliary_joint_names and j.child in self.eef_link_names and j.parent == current:
                        if current not in self.eef_aux_joints:
                            self.eef_aux_joints[current] = []
                        self.eef_aux_joints[current].append((j, j.child))
                    else:
                        continue
                    continue
                
                if j.parent == current: # traversing to child
                    child_link = j.child
                    if child_link not in visited:
                        transform = np.matmul(j.origin, current_transform)
                        self.fixed_eef_link_transforms[child_link] = transform
                        queue.append((child_link, transform))
                        visited.add(child_link)
                elif j.child == current: # traversing to parent
                    parent_link = j.parent
                    if parent_link not in visited:
                        transform = np.matmul(np.linalg.inv(j.origin), current_transform)
                        self.fixed_eef_link_transforms[parent_link] = transform
                        queue.append((parent_link, transform))
                        visited.add(parent_link)

        self.torch_fixed_eef_link_transforms = {
            k: torch.as_tensor(v, dtype=torch.float64, device=self.device)
            for k, v in self.fixed_eef_link_transforms.items()
        }

    
    def get_visual_transform(self, link_name: str) -> np.ndarray:
        """
        Get visual transform for a link from URDF.
        
        Args:
            link_name: Name of the link
            
        Returns:
            4x4 transform matrix for the visual origin
        """
        link = None
        for l in self.urdf.links:
            if l.name == link_name:
                link = l
                break

        if link is None or not link.visuals:
            cprint(f"get_visual_transform() | Warning: No visual found for link {link_name}", "yellow")
            return np.eye(4)

        visual = link.visuals[0] # first visual element (assumed only)
        if hasattr(visual, 'origin') and visual.origin is not None:
            return np.asarray(visual.origin, dtype=np.float64)

        cprint(f"get_visual_transform() | Warning: No visual origin found for link {link_name}", "yellow")
        return np.eye(4)
    
    def torch_get_visual_transform(self,
                                   link_name: str,
                                   dtype: Optional[torch.dtype]=None) -> torch.Tensor:
        """
        Get visual transform for a link from URDF.
        
        Args:
            link_name: Name of the link
            
        Returns:
            4x4 transform matrix for the visual origin
        """
        link = None
        for l in self.torch_urdf.links:
            if l.name == link_name:
                link = l
                break

        if link is None or not link.visuals:
            cprint(f"torch_get_visual_transform() | Warning: No visual found for link {link_name}", "yellow")
            return torch.eye(4)

        visual = link.visuals[0] # first visual element (assumed only)
        if hasattr(visual, 'origin') and visual.origin is not None:
            return torch.as_tensor(visual.origin,
                                   dtype=dtype if dtype is not None else torch.float32,
                                   device=self.device)

        cprint(f"torch_get_visual_transform() | Warning: No visual origin found for link {link_name}", "yellow")
        return torch.eye(4)


    def within_limits(self, config):
        """Check if configuration is within joint limits"""
        # Handle both main manipulator and full robot configs
        if len(config) == self.MAIN_DOF:
            limits = self.main_joint_limits
        elif len(config) == self.DOF:
            limits = self.joint_limits
        else:
            raise ValueError(f"Config length {len(config)} doesn't match robot DOF {self.DOF} or main DOF {self.MAIN_DOF}")
        
        # Add small tolerance for float math
        return np.all(config >= limits[:, 0] - 1e-5) and np.all(config <= limits[:, 1] + 1e-5)
    
    def random_neutral(self, method="normal", scale=0.25):
        """Generate random configuration near neutral position"""
        if method == "normal":
            return np.clip(
                self.neutral_config + np.random.normal(0, scale, self.MAIN_DOF),
                self.main_joint_limits[:, 0],
                self.main_joint_limits[:, 1],
            )
        elif method == "uniform":
            return self.neutral_config + np.random.uniform(-scale, scale, self.MAIN_DOF)
        else:
            raise ValueError("method must be either 'normal' or 'uniform'")
    
    def random_configuration(self, joint_range_scalar=1.0):
        """Generate random configuration within joint limits"""
        limits = joint_range_scalar * self.main_joint_limits
        return (limits[:, 1] - limits[:, 0]) * np.random.rand(self.MAIN_DOF) + limits[:, 0]
    
    def _construct_full_config(self, 
                               config: np.ndarray, 
                               auxiliary_joint_values: Optional[Dict[str, float]]=None) -> np.ndarray:
        """
        Construct full configuration vector from main config and auxiliary 
        joint values.
        B: batch dimension.
        
        Args:
            config : np.ndarray of shape (B, MAIN_DOF) or (MAIN_DOF,), a configuration vector for 
                the main joints.
            auxiliary_joint_values : Dict[str, float], a map from auxiliary 
                joint names to values
            
        Returns:
            full_config : np.ndarray of shape (B, DOF), a full configuration 
                vector including auxiliary joints.
        """
        if auxiliary_joint_values is None:
            auxiliary_joint_values = self.auxiliary_joint_defaults
        if config.ndim == 1:
            config = config[None, :]
        assert config.shape[1] == self.MAIN_DOF, "Input config dimension does not match MAIN_DOF"

        batch_size = config.shape[0]
        full_config = np.zeros((batch_size, self.DOF), dtype=config.dtype)
        input_config_idx = 0
        for i, joint in enumerate(self.urdf.actuated_joints):
            if joint.name in self.main_joint_names:
                full_config[:, i] = config[:, input_config_idx]
                input_config_idx += 1
            elif joint.name in self.auxiliary_joint_names:
                full_config[:, i] = auxiliary_joint_values[joint.name]
            else:
                raise ValueError(f"Unknown actuated joint: {joint.name}")
        assert input_config_idx == self.MAIN_DOF
        assert full_config.shape == (batch_size, self.DOF)
        
        return full_config

    def _torch_construct_full_config(self,
                                     config: torch.Tensor,
                                     auxiliary_joint_values: Optional[Dict[str, float]]=None) -> torch.Tensor:
        """
        Construct full configuration vector from main config and auxiliary 
        joint values.
        B: batch dimension.
        
        Args:
            config : torch.Tensor of shape (B, MAIN_DOF) or (MAIN_DOF,), a 
                configuration vector for the main joints.
            auxiliary_joint_values : Dict[str, float], a map from auxiliary 
                joint names to values
            
        Returns:
            full_config : torch.Tensor of shape (B, DOF), a full configuration 
                vector including auxiliary joints.
        """

        assert str(config.device) == str(self.device), f"Config device '{config.device}' does not match robot device '{self.device}'"

        if auxiliary_joint_values is None:
            auxiliary_joint_values = self.auxiliary_joint_defaults
        if config.dim() == 1:
            config = config.unsqueeze(0)
        assert config.shape[1] == self.MAIN_DOF, "Input config dimension does not match MAIN_DOF"

        batch_size = config.shape[0]
        full_config = torch.zeros((batch_size, self.DOF), dtype=config.dtype, device=config.device)
        input_config_idx = 0
        for i, joint in enumerate(self.urdf.actuated_joints):
            if joint.name in self.main_joint_names:
                full_config[:, i] = config[:, input_config_idx]
                input_config_idx += 1
            elif joint.name in self.auxiliary_joint_names:
                full_config[:, i] = auxiliary_joint_values[joint.name]
            else:
                raise ValueError(f"Unknown actuated joint: {joint.name}")
        assert input_config_idx == self.MAIN_DOF
        assert full_config.shape == (batch_size, self.DOF)
        
        return full_config

    def fk(self,
        config: np.ndarray,
        auxiliary_joint_values: Optional[Dict[str, float]]=None, 
        link_name: Optional[str]=None, 
        base_pose: np.ndarray=np.eye(4)
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Forward kinematics for specified link (numpy version).

        Args:
            config : np.ndarray of shape (batch_size, MAIN_DOF), an array 
                containing ``batch_size`` configuration vectors, each with an 
                entry for every main joint.
            auxiliary_joint_values : Dict[str, float], a map from auxiliary 
                joint names to values
            link_name : str, the name of the link to compute FK for. If None, 
                FK for all links are returned.
            base_pose : np.ndarray of shape (4, 4), the pose of the robot base 
                link
            
        Returns:
            fk_result : np.ndarray of shape (batch_size, 4, 4) (poses for given 
                link) or if link_name is None, a Dict[str, np.ndarray] with 
                np.ndarrays of shape (batch_size, 4, 4) (map with poses for all 
                links).
        """
        full_config = self._construct_full_config(config, auxiliary_joint_values)

        if link_name is None:
            fk_result = self.urdf.link_fk_batch(full_config, use_names=True)
            if not np.allclose(base_pose, np.eye(4)):
                for k in fk_result:
                    fk_result[k] = np.matmul(base_pose, fk_result[k])
            return fk_result

        link_fk_result = self.urdf.link_fk_batch(full_config, link=link_name, use_names=True)
        assert isinstance(link_fk_result, np.ndarray)
        if not np.allclose(base_pose, np.eye(4)):
            return np.matmul(base_pose, link_fk_result)
        return link_fk_result

    def fk_torch(self,
        config: torch.Tensor,
        auxiliary_joint_values: Optional[Dict[str, float]]=None,
        link_name: Optional[str]=None,
        base_pose: Optional[torch.Tensor]=None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward kinematics for specified link (torch version).

        Args:
            config : torch.Tensor of shape (batch_size, MAIN_DOF), a tensor 
                containing ``batch_size`` configuration vectors, each with an 
                entry for every main joint.
            auxiliary_joint_values : Dict[str, float], a map from auxiliary 
                joint names to values
            link_name : str, the name of the link to compute FK for. If None, 
                FK for all links are returned.
            base_pose : None or torch.Tensor of shape (4, 4), the pose of the 
                robot base link. If None, acts as if identity pose.

        Returns:
            fk_result : Tensor of shape (batch_size, 4, 4) (poses for given link) 
            or if link_name is None, a Dict[str, Tensor] with Tensors of shape 
            (batch_size, 4, 4) (map with poses for all links).
        """
        full_config = self._torch_construct_full_config(config, auxiliary_joint_values)

        fk_result = self.torch_urdf.link_fk_batch(full_config, use_names=True)
        if link_name is None:
            if base_pose is not None:
                for k in fk_result:
                    assert base_pose.device == fk_result[k].device
                    fk_result[k] = torch.matmul(base_pose, fk_result[k])
            return fk_result

        if base_pose is not None:
            assert base_pose.device == fk_result[link_name].device
            return torch.matmul(base_pose, fk_result[link_name])
        return fk_result[link_name]


    def visual_fk(self,
        config: np.ndarray,
        auxiliary_joint_values: Optional[Dict[str, float]]=None,
        link_name: Optional[str]=None,
        base_pose: np.ndarray=np.eye(4)
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Forward kinematics for all links with visual meshes (numpy version).
        
        Args:
            config : np.ndarray of shape (batch_size, MAIN_DOF), an array 
                containing ``batch_size`` configuration vectors, each with an 
                entry for every main joint.
            auxiliary_joint_values : Dict[str, float], a map from auxiliary 
                joint names to values
            link_name : str, the name of the link to compute FK for. If None, 
                FK for all links are returned.
            base_pose : np.ndarray of shape (4, 4), the pose of the robot base 
                link
        
        Returns:
            fk_result : np.ndarray of shape (batch_size, 4, 4) (poses for given 
                link) or if link_name is None, a Dict[str, np.ndarray] with 
                np.ndarrays of shape (batch_size, 4, 4) (map with poses for all 
                links).
        """
        full_config = self._construct_full_config(config, auxiliary_joint_values)

        fk_result = self.urdf.link_fk_batch(full_config)
        fk_visual_result = OrderedDict()
        for link in fk_result:
            assert len(link.visuals) <= 1
            for visual in link.visuals:
                key = link.name
                fk_visual_result[key] = np.matmul(
                    fk_result[link], visual.origin.astype(fk_result[link].dtype))

        if link_name is None:
            for k in fk_visual_result:
                fk_visual_result[k] = np.matmul(base_pose, fk_visual_result[k])
            return fk_visual_result
        return np.matmul(base_pose, fk_visual_result[link_name])

    def visual_fk_torch(self,
        config: torch.Tensor,
        auxiliary_joint_values: Optional[Dict[str, float]]=None,
        link_name: Optional[str]=None,
        base_pose: Optional[torch.Tensor]=None
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward kinematics for all links with visual meshes (torch version).
        
        Args:
            config : torch.Tensor of shape (batch_size, MAIN_DOF), a tensor 
                containing ``batch_size`` configuration vectors, each with an 
                entry for every main joint.
            auxiliary_joint_values : Dict[str, float], a map from auxiliary 
                joint names to values
            link_name : str, the name of the link to compute FK for. If None, 
                FK for all links are returned.
            base_pose : None or torch.Tensor of shape (4, 4), the pose of the robot base 
                link. If None, acts as if identity pose.
        
        Returns:
            fk_result : torch.Tensor of shape (batch_size, 4, 4) (poses for given 
                link) or if link_name is None, a Dict[str, torch.Tensor] with 
                tensors of shape (batch_size, 4, 4) (map with poses for all 
                links).
        """
        full_config = self._torch_construct_full_config(config, auxiliary_joint_values)
        fk_result = self.torch_urdf.visual_geometry_fk_batch(full_config, use_names=True)
    
        if link_name is None:
            if base_pose is not None:
                for k in fk_result:
                    assert base_pose.device == fk_result[k].device
                    fk_result[k] = torch.matmul(base_pose, fk_result[k])
            return fk_result
        
        if base_pose is not None:
            assert base_pose.device == fk_result[link_name].device
            return torch.matmul(base_pose, fk_result[link_name])
    
        return fk_result[link_name]

    def _get_joint_transform(self, joint: urchin.Joint, q: float) -> np.ndarray:
        """
        Homogeneous transform for a URDF joint at position q.

        Args:
            joint: urchin.Joint
            q: joint coordinate (rad or m)

        Returns:
            (4,4) SE(3) transform parent→child.
        """
        origin = joint.origin.astype(float)

        if joint.joint_type in ("revolute", "continuous"):
            axis = np.asarray(joint.axis, dtype=float)
            axis /= np.linalg.norm(axis)

            R = Rotation.from_rotvec(q * axis).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            return origin @ T

        if joint.joint_type == "prismatic":
            axis = np.asarray(joint.axis, dtype=float)
            axis /= np.linalg.norm(axis)

            T = np.eye(4)
            T[:3, 3] = q * axis
            return origin @ T

        if joint.joint_type == "fixed":
            return origin

        raise ValueError(
            f"Unsupported joint type '{joint.joint_type}'. "
            "Expected fixed, revolute, continuous, or prismatic."
        )

    def eef_fk(self,
        pose: np.ndarray,
        frame: Optional[str]=None,
        auxiliary_joint_values: Optional[Dict[str, float]]=None,
    ) -> Dict[str, np.ndarray]:
        """
        Forward kinematics for the end effector links (numpy version).

        B is batch dimension.

        NOTE: An assumption is made that any joints between end-effector links
        are part of the auxiliary joints. All main joints are assumed to be
        before any end-effector link in the kinematic chain.

        NOTE: Does not apply visual transform of the mesh as stored in the urdf

        Args:
            pose: np.ndarray of shape (B, 4, 4) or (4, 4), the pose of the link 
                with name `frame`
            frame: str, link that is the reference frame for the pose
            auxiliary_joint_values: Dict[str, float], a map from auxiliary 
                joint names to values

        Returns:
            fk_result: Dict[str, np.ndarray], keys are link names, values are 
            poses of shape (B, 4, 4) or (4, 4) (matches input). The absolute 
            transforms of the links belonging to the end effector.
        """
        if frame is None:
            frame = self.tcp_link_name
        if frame not in self.fixed_eef_link_transforms:
            raise ValueError(f"Frame {frame} is not a valid end effector frame (must be a link fixed to tcp)")
        if auxiliary_joint_values is None:
            auxiliary_joint_values = self.auxiliary_joint_defaults
        
        fk_result = {}
        fk_result[frame] = pose
        tcp_to_frame = self.fixed_eef_link_transforms[frame]
        tcp_absolute_pose = np.matmul(pose, np.linalg.inv(tcp_to_frame))
        for link_name, tcp_to_link in self.fixed_eef_link_transforms.items():
            if link_name == frame:
                continue
            fk_result[link_name] = np.matmul(tcp_absolute_pose, tcp_to_link)

        # add auxiliary jointed links ("fingers") to the fk_result
        for parent_link, joint_child_pairs in self.eef_aux_joints.items():
            if parent_link in fk_result:
                parent_pose = fk_result[parent_link]
                for joint, child_link in joint_child_pairs:
                    if joint.name in auxiliary_joint_values:
                        joint_value = auxiliary_joint_values[joint.name]
                        joint_transform = self._get_joint_transform(joint, 
                                                                    joint_value)
                        child_pose = np.matmul(parent_pose, joint_transform)
                        fk_result[child_link] = child_pose


        return fk_result

    def eef_fk_torch(self,
        pose: torch.Tensor,
        frame: Optional[str]=None,
        auxiliary_joint_values: Optional[Dict[str, float]]=None,
        only_visual: bool=False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward kinematics for the end effector links (torch version).

        NOTE: An assumption is made that any joints between end-effector links
        are part of the auxiliary joints. All main joints are assumed to be
        before any end-effector link in the kinematic chain.
        
        NOTE: Does not apply visual transform of the mesh as stored in the urdf

        Args:
            pose: torch.Tensor of shape (batch_size, 4, 4), the pose of the link with name `frame`
            frame: str, link that is the reference frame for the pose
            auxiliary_joint_values: Dict[str, float], a map from auxiliary 
                joint names to values
            only_visual: bool, if True, only return the poses for the visual links
        Returns:
            fk_result: Dict[str, torch.Tensor], keys are link names, values are 
            poses of shape (batch_size, 4, 4). The absolute transforms of the links 
            belonging to the end effector.
        """
        if frame is None:
            frame = self.tcp_link_name
        if frame not in self.torch_fixed_eef_link_transforms:
            raise ValueError(f"Frame {frame} is not a valid end effector frame (must be a link fixed to tcp)")
        if auxiliary_joint_values is None:
            auxiliary_joint_values = self.auxiliary_joint_defaults
        
        if pose.ndim == 2:
            pose = pose.unsqueeze(0)

        fk_result = {}
        fk_result[frame] = pose
        tcp_to_frame = self.torch_fixed_eef_link_transforms[frame].to(
            dtype=pose.dtype, device=pose.device)
        tcp_absolute_pose = torch.matmul(pose, torch.linalg.inv(tcp_to_frame)) # pylint: disable=not-callable
        for link_name, tcp_to_link in self.torch_fixed_eef_link_transforms.items():
            if only_visual and link_name not in self.eef_visual_link_names:
                continue
            if link_name == frame:
                continue
            fk_result[link_name] = torch.matmul(tcp_absolute_pose,
                                                tcp_to_link.to(dtype=pose.dtype, 
                                                               device=pose.device))

        # add auxiliary jointed links ("fingers") to the fk_result
        for parent_link, joint_child_pairs in self.eef_aux_joints.items():
            if parent_link in fk_result:
                parent_pose = fk_result[parent_link]
                for joint, child_link in joint_child_pairs:
                    if only_visual and child_link not in self.eef_visual_link_names:
                        continue
                    if joint.name in auxiliary_joint_values:
                        joint_value = auxiliary_joint_values[joint.name]
                        joint_transform = torch.as_tensor(
                            self._get_joint_transform(joint, joint_value), 
                            device=pose.device, dtype=pose.dtype)
                        child_pose = torch.matmul(parent_pose, joint_transform)
                        fk_result[child_link] = child_pose

        return fk_result

    def eef_visual_fk(self,
        pose: np.ndarray,
        frame: Optional[str]=None,
        auxiliary_joint_values: Optional[Dict[str, float]]=None
    ) -> Dict[str, np.ndarray]:
        """
        Forward kinematics for the end effector's links with visual meshes (numpy version).
        
        B is batch dimension.

        NOTE: Also applies visual transform of the mesh as stored in the urdf

        Args:
            pose: np.ndarray of shape (B, 4, 4) or (4, 4), the pose of the link with name `frame`
            frame: str, link that is the reference frame for the pose
            auxiliary_joint_values: Dict[str, float], a map from auxiliary 
                joint names to values

        Returns:
            fk_result: Dict[str, np.ndarray], keys are link names, values are 
            poses of shape (B, 4, 4) or (4, 4) (matches input). The absolute 
            transforms of the links belonging to the end effector with visual meshes.

        """
        fk_result = self.eef_fk(pose, frame, auxiliary_joint_values)
        fk_visual_result = {}
        for link_name, link_pose in fk_result.items():
            if link_name in self.eef_visual_link_names:
                vis_transform: np.ndarray = self.get_visual_transform(link_name)
                fk_visual_result[link_name] = np.matmul(link_pose, vis_transform)
        return fk_visual_result


    def eef_visual_fk_torch(self,
        pose: torch.Tensor,
        frame: Optional[str]=None,
        auxiliary_joint_values: Optional[Dict[str, float]]=None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward kinematics for the end effector's links with visual meshes (torch version).
        
        NOTE: Also applies visual transform of the mesh as stored in the urdf

        Args:
            pose: torch.Tensor of shape (batch_size, 4, 4), the pose of the link with name `frame`
            frame: str, link that is the reference frame for the pose
            auxiliary_joint_values: Dict[str, float], a map from auxiliary 
                joint names to values

        Returns:
            fk_result: Dict[str, torch.Tensor], keys are link names, values are 
            poses of shape (batch_size, 4, 4). The absolute transforms of the links with 
            visual meshes belonging to the end effector.
        """
        if frame is None:
            frame = self.tcp_link_name
        if frame not in self.torch_fixed_eef_link_transforms:
            raise ValueError(f"Frame {frame} is not a valid end effector frame (must be a link fixed to tcp)")

        fk_result = self.eef_fk_torch(pose, frame, auxiliary_joint_values)
        fk_visual_result = {}
        for link_name, link_pose in fk_result.items():
            if link_name in self.eef_visual_link_names:
                vis_transform: torch.Tensor = self.torch_get_visual_transform(
                    link_name, dtype=link_pose.dtype)
                fk_visual_result[link_name] = torch.matmul(link_pose, vis_transform)
        return fk_visual_result

    def ik(self, pose, fixed_joint_value, eff_frame=None, joint_range_scalar=1.0):
        """
        Generic IK not yet implemented for arbitrary robots.
        
        Args:
            pose: Target SE3 pose
            fixed_joint_value: Value for a specific joint to constrain IK solution
            eff_frame: Target end effector frame (defaults to configured TCP)
            joint_range_scalar: Scale factor for joint limits
        """
        if eff_frame is None:
            eff_frame = self.tcp_link_name
        print("Generic IK not yet implemented for arbitrary robots")
        return None

    def random_ik(self, pose, eff_frame=None, joint_range_scalar=1.0):
        """
        Generic random IK not yet implemented for arbitrary robots.
        """
        if eff_frame is None:
            eff_frame = self.tcp_link_name
        print("Generic random IK not yet implemented for arbitrary robots")
        return None

    def collision_free_ik(
        self,
        pose,
        auxiliary_joint_value,
        cooo,
        primitive_arrays,
        scene_buffer=0.0,
        self_collision_buffer=0.0,
        joint_range_scalar=1.0,
        eff_frame=None,
        retries=1000,
        bad_state_callback=lambda x: False,
        choose_close_to=None,
    ):
        """
        Generic collision-free IK not yet implemented for arbitrary robots.
        
        Args:
            pose: Target SE3 pose
            auxiliary_joint_value: Value for auxiliary joints (e.g., gripper opening)
            cooo: Collision checker object
            primitive_arrays: Collision primitive arrays
            scene_buffer: Buffer for scene collision detection
            self_collision_buffer: Buffer for self-collision detection
            joint_range_scalar: Scale factor for joint limits
            eff_frame: Target end effector frame (defaults to configured TCP)
            retries: Maximum number of attempts
            bad_state_callback: Function to check if state is invalid
            choose_close_to: If provided, choose solution closest to this config
        """
        if eff_frame is None:
            eff_frame = self.tcp_link_name
        print("Generic collision-free IK not yet implemented for arbitrary robots")
        return None

    def normalize_joints(self,
                         config: np.ndarray | torch.Tensor,
                         limits=(-1, 1)):
        """
        Normalizes joint angles to be within a specified range according to the 
        robot's joint limits.

        Args:
            config: Joint configuration as numpy array or torch.Tensor. 
                Can have dims:
                    [MAIN_DOF] if a single configuration
                    [B, MAIN_DOF] if a batch of configurations
                    [B, T, MAIN_DOF] if a batched time-series of configurations
            limits: Tuple of (min, max) values to normalize to, default (-1, 1)

        Returns:
            Normalized joint angles with same shape and type as input
        """
        if isinstance(config, torch.Tensor):
            input_dof_dim = 0 if config.dim() == 1 else (1 if config.dim() == 2 else 2)
            input_dof = config.size(input_dof_dim)
            assert input_dof == self.MAIN_DOF, f"Expected {self.MAIN_DOF} DOF, got {input_dof}"

            joint_limits = torch.tensor(
                self.main_joint_limits, dtype=config.dtype, device=config.device
            )

            joint_range = joint_limits[:, 1] - joint_limits[:, 0]
            joint_min = joint_limits[:, 0]

            if config.dim() > 1:
                for _ in range(config.dim() - 1):
                    joint_range = joint_range.unsqueeze(0)
                    joint_min = joint_min.unsqueeze(0)

            normalized = (config - joint_min) / joint_range
            normalized = normalized * (limits[1] - limits[0]) + limits[0]

            return normalized

        if not isinstance(config, np.ndarray):
            config = np.array(config)

        input_dof_dim = 0 if config.ndim == 1 else (1 if config.ndim == 2 else 2)
        input_dof = config.shape[input_dof_dim]
        assert input_dof == self.MAIN_DOF, f"Expected {self.MAIN_DOF} DOF, got {input_dof}"

        joint_limits = self.main_joint_limits

        joint_range = joint_limits[:, 1] - joint_limits[:, 0]
        joint_min = joint_limits[:, 0]

        if config.ndim > 1:
            for _ in range(config.ndim - 1):
                joint_range = joint_range[np.newaxis, ...]
                joint_min = joint_min[np.newaxis, ...]

        normalized = (config - joint_min) / joint_range
        normalized = normalized * (limits[1] - limits[0]) + limits[0]

        return normalized

    def unnormalize_joints(self,
                           config: np.ndarray | torch.Tensor,
                           limits=(-1, 1)):
        """
        Unnormalizes joint angles from a specified range back to the robot's 
        joint limits.

        Args:
            config: Normalized joint configuration as numpy array or 
                torch.Tensor. Can have dims:
                    [MAIN_DOF] if a single configuration
                    [B, MAIN_DOF] if a batch of configurations
                    [B, T, MAIN_DOF] if a batched time-series of configurations
            limits: Tuple of (min, max) values the config was normalized to, 
                default (-1, 1)

        Returns:
            Unnormalized joint angles within the robot's joint limits, with 
            same shape and type as input
        """

        if isinstance(config, torch.Tensor):
            input_dof_dim = 0 if config.dim() == 1 else (1 if config.dim() == 2 else 2)
            input_dof = config.size(input_dof_dim)
            assert input_dof == self.MAIN_DOF, f"Expected {self.MAIN_DOF} DOF, got {input_dof}"

            assert torch.all(
                (config >= limits[0]) & (config <= limits[1])
            ), f"Normalized values must be in range [{limits[0]}, {limits[1]}]"

            joint_limits = torch.tensor(
                self.main_joint_limits, dtype=config.dtype, device=config.device
            )

            joint_range = joint_limits[:, 1] - joint_limits[:, 0]
            joint_min = joint_limits[:, 0]

            if config.dim() > 1:
                for _ in range(config.dim() - 1):
                    joint_range = joint_range.unsqueeze(0)
                    joint_min = joint_min.unsqueeze(0)

            unnormalized = (config - limits[0]) / (limits[1] - limits[0])
            unnormalized = unnormalized * joint_range + joint_min

            return unnormalized

        if not isinstance(config, np.ndarray):
            config = np.array(config)

        input_dof_dim = 0 if config.ndim == 1 else (1 if config.ndim == 2 else 2)
        input_dof = config.shape[input_dof_dim]
        assert input_dof == self.MAIN_DOF, f"Expected {self.MAIN_DOF} DOF, got {input_dof}"

        assert np.all(
            (config >= limits[0]) & (config <= limits[1])
        ), f"Normalized values must be in range [{limits[0]}, {limits[1]}]"

        joint_limits = self.main_joint_limits

        joint_range = joint_limits[:, 1] - joint_limits[:, 0]
        joint_min = joint_limits[:, 0]

        if config.ndim > 1:
            for _ in range(config.ndim - 1):
                joint_range = joint_range[np.newaxis, ...]
                joint_min = joint_min[np.newaxis, ...]

        unnormalized = (config - limits[0]) / (limits[1] - limits[0])
        unnormalized = unnormalized * joint_range + joint_min

        return unnormalized
        
    def compute_spheres(self, config, auxiliary_joint_values=None):
        """
        Compute collision spheres for the robot in a given configuration. 
        Formatted for speedy sdf calculations.
        
        Args:
            config: np.ndarray of shape (batch_size, MAIN_DOF), the robot configuration
            auxiliary_joint_values: Dict[str, float], a map from auxiliary joint names to values
            device: torch.device, the device to use for computation (if using torch tensors)
            
        Returns:
            List of tuples (radius: float, sphere_centers: torch.Tensor of shape [batch_size, num_spheres, 3])
        """
        full_config = self._torch_construct_full_config(config, auxiliary_joint_values)
        
        fk = self.torch_urdf.link_fk_batch(full_config, use_names=True)
        points = []
        for radius, spheres in self.spheres:
            fk_points = []
            for link_name in spheres:
                pc = transform_point_cloud(
                    spheres[link_name]
                    .type_as(full_config)
                    .repeat((fk[link_name].shape[0], 1, 1)),
                    fk[link_name].type_as(full_config),
                    in_place=True,
                )
                fk_points.append(pc)
            points.append((radius, torch.cat(fk_points, dim=1)))
        return points
