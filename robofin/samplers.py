import logging
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import numpy as np
import numba as nb
import torch
import trimesh
from geometrout.maths import transform_in_place

from robofin.point_cloud_tools import transform_point_cloud
from robofin.robots import Robot

@nb.jit(nopython=True, cache=True)
def label(array, lbl):
    """Used for adding label feature to points in point clouds"""
    return np.concatenate((array, lbl * np.ones((array.shape[0], 1))), axis=1)

class SamplerBase:
    """Base class for robot point cloud samplers."""
    def __init__(
        self,
        robot: Robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        cache_file_path: Optional[str | Path]=None,
        with_base_link=True,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.robot = robot
        self.with_base_link = with_base_link
        self.num_robot_points = num_robot_points
        self.num_eef_points = num_eef_points
        self.cache_file_path = cache_file_path

        if use_cache and self._init_from_cache_():
            return

        link_points, link_normals = self._initialize_robot_points(
            robot, num_robot_points
        )
        eef_points, eef_normals = self._initialize_eef_points_and_normals(
            robot, num_eef_points
        )
        self.points = {
            **link_points,
            **eef_points,
        }
        self.normals = {
            **link_normals,
            **eef_normals,
        }

        if use_cache:
            points_to_save = {}
            for key, pc in self.points.items():
                assert key in self.normals
                normals = self.normals[key]
                points_to_save[key] = {"pc": pc, "normals": normals}
            file_name = self._get_cache_file_name_()
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save) # type: ignore

    def _initialize_eef_points_and_normals(self, robot: Robot, N: int) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        # Use configured end effector visual links
        links = [
            link
            for link in robot.urdf.links
            if link.name in set(self.robot.eef_visual_link_names)
        ]
        
        meshes = []
        for link in links:
            mesh_filename = link.visuals[0].geometry.mesh.filename
            
            # Handle both absolute file URIs and relative paths
            if mesh_filename.startswith('file://'):
                # Remove file:// prefix and use absolute path
                mesh_path = mesh_filename[7:]  # Remove 'file://'
            else:
                # Relative path - prepend robot directory
                mesh_path = Path(self.robot.urdf_path).parent / mesh_filename
            
            meshes.append(trimesh.load(mesh_path, force="mesh"))
            
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        num_points = np.round(N * np.array(areas) / np.sum(areas))

        points = {}
        normals = {}
        for ii, mesh in enumerate(meshes):
            link_pc, face_indices = trimesh.sample.sample_surface( # type: ignore
                mesh, int(num_points[ii])
            )
            points[f"eef_{links[ii].name}"] = link_pc
            normals[f"eef_{links[ii].name}"] = self._init_normals(
                mesh, link_pc, face_indices
            )
        return points, normals

    def _initialize_robot_points(self, robot, N):
        # Use all links that have visual meshes
        links = [
            link
            for link in robot.urdf.links
            if len(link.visuals) > 0 and
               hasattr(link.visuals[0].geometry, 'mesh') and
               link.visuals[0].geometry.mesh is not None
        ]

        meshes = []
        for link in links:
            mesh_filename = link.visuals[0].geometry.mesh.filename
            
            # Handle both absolute file URIs and relative paths
            if mesh_filename.startswith('file://'):
                # Remove file:// prefix and use absolute path
                mesh_path = mesh_filename[7:]  # Remove 'file://'
            else:
                # Relative path - prepend robot directory
                mesh_path = Path(self.robot.urdf_path).parent / mesh_filename
            
            meshes.append(trimesh.load(mesh_path, force="mesh"))
            
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        num_points = np.round(N * np.array(areas) / np.sum(areas)).astype(int)
        rounding_error = N - np.sum(num_points)
        if rounding_error > 0:
            while rounding_error > 0:
                jj = np.random.choice(np.arange(len(num_points)))
                num_points[jj] += 1
                rounding_error = N - np.sum(num_points)
        elif rounding_error < 0:
            while rounding_error < 0:
                jj = np.random.choice(np.arange(len(num_points)))
                num_points[jj] -= 1
                rounding_error = N - np.sum(num_points)

        points = {}
        normals = {}
        for ii, mesh in enumerate(meshes):
            link_pc, face_indices = trimesh.sample.sample_surface(mesh, num_points[ii]) # type: ignore
            points[links[ii].name] = link_pc
            normals[f"{links[ii].name}"] = self._init_normals(
                mesh, link_pc, face_indices
            )
        return points, normals

    def _init_normals(self, mesh, pc, face_indices):
        bary = trimesh.triangles.points_to_barycentric(
            triangles=mesh.triangles[face_indices], points=pc
        )
        # interpolate vertex normals from barycentric coordinates
        normals = trimesh.unitize(
            (
                mesh.vertex_normals[mesh.faces[face_indices]]
                * trimesh.unitize(bary).reshape((-1, 3, 1)) # type: ignore
            ).sum(axis=1)
        )
        return normals

    def _get_cache_file_name_(self):
        if self.cache_file_path is not None:
            return Path(self.cache_file_path)

        return (
            self.robot.robot_directory / "point_cloud_cache" 
            / f"robot_point_cloud_{self.num_robot_points}_{self.num_eef_points}.npy"
        )

    def _init_from_cache_(self):
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.points = {key: v["pc"] for key, v in points.item().items()}
        self.normals = {key: v["normals"] for key, v in points.item().items()}
        return True

def get_points_on_robot_arm(robot: Robot,
                            config: np.ndarray,
                            num_points: int,
                            link_points: Dict[str, np.ndarray],
                            auxiliary_joint_values: Optional[Dict[str, float]]=None, 
                            base_pose: np.ndarray = np.eye(4)) -> np.ndarray:
    """
    Get points on the robot arm with the given joint configuration.

    Currently not batched.

    Deterministically samples all stored points if num_points = 0.

    Args:
        robot (Robot): The robot object.
        config (np.ndarray): The configuration of the robot.
        num_points (int): The number of points to sample.
        link_points (Dict[str, np.ndarray]): Dictionary mapping link names to 
            their points.
        auxiliary_joint_values (Dict[str, float]): Dictionary mapping auxiliary 
            joint names to their values.
        base_pose (np.ndarray, optional): The pose of the robot's base link. 
            Defaults to np.eye(4).

    Returns:
        np.ndarray: A numpy array of shape (N, 3) containing the sampled points.
    """
    fk = robot.visual_fk(config, auxiliary_joint_values, base_pose=base_pose)
    assert len(fk[next(iter(fk))].shape) == 3, "fk results expected to have batch dimension"
    assert fk[next(iter(fk))].shape[0] == 1, "get_points_on_robot_arm() currently only supports fk batches of size 1"

    point_list = [
        label(transform_in_place(np.copy(link_points[link_name]), 
                                 fk[link_name].squeeze()), float(i))
        for i, link_name in enumerate(robot.arm_visual_link_names)
    ]
    all_points = np.concatenate(point_list, axis=0)
    
    if num_points > 0:
        return all_points[np.random.choice(all_points.shape[0], num_points, replace=False), :]
    return all_points


def get_points_on_robot_arm_from_poses(robot: Robot,
                                       poses: Dict[str, np.ndarray],
                                       num_points: int,
                                       link_points: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Get points on the robot arm with the given link poses.

    Currently not batched.

    Deterministically samples all stored points if num_points = 0.

    Args:
        robot (Robot): The robot object.
        poses (np.ndarray): The set of link poses.
        num_points (int): The number of points to sample.
        link_points (Dict[str, np.ndarray]): Dictionary mapping link names to 
            their points.

    Returns:
        np.ndarray: A numpy array of shape (N, 3) containing the sampled points.
    """
    assert len(poses) == len(robot.arm_visual_link_names)
    assert poses[next(iter(poses))].shape == (4,4), "Input poses must be of shape (4,4)"

    point_list = [
        label(transform_in_place(np.copy(link_points[link_name]), poses[link_name]), float(i))
        for i, link_name in enumerate(robot.arm_visual_link_names)
    ]
    all_points = np.concatenate(point_list, axis=0)

    if num_points > 0:
        return all_points[
            np.random.choice(all_points.shape[0], num_points, replace=False), :
        ]
    return all_points

def get_points_on_robot_eef(robot: Robot,
                            pose: np.ndarray,
                            num_points: int,
                            link_points: Dict[str, np.ndarray],
                            auxiliary_joint_values: Optional[Dict[str, float]]=None,
                            frame: Optional[str]=None) -> np.ndarray:
    """
    Get points on the robot end effector.

    Deterministically samples all stored points if num_points = 0.

    Currently does not support a batch dimension that isn't 1.

    Args:
        robot (Robot): The robot object.
        pose (np.ndarray): Absolute pose of the link with name `frame`.
        num_points (int): The number of points to sample.
        auxiliary_joint_values (Dict[str, float]): Dictionary mapping auxiliary 
            joint names to their values.
        frame (str): End effector link name that has pose `pose`.
    """
    if pose.ndim == 3:
        assert pose.shape[0] <= 1, "Batch dim greater than 1 not supported"
        pose = pose.squeeze()
    assert pose.ndim == 2, "pose.ndim must be 2: (4,4)"
    fk = robot.eef_visual_fk(pose, frame, auxiliary_joint_values)

    point_list = [
        label(transform_in_place(np.copy(link_points["eef_" + link_name]), fk[link_name]), float(i))
        for i, link_name in enumerate(robot.eef_visual_link_names)
    ]
    all_points = np.concatenate(point_list, axis=0)
    if num_points > 0:
        return all_points[np.random.choice(all_points.shape[0], num_points, replace=False), :]
    return all_points



class NumpyRobotSampler(SamplerBase):
    """
    This class allows for fast point cloud sampling from the surface of a robot.
    """
    def __init__(self, robot: Robot, **kwargs):
        super().__init__(robot, **kwargs)
        self.robot = robot

    def sample(self,
        cfg: np.ndarray,
        auxiliary_joint_values: Optional[Dict[str, float]] = None,
        num_points: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample points from the full robot visual mesh surface. Link poses are
        calculated using forward kinematics, based on the input joint config.

        When num_points = None, all stored points are used, and they are sampled 
        deterministically (as stored/cached).
        """
        assert num_points is None or 0 < num_points <= self.num_robot_points
        return get_points_on_robot_arm(
            self.robot,
            cfg,
            num_points or 0,
            self.points,
            auxiliary_joint_values
        )

    def sample_from_poses(self,
        poses: Dict[str, np.ndarray],
        num_points: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample points from the robot arm visual mesh surface based on a batch 
        of link poses.

        When num_points = None, all stored points are used, and they are sampled 
        deterministically (as stored/cached).

        Args:
            poses (Dict[str, np.ndarray]): A batch of poses.
            num_points (Optional[int]): The number of points to sample.
        """
        assert num_points is None or 0 < num_points <= self.num_eef_points
        return get_points_on_robot_arm_from_poses(
            self.robot,
            poses,
            num_points or 0,
            self.points
        )

    def sample_end_effector(
        self,
        pose: np.ndarray,
        auxiliary_joint_values: Optional[Dict[str, float]] = None,
        frame: Optional[str] = None,
        num_points: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample points from the end effector visual mesh surface.

        When num_points = None, all stored points are used, and they are sampled 
        deterministically (as stored/cached).
        """
        assert num_points is None or 0 < num_points <= self.num_eef_points
        return get_points_on_robot_eef(
            self.robot,
            pose,
            num_points or 0,
            self.points,
            auxiliary_joint_values,
            frame,
        )


class TorchRobotSampler(SamplerBase):
    """
    This class allows for fast point cloud sampling from the surface of a robot.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces point clouds for each configuration by running FK on a subsample
    of the per-link point clouds that are established at initialization.

    """

    def __init__(
        self,
        robot: Robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        cache_file_path: Optional[str | Path]=None,
        with_base_link=True,
        device: str | torch.device="cpu",
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        super().__init__(robot,
                         num_robot_points,
                         num_eef_points,
                         use_cache,
                         cache_file_path,
                         with_base_link)
        self.robot = robot
        self.links = [l for l in self.robot.torch_urdf.links if len(l.visuals)]
        self.points = {
            key: torch.as_tensor(val).unsqueeze(0).to(device)
            for key, val in self.points.items()
        }
        self.normals = {
            key: torch.as_tensor(val).unsqueeze(0).to(device)
            for key, val in self.normals.items()
        }

    def end_effector_pose(self,
                          config,
                          auxiliary_joint_values: Optional[Dict[str, float]]=None, 
                          frame: Optional[str]=None) -> torch.Tensor:
        """
        Return the pose of the end effector link `frame` when the robot's 
        configuration is `config`. Return tensor has batch dimension.
        """
        if frame is None:
            frame = self.robot.tcp_link_name
        assert frame in self.robot.fixed_eef_link_transforms, "Other frames not yet suppported"
        if config.ndim == 1:
            config = config.unsqueeze(0)

        eef_pose = self.robot.fk_torch(config, auxiliary_joint_values, link_name=frame)
        assert isinstance(eef_pose, torch.Tensor)
        return eef_pose

    def _sample_end_effector(
        self,
        with_normals: bool,
        pose: torch.Tensor,
        auxiliary_joint_values: Optional[Dict[str, float]]=None,
        num_points: Optional[int]=None,
        frame: Optional[str]=None,
        in_place: bool=True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample points from the end effector based on its pose.
        Args:
            with_normals (bool): Whether to return normals (returns tuple if so).
            pose (torch.Tensor): The pose of the end effector link in `frame`.
                Batched tensor of shape (B, 4, 4).
            auxiliary_joint_values (Dict[str, float]): Dictionary mapping 
                auxiliary joint names to their values.
            num_points (Optional[int]): The number of points to sample.
            frame (Optional[str]): The link frame in which the pose is defined.
            only_eff (bool): Whether to sample only the end effector.
            in_place (bool): Whether to perform the transformation in place.
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: A tensor of 
                shape (B, N, 4) (feature is [x,y,z,link_idx]) containing the 
                sampled points. If with_normals is True, also returns the 
                mesh normals at the sampled points.
        """
        assert num_points is None or 0 < num_points <= self.num_eef_points
        assert pose.ndim in [2, 3]
        
        if pose.ndim == 2:
            pose = pose.unsqueeze(0)
        
        visual_eef_fk = self.robot.eef_visual_fk_torch(pose, frame, auxiliary_joint_values)

        fk_points = []
        if with_normals:
            fk_normals = []
        
        for link_idx, link_name in enumerate(visual_eef_fk):
            pc = transform_point_cloud(
                self.points[f"eef_{link_name}"].float().repeat((pose.shape[0], 1, 1)),
                visual_eef_fk[link_name].float(),
                in_place=in_place,
            )
            fk_points.append(
                torch.cat(
                    (
                        pc,
                        link_idx * torch.ones((pc.size(0), pc.size(1), 1)).type_as(pc), # type: ignore
                    ), # type: ignore
                    dim=-1,
                ) # type: ignore
            )
            if with_normals:
                normals = transform_point_cloud(
                    self.normals[f"eef_{link_name}"]
                    .float()
                    .repeat((pose.shape[0], 1, 1)),
                    visual_eef_fk[link_name].float(),
                    vector=True,
                    in_place=in_place,
                )

                fk_normals.append(
                    torch.cat(
                        (
                            normals,
                            link_idx
                            * torch.ones((normals.size(0), normals.size(1), 1)).type_as( # type: ignore
                                normals # type: ignore
                            ), 
                        ), # type: ignore
                        dim=-1,
                    )
                )
        pc = torch.cat(fk_points, dim=1)
        if with_normals:
            normals = torch.cat(fk_normals, dim=1)
        if num_points is None:
            if with_normals:
                return pc, normals # type: ignore
            else:
                return pc
        sample_idxs = np.random.choice(pc.shape[1], num_points, replace=False)
        if with_normals:
            return (pc[:, sample_idxs, :], normals[:, sample_idxs, :]) # type: ignore
        return pc[:, sample_idxs, :]

    def sample_end_effector_with_normals(
        self,
        pose,
        auxiliary_joint_values: Optional[Dict[str, float]] = None,
        num_points: Optional[int]=None,
        frame: Optional[str]=None,
    ):
        return self._sample_end_effector(
            with_normals=True,
            pose=pose,
            auxiliary_joint_values=auxiliary_joint_values,
            num_points=num_points,
            frame=frame,
        )

    def sample_end_effector(
        self,
        pose: torch.Tensor,
        auxiliary_joint_values: Optional[Dict[str, float]] = None,
        num_points: Optional[int]=None,
        frame: Optional[str]=None,
        in_place: bool=True,
    ):
        return self._sample_end_effector(
            with_normals=False,
            pose=pose,
            auxiliary_joint_values=auxiliary_joint_values,
            num_points=num_points,
            frame=frame,
            in_place=in_place,
        )

    def _sample_from_poses(self,
                           with_normals: bool,
                           poses: Dict[str, torch.Tensor],
                           num_points: Optional[int]=None,
                           in_place: bool=True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample points from the robot arm based on a a set of link poses 
        (no FK needed).
        Args:
            with_normals (bool): Whether to return normals (returns tuple if so).
            poses (Dict[str, torch.Tensor]): The configuration of the robot.
            auxiliary_joint_values (Dict[str, float]): Dictionary mapping 
                auxiliary joint names to their values.
            num_points (Optional[int]): The number of points to sample.
            in_place (bool): Whether to perform the transformation in place.
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: A tensor of 
                shape (B, N, 4) (feature is [x,y,z,link_idx]) containing the 
                sampled points. If with_normals is True, also returns the 
                mesh normals at the sampled points.
        """
        points = []
        if with_normals:
            normals = []
        for link_idx, link_name in enumerate(self.robot.arm_visual_link_names):
            if not self.with_base_link and link_name == self.robot.base_link_name:
                continue
            pc = transform_point_cloud(
                self.points[link_name]
                .float()
                .repeat((poses[link_name].shape[0], 1, 1)),
                poses[link_name].float(),
                in_place=in_place,
            )
            points.append(
                torch.cat(
                    (
                        pc,
                        link_idx * torch.ones((pc.size(0), pc.size(1), 1)).type_as(pc), # type: ignore
                    ), # type: ignore
                    dim=-1, # type: ignore
                )
            )
            if with_normals:
                normals = transform_point_cloud(
                    self.normals[link_name]
                    .float()
                    .repeat((poses[link_name].shape[0], 1, 1)),
                    poses[link_name].float(),
                    vector=True,
                    in_place=in_place,
                )
                normals.append( # type: ignore
                    torch.cat(
                        (
                            normals,
                            link_idx
                            * torch.ones((normals.size(0), normals.size(1), 1)).type_as( # type: ignore
                                normals # type: ignore
                            ),
                        ), # type: ignore
                        dim=-1,
                    ) # type: ignore
                )
        pc = torch.cat(points, dim=1)
        if with_normals:
            normals = torch.cat(normals, dim=1) # type: ignore
        if num_points is None:
            if with_normals:
                return pc, normals # type: ignore
            return pc
        random_idxs = np.random.choice(pc.shape[1], num_points, replace=False)
        if with_normals:
            return pc[:, random_idxs, :], normals[:, random_idxs, :] # type: ignore
        return pc[:, random_idxs, :]

    def _sample(
        self,
        with_normals: bool,
        config: torch.Tensor,
        auxiliary_joint_values: Optional[Dict[str, float]]=None,
        num_points=None,
        only_eff=False,
        in_place=True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample points from the robot arm based on a configuration.
        Args:
            with_normals (bool): Whether to return normals (returns tuple if so).
            config (torch.Tensor): The configuration of the robot.
            auxiliary_joint_values (Dict[str, float]): Dictionary mapping 
                auxiliary joint names to their values.
            num_points (Optional[int]): The number of points to sample.
            only_eff (bool): Whether to sample only the end effector.
            in_place (bool): Whether to perform the transformation in place.
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: A tensor of 
                shape (B, N, 4) (feature is [x,y,z,link_idx]) containing the 
                sampled points. If with_normals is True, also returns the 
                mesh normals at the sampled points.
        """
        if config.ndim == 1:
            config = config.unsqueeze(0)

        fk = self.robot.visual_fk_torch(config, auxiliary_joint_values)
        fk_points = []
        if with_normals:
            fk_normals = []
        for link_idx, link_name in enumerate(self.robot.arm_visual_link_names):
            if only_eff and link_name not in self.robot.eef_visual_link_names:
                continue
            if not self.with_base_link and link_name == self.robot.base_link_name:
                continue
            pc = transform_point_cloud(
                self.points[link_name].float().repeat((fk[link_name].shape[0], 1, 1)),
                fk[link_name].float(),
                in_place=in_place,
            )
            fk_points.append(
                torch.cat( # type: ignore
                    (
                        pc,
                        link_idx * torch.ones((pc.size(0), pc.size(1), 1)).type_as(pc), # type: ignore
                    ),
                    dim=-1,
                )
            )
            if with_normals:
                normals = transform_point_cloud(
                    self.normals[link_name]
                    .float()
                    .repeat((fk[link_name].shape[0], 1, 1)),
                    fk[link_name].float(),
                    vector=True,
                    in_place=in_place,
                )
                fk_normals.append(
                    torch.cat( # type: ignore
                        (
                            normals,
                            link_idx
                            * torch.ones((normals.size(0), normals.size(1), 1)).type_as( # type: ignore
                                normals # type: ignore
                            ),
                        ),
                        dim=-1,
                    )
                )
        pc = torch.cat(fk_points, dim=1)
        if with_normals:
            normals = torch.cat(fk_normals, dim=1)
        if num_points is None:
            if with_normals:
                return pc, normals # type: ignore
            return pc
        random_idxs = np.random.choice(pc.shape[1], num_points, replace=False)
        if with_normals:
            return pc[:, random_idxs, :], normals[:, random_idxs, :] # type: ignore
        return pc[:, random_idxs, :]

    def sample(
        self,
        config,
        auxiliary_joint_values: Optional[Dict[str, float]] = None,
        num_points=None,
        only_eff=False,
        in_place=True
    ):
        return self._sample(
            with_normals=False,
            config=config,
            auxiliary_joint_values=auxiliary_joint_values,
            num_points=num_points,
            only_eff=only_eff,
            in_place=in_place,
        )

    def sample_from_poses(self, poses, num_points=None, in_place=True):
        return self._sample_from_poses(False, poses, num_points=num_points, in_place=in_place)

    def sample_with_normals(
        self,
        config,
        auxiliary_joint_values: Optional[Dict[str, float]]=None,
        num_points=None,
        only_eff=False,
        in_place=True
    ):
        return self._sample(
            with_normals=True,
            config=config,
            auxiliary_joint_values=auxiliary_joint_values,
            num_points=num_points,
            only_eff=only_eff,
            in_place=in_place,
        )
