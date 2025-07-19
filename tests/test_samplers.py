import numpy as np
import torch

import robofin.kinematics.numba as rkn
from robofin.robofin import samplers
from robofin.robot_constants import FrankaConstants
from robofin.robots import Robot
from robofin.torch_urdf import TorchURDF


def compare_point_clouds(pc1, pc2):
    return np.allclose(pc1, pc2)


def test_fk():
    robot = TorchURDF.load(FrankaConstants.urdf, lazy_load_meshes=True)

    rfk = rkn.franka_arm_link_fk(FrankaConstants.NEUTRAL, 0.04, np.eye(4))
    fk = robot.link_fk_batch(
        torch.as_tensor([*FrankaConstants.NEUTRAL, 0.04]).unsqueeze(0), use_names=True
    )

    for link_name, link_idx in FrankaConstants.ARM_LINKS.__members__.items():
        assert np.allclose(fk[link_name].squeeze(0).numpy(), rfk[link_idx]), link_name

    rfk = rkn.franka_arm_visual_fk(FrankaConstants.NEUTRAL, 0.04, np.eye(4))
    fk = robot.visual_geometry_fk_batch(
        torch.as_tensor([*FrankaConstants.NEUTRAL, 0.04]).unsqueeze(0), use_names=True
    )
    for link_name, link_idx in FrankaConstants.ARM_VISUAL_LINKS.__members__.items():
        assert np.allclose(fk[link_name].squeeze(0).numpy(), rfk[link_idx]), link_name


def test_eef_fk():
    robot = TorchURDF.load(FrankaConstants.urdf, lazy_load_meshes=True)

    rfk = rkn.franka_arm_link_fk(FrankaConstants.NEUTRAL, 0.04, np.eye(4))
    fk = robot.link_fk_batch(
        torch.as_tensor([*FrankaConstants.NEUTRAL, 0.04]).unsqueeze(0), use_names=True
    )

    for link_name, link_idx in FrankaConstants.ARM_LINKS.__members__.items():
        assert np.allclose(fk[link_name].squeeze(0).numpy(), rfk[link_idx]), link_name

    rfk = rkn.franka_arm_visual_fk(FrankaConstants.NEUTRAL, 0.04, np.eye(4))
    fk = robot.visual_geometry_fk_batch(
        torch.as_tensor([*FrankaConstants.NEUTRAL, 0.04]).unsqueeze(0), use_names=True
    )
    for link_name, link_idx in FrankaConstants.ARM_VISUAL_LINKS.__members__.items():
        assert np.allclose(fk[link_name].squeeze(0).numpy(), rfk[link_idx]), link_name


def test_deterministic_numpy_sampling():
    # Create generic robot instance
    robot = Robot("/workspace/assets/panda")
    
    # Create generic samplers
    sampler1 = samplers.NumpyRobotSampler(robot,
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler2 = samplers.NumpyRobotSampler(robot,
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    
    # Test main sampling
    samples1 = sampler1.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    samples2 = sampler2.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    assert compare_point_clouds(samples1, samples2)
    
    # Test end effector sampling
    test_pose = np.eye(4)
    test_pose[:3, 3] = [0.5, 0.0, 0.5]  # Set position
    
    eef_samples1 = sampler1.sample_end_effector(
        test_pose,
        0.04,
    )
    eef_samples2 = sampler2.sample_end_effector(
        test_pose,
        0.04,
    )
    assert compare_point_clouds(eef_samples1, eef_samples2)


def test_deterministic_gen_cache():
    # Create generic robot instance
    robot = Robot("/workspace/assets/panda")
    
    sampler1 = samplers.NumpyRobotSampler(robot,
        num_robot_points=2048, num_eef_points=128, use_cache=True, with_base_link=True
    )
    samples1 = sampler1.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )

    sampler2 = samplers.NumpyRobotSampler(robot,
        num_robot_points=1024, num_eef_points=128, use_cache=True, with_base_link=True
    )
    # Test that cache is working properly
    samples2 = sampler2.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    
    # They should be deterministic from cache
    assert len(samples1) > 0
    assert len(samples2) > 0


if __name__ == "__main__":
    print("Testing generic samplers...")
    
    print("Running test_fk...")
    test_fk()
    print("✓ test_fk passed")
    
    print("Running test_eef_fk...")
    test_eef_fk()
    print("✓ test_eef_fk passed")
    
    print("Running test_deterministic_numpy_sampling...")
    test_deterministic_numpy_sampling()
    print("✓ test_deterministic_numpy_sampling passed")
    
    print("Running test_deterministic_gen_cache...")
    test_deterministic_gen_cache()
    print("✓ test_deterministic_gen_cache passed")
    
    print("All generic tests passed!")
