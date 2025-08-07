import numpy as np
import torch

import robofin.kinematics.numba as rkn
from robofin import samplers
from robofin.robot_constants import FrankaConstants
from robofin.robots import Robot
from robofin.torch_urdf import TorchURDF


robot = Robot("/workspace/assets/panda/panda.urdf")

def compare_point_clouds(pc1: np.ndarray,
                         pc2: np.ndarray,
                         abs_tol: float=1e-7) -> bool:
    """
    Return True if input point clouds are identical (within given tolerance)
    """
    return np.allclose(pc1, pc2, atol=abs_tol)

def test_fk():
    """
    Test that numpy and torch forward kinematics give the same results.
    """

    np_fk = robot.fk(robot.neutral_config)
    torch_fk = robot.fk_torch(torch.as_tensor(robot.neutral_config))

    for link_name in robot.arm_link_names:
        assert np.allclose(torch_fk[link_name].squeeze(0).numpy(),
                           np_fk[link_name]), link_name

    np_visual_fk = robot.visual_fk(robot.neutral_config)
    torch_visual_fk = robot.visual_fk_torch(torch.as_tensor(robot.neutral_config))

    for link_name in robot.arm_visual_link_names:
        assert np.allclose(torch_visual_fk[link_name].squeeze(0).numpy(),
                           np_visual_fk[link_name]), link_name


def test_eef_fk():
    """
    Test that numpy and torch end-effector forward kinematics give the same results.
    """
    np_fk = robot.fk(robot.neutral_config)
    eef_pose = np_fk[robot.tcp_link_name]

    eef_np_fk = robot.eef_fk(eef_pose)
    eef_torch_fk = robot.eef_fk_torch(torch.as_tensor(eef_pose))

    for link_name in robot.eef_link_names:
        assert np.allclose(eef_torch_fk[link_name].squeeze(0).numpy(), 
                           eef_np_fk[link_name]), link_name

    eef_np_visual_fk = robot.eef_visual_fk(eef_pose)
    eef_torch_visual_fk = robot.eef_visual_fk_torch(torch.as_tensor(eef_pose))

    for link_name in robot.eef_visual_link_names:
        assert np.allclose(eef_torch_visual_fk[link_name].squeeze(0).numpy(), 
                           eef_np_visual_fk[link_name]), link_name


def test_deterministic_numpy_sampling():
    """ 
    Test that two samplers with the same parameters produce the same samples.
    """
    sampler1 = samplers.NumpyRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True
    )
    sampler2 = samplers.NumpyRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True
    )
    samples1 = sampler1.sample(robot.neutral_config)
    samples2 = sampler2.sample(robot.neutral_config)
    assert compare_point_clouds(samples1, samples2)
    test_pose = np.eye(4)
    test_pose[:3, 3] = [0.5, 0.0, 0.5]
    eef_samples1 = sampler1.sample_end_effector(test_pose)
    eef_samples2 = sampler2.sample_end_effector(test_pose)
    assert compare_point_clouds(eef_samples1, eef_samples2)



def test_deterministic_torch_sampling():
    sampler1 = samplers.TorchRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True
    )
    sampler2 = samplers.TorchRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True
    )
    samples1 = sampler1.sample(torch.as_tensor(robot.neutral_config))
    samples2 = sampler2.sample(torch.as_tensor(robot.neutral_config))
    assert isinstance(samples1, torch.Tensor) and isinstance(samples2, torch.Tensor)
    assert compare_point_clouds(samples1.squeeze().numpy(), samples2.squeeze().numpy())

    neutral_fk = robot.fk_torch(torch.as_tensor(robot.neutral_config))
    eef_pose = neutral_fk[robot.tcp_link_name]

    eef_samples1 = sampler1.sample_end_effector(eef_pose)
    eef_samples2 = sampler2.sample_end_effector(eef_pose)
    assert isinstance(eef_samples1, torch.Tensor) and isinstance(eef_samples2, torch.Tensor)
    assert compare_point_clouds(
        eef_samples1.squeeze().numpy(), eef_samples2.squeeze().numpy()
    )


def test_deterministic_compare():
    sampler1 = samplers.NumpyRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True
    )
    sampler2 = samplers.TorchRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True
    )
    samples1 = sampler1.sample(robot.neutral_config)
    samples2 = sampler2.sample(torch.as_tensor(robot.neutral_config))
    assert isinstance(samples2, torch.Tensor)
    assert compare_point_clouds(samples1, samples2.squeeze().numpy())

    neutral_fk = robot.fk(robot.neutral_config)
    eef_pose = neutral_fk[robot.tcp_link_name]

    eef_samples1 = sampler1.sample_end_effector(eef_pose)
    eef_samples2 = sampler2.sample_end_effector(torch.as_tensor(eef_pose))
    assert isinstance(eef_samples2, torch.Tensor)
    assert compare_point_clouds(eef_samples1, eef_samples2.squeeze().numpy())


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

    print("Running test_deterministic_torch_sampling...")
    test_deterministic_torch_sampling()
    print("✓ test_deterministic_torch_sampling passed")
    
    print("Running test_deterministic_compare...")
    test_deterministic_compare()
    print("✓ test_deterministic_compare passed")
    
    print("All generic tests passed!")