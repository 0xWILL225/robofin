import numpy as np
import torch

import robofin.old.kinematics.numba as rkn
from old import samplers as samplers
from old.robot_constants import FrankaConstants
from old.robots import FrankaRobot
from robofin.torch_urdf import TorchURDF


def has_point(point_cloud: np.ndarray,
              point: np.ndarray,
              tolerance: float = 1e-6) -> tuple[bool, np.ndarray]:
    """
    Returns:
    exists, idx (bool, np.ndarray): Whether `point` exists in the point cloud
        and the indices in the point cloud array where the point exists.
    """
    mask = np.max(np.abs(point_cloud - point), axis=1) <= tolerance
    exists = mask.any()
    idx = np.flatnonzero(mask)
    return exists, idx

def compare_point_clouds(pc1: np.ndarray,
                         pc2: np.ndarray,
                         abs_tol: float=1e-7) -> bool:
    """
    Return True if input point clouds are identical
    """
    return np.allclose(pc1, pc2, atol=abs_tol)


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
    sampler1 = samplers.NumpyFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler2 = samplers.NumpyFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    samples1 = sampler1.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    samples2 = sampler2.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    assert compare_point_clouds(samples1, samples2)
    eef_samples1 = sampler1.sample_end_effector(
        FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix,
        0.04,
    )
    eef_samples2 = sampler2.sample_end_effector(
        FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix,
        0.04,
    )
    assert compare_point_clouds(eef_samples1, eef_samples2)

def test_deterministic_numpy_sampling_2048():
    sampler1 = samplers.NumpyFrankaSampler(
        num_robot_points=2048, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler2 = samplers.NumpyFrankaSampler(
        num_robot_points=2048, num_eef_points=128, use_cache=True, with_base_link=True
    )
    samples1 = sampler1.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    samples2 = sampler2.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    assert compare_point_clouds(samples1, samples2)
    eef_samples1 = sampler1.sample_end_effector(
        FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix,
        0.04,
    )
    eef_samples2 = sampler2.sample_end_effector(
        FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix,
        0.04,
    )
    assert compare_point_clouds(eef_samples1, eef_samples2)


def test_deterministic_torch_sampling():
    sampler1 = samplers.TorchFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler2 = samplers.TorchFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    samples1 = sampler1.sample(
        torch.as_tensor(FrankaConstants.NEUTRAL),
        0.04,
    )
    samples2 = sampler2.sample(
        torch.as_tensor(FrankaConstants.NEUTRAL),
        0.04,
    )
    assert compare_point_clouds(samples1.squeeze().numpy(), samples2.squeeze().numpy())
    eef_samples1 = sampler1.sample_end_effector(
        torch.as_tensor(FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix).float(),
        0.04,
    )
    eef_samples2 = sampler2.sample_end_effector(
        torch.as_tensor(FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix).float(),
        0.04,
    )
    assert compare_point_clouds(
        eef_samples1.squeeze().numpy(), eef_samples2.squeeze().numpy()
    )


def test_deterministic_compare():
    sampler1 = samplers.NumpyFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler2 = samplers.TorchFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )
    samples1 = sampler1.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    samples2 = sampler2.sample(
        torch.as_tensor(FrankaConstants.NEUTRAL),
        0.04,
    )

    assert isinstance(samples2, torch.Tensor)
    # for i, point in enumerate(samples1):
    #     if not has_point(samples2.squeeze().numpy(), point):
    #         print(f"The numpy point {i} is NOT in the torch points.")

    assert compare_point_clouds(samples1, samples2.squeeze().numpy())
    eef_samples1 = sampler1.sample_end_effector(
        FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix,
        0.04,
    )
    eef_samples2 = sampler2.sample_end_effector(
        torch.as_tensor(FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix).float(),
        0.04,
    )
    assert compare_point_clouds(eef_samples1, eef_samples2.squeeze().numpy())


def test_deterministic_gen_cache():
    sampler1 = samplers.NumpyFrankaSampler(
        num_robot_points=2048, num_eef_points=128, use_cache=True, with_base_link=True
    )
    samples1 = sampler1.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )

    sampler2 = samplers.NumpyFrankaSampler(
        num_robot_points=1024, num_eef_points=128, use_cache=True, with_base_link=True
    )
    sampler3 = samplers.TorchFrankaSampler(
        num_robot_points=4096, num_eef_points=128, use_cache=True, with_base_link=True
    )


if __name__ == "__main__":
    print("Testing original samplers...")
    
    print("Running test_fk...")
    test_fk()
    print("✓ test_fk passed")
    
    print("Running test_eef_fk...")
    test_eef_fk()
    print("✓ test_eef_fk passed")
    
    print("Running test_deterministic_numpy_sampling...")
    test_deterministic_numpy_sampling()
    print("✓ test_deterministic_numpy_sampling passed")
    
    print("Running test_nondeterministic_numpy_sampling...")
    test_deterministic_numpy_sampling_2048()
    print("✓ test_nondeterministic_numpy_sampling passed")

    print("Running test_deterministic_torch_sampling...")
    test_deterministic_torch_sampling()
    print("✓ test_deterministic_torch_sampling passed")
    
    print("Running test_deterministic_compare...")
    test_deterministic_compare()
    print("✓ test_deterministic_compare passed")
    
    print("Running test_deterministic_gen_cache...")
    test_deterministic_gen_cache()
    print("✓ test_deterministic_gen_cache passed")
    
    print("All original tests passed!") 