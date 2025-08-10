"""
Script for testing the output of forward kinematics and point sampling
functions. Ensures determinisim when it is expected, and that numpy and
pytorch based implementations return the same thing.
"""
import numpy as np
import torch

from robofin import samplers
from robofin.robots import Robot


robot = Robot("/workspace/assets/panda/panda.urdf")

def compare_point_clouds(
    pc1: np.ndarray, pc2: np.ndarray, abs_tol: float = 1e-7
) -> bool:
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
        assert np.allclose(
            torch_fk[link_name].squeeze(0).numpy(), np_fk[link_name]
        ), link_name

    np_visual_fk = robot.visual_fk(robot.neutral_config)
    torch_visual_fk = robot.visual_fk_torch(torch.as_tensor(robot.neutral_config))

    for link_name in robot.arm_visual_link_names:
        assert np.allclose(
            torch_visual_fk[link_name].squeeze(0).numpy(), np_visual_fk[link_name]
        ), link_name


def test_eef_fk():
    """
    Test that numpy and torch end-effector forward kinematics give the same results.
    """
    np_fk = robot.fk(robot.neutral_config)
    eef_pose = np_fk[robot.tcp_link_name]

    eef_np_fk = robot.eef_fk(eef_pose)
    eef_torch_fk = robot.eef_fk_torch(torch.as_tensor(eef_pose))

    for link_name in robot.eef_link_names:
        assert np.allclose(
            eef_torch_fk[link_name].squeeze(0).numpy(), eef_np_fk[link_name]
        ), link_name

    eef_np_visual_fk = robot.eef_visual_fk(eef_pose)
    eef_torch_visual_fk = robot.eef_visual_fk_torch(torch.as_tensor(eef_pose))

    for link_name in robot.eef_visual_link_names:
        assert np.allclose(
            eef_torch_visual_fk[link_name].squeeze(0).numpy(),
            eef_np_visual_fk[link_name],
        ), link_name


def test_deterministic_numpy_sampling():
    """
    Test that two samplers with the same parameters produce the same samples.
    """
    sampler1 = samplers.NumpyRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True,
    )
    sampler2 = samplers.NumpyRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True,
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
    """
    Test that two torch samplers with the same parameters produce the same samples.
    """
    sampler1 = samplers.TorchRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True,
    )
    sampler2 = samplers.TorchRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True,
    )
    samples1 = sampler1.sample(torch.as_tensor(robot.neutral_config))
    samples2 = sampler2.sample(torch.as_tensor(robot.neutral_config))
    assert isinstance(samples1, torch.Tensor) and isinstance(samples2, torch.Tensor)
    assert compare_point_clouds(samples1.squeeze().numpy(), samples2.squeeze().numpy())

    neutral_fk = robot.fk_torch(torch.as_tensor(robot.neutral_config))
    eef_pose = neutral_fk[robot.tcp_link_name]

    eef_samples1 = sampler1.sample_end_effector(eef_pose)
    eef_samples2 = sampler2.sample_end_effector(eef_pose)
    assert isinstance(eef_samples1, torch.Tensor) and isinstance(
        eef_samples2, torch.Tensor
    )
    assert compare_point_clouds(
        eef_samples1.squeeze().numpy(), eef_samples2.squeeze().numpy()
    )


def test_deterministic_compare():
    """
    Test that torch and numpy based samplers produce the same results when 
    sampling deterministically, using the same cache.
    """
    sampler1 = samplers.NumpyRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True,
    )
    sampler2 = samplers.TorchRobotSampler(
        robot,
        num_robot_points=4096,
        num_eef_points=128,
        use_cache=True,
        with_base_link=True,
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


from geometrout.transform import SE3

from robofin.old import samplers as old_samplers
from robofin.old.robot_constants import FrankaConstants
from robofin.old.robots import FrankaRobot
from robofin.old.samplers import TorchFrankaCollisionSampler

franka_robot = Robot("/workspace/assets/panda/panda.urdf")

def test_compare_with_original_samplers():
    """
    Test that the old Franka Panda specific samplers produce identical results
    to the new generic implementation, given that the franka panda urdf is used
    to instanciate the robot class, and that the same cache is used.
    """

    n_robot_pts = 4096
    n_eef_pts = 128

    franka_cache_file_path = FrankaConstants.point_cloud_cache / f"franka_point_cloud_{n_robot_pts}_{n_eef_pts}.npy"
    np_sampler = samplers.NumpyRobotSampler(
        franka_robot,
        num_robot_points=n_robot_pts,
        num_eef_points=n_eef_pts,
        use_cache=True,
        cache_file_path=franka_cache_file_path,
        with_base_link=True,
    )
    torch_sampler = samplers.TorchRobotSampler(
        franka_robot,
        num_robot_points=n_robot_pts,
        num_eef_points=n_eef_pts,
        use_cache=True,
        cache_file_path=franka_cache_file_path,
        with_base_link=True,
    )
    old_np_sampler = old_samplers.NumpyFrankaSampler(
        num_robot_points=n_robot_pts,
        num_eef_points=n_eef_pts,
        use_cache=True,
        with_base_link=True
    )
    old_torch_sampler = old_samplers.TorchFrankaSampler(
        num_robot_points=n_robot_pts,
        num_eef_points=n_eef_pts,
        use_cache=True,
        with_base_link=True
    )

    np_samples = np_sampler.sample(franka_robot.neutral_config)
    torch_samples = torch_sampler.sample(torch.as_tensor(franka_robot.neutral_config))
    old_np_samples = old_np_sampler.sample(
        FrankaConstants.NEUTRAL,
        0.04,
    )
    old_torch_samples = old_torch_sampler.sample(
        torch.as_tensor(FrankaConstants.NEUTRAL),
        0.04,
    )
    assert isinstance(torch_samples, torch.Tensor)
    assert isinstance(old_torch_samples, torch.Tensor)
    assert compare_point_clouds(np_samples, old_np_samples)
    assert compare_point_clouds(
        torch_samples.squeeze().numpy(), old_torch_samples.squeeze().numpy()
    )

    neutral_fk = franka_robot.fk(franka_robot.neutral_config)
    eef_pose = neutral_fk[franka_robot.tcp_link_name]
    old_eef_pose = FrankaRobot.fk(FrankaConstants.NEUTRAL).matrix

    assert np.allclose(eef_pose, old_eef_pose)

    frame = franka_robot.tcp_link_name
    np_eef_samples = np_sampler.sample_end_effector(eef_pose, frame=frame)
    torch_eef_samples = torch_sampler.sample_end_effector(torch.as_tensor(eef_pose), frame=frame)
    old_np_eef_samples = old_np_sampler.sample_end_effector(old_eef_pose, 0.04, frame=frame)
    old_torch_eef_samples = old_torch_sampler.sample_end_effector(
        torch.as_tensor(old_eef_pose).float(), 0.04, frame=frame
    )
    assert isinstance(torch_eef_samples, torch.Tensor)
    assert isinstance(old_torch_eef_samples, torch.Tensor)
    assert compare_point_clouds(np_eef_samples, old_np_eef_samples)
    assert compare_point_clouds(
        torch_eef_samples.squeeze().numpy(), old_torch_eef_samples.squeeze().numpy()
    )


def test_compare_fk_with_franka_robot():
    frame = franka_robot.tcp_link_name
    franka_eef_pose_se3: SE3 = FrankaRobot.fk(FrankaConstants.NEUTRAL, eff_frame=frame)
    franka_eef_pose = franka_eef_pose_se3.matrix
    eef_pose = franka_robot.fk(franka_robot.neutral_config, link_name=frame)

    assert isinstance(eef_pose, np.ndarray)
    assert np.allclose(franka_eef_pose, eef_pose.squeeze())

from math import isclose

def test_compare_compute_spheres_with_original():
    device = franka_robot.device
    c_sampler = TorchFrankaCollisionSampler(device)

    batch_dim = 5
    torch_neutral_config = torch.tensor(franka_robot.neutral_config,
                                        dtype=torch.float32,
                                        device=device)
    batched_configs = torch_neutral_config.unsqueeze(0).repeat(batch_dim, 1).to(device)
    prismatic_joint = 0.04
    auxiliary_joint_values = {'panda_finger_joint1': 0.04, 'panda_finger_joint2': 0.04}

    franka_collision_spheres = c_sampler.compute_spheres(batched_configs, 
                                                         prismatic_joint=prismatic_joint)
    generic_collision_spheres = robot.compute_spheres(batched_configs, 
                                                      auxiliary_joint_values)

    # check all sphere radii groups except the first two, which I altered
    for (franka_radius, franka_spheres), (radius, spheres) in zip(franka_collision_spheres[2:],
                                                                  generic_collision_spheres[2:]):
        # franka_spheres and spheres: torch.Tensor [B, num_spheres, 3], 3-dim is x,y,z
        assert isclose(franka_radius, radius), f"franka_radius: {franka_radius} != radius: {radius}"
        assert franka_spheres.shape == spheres.shape, f"franka_spheres.shape: {franka_spheres.shape} != spheres.shape: {spheres.shape}"
        assert torch.allclose(franka_spheres, spheres), "franka_spheres != spheres"

if __name__ == "__main__":
    print("Testing samplers...")

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

    print("Running test_compare_with_original_samplers...")
    test_compare_with_original_samplers()
    print("✓ test_compare_with_original_samplers passed")

    print("Running test_compare_fk_with_franka_robot...")
    test_compare_fk_with_franka_robot()
    print("✓ test_compare_fk_with_franka_robot passed")

    print("Running test_compare_compute_spheres_with_original...")
    test_compare_compute_spheres_with_original()
    print("✓ test_compare_compute_spheres_with_original passed")

    print("All tests passed!")
