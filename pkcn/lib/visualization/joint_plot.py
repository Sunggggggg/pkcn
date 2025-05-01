import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

skeleton = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (4, 7),
        (2, 5), (5, 8),
        (3, 6), (6, 9),
        (9, 12), (12, 13), (13, 14), (14, 15),
        (12, 16), (16, 17), (17, 18),
        (12, 19), (19, 20), (20, 21),
        (0, 22), (22, 23)
    ]

def visualize_two_joints(joints1, joints2, save_path='joint_comparison.png'):
    """
    두 개의 관절 세트를 하나의 Plot에 시각화하여 이미지로 저장하는 함수
    Args:
        joints1 (numpy.ndarray or torch.Tensor): 첫 번째 관절, shape [54, 3]
        joints2 (numpy.ndarray or torch.Tensor): 두 번째 관절, shape [54, 3]
        save_path (str): 저장할 이미지 경로
    """
    # 텐서일 경우 numpy 배열로 변환
    if not isinstance(joints1, np.ndarray):
        joints1 = joints1.detach().cpu().numpy()
    if not isinstance(joints2, np.ndarray):
        joints2 = joints2.detach().cpu().numpy()

    # 3D 시각화
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Joint Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 첫 번째 관절 세트 시각화
    x1, y1, z1 = joints1[:, 0], joints1[:, 1], joints1[:, 2]
    ax.scatter(x1, y1, z1, color='blue', s=50, label="Joints Set 1")
    for i in range(len(x1)):
        ax.text(x1[i], y1[i], z1[i], f'{i}', color='blue', fontsize=8)

    # 두 번째 관절 세트 시각화
    x2, y2, z2 = joints2[:, 0], joints2[:, 1], joints2[:, 2]
    ax.scatter(x2, y2, z2, color='red', s=50, label="Joints Set 2")
    for i in range(len(x2)):
        ax.text(x2[i], y2[i], z2[i], f'{i}', color='red', fontsize=8)

    # 축 범위 자동 조정
    all_joints = np.concatenate([joints1, joints2], axis=0)
    max_range = np.ptp(all_joints, axis=0).max() / 2
    mid_x = np.mean(all_joints[:, 0])
    mid_y = np.mean(all_joints[:, 1])
    mid_z = np.mean(all_joints[:, 2])
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.legend()
    plt.savefig(save_path)
    plt.close()
    # print(f"Joint comparison visualization saved to {save_path}")

def visualize_two_joints_2d(joints1, joints2, save_path='joint_comparison_2d.png'):
    """
    두 개의 2D 관절 세트를 하나의 Plot에 시각화하여 이미지로 저장하는 함수
    Args:
        joints1 (numpy.ndarray or torch.Tensor): 첫 번째 관절, shape [54, 2]
        joints2 (numpy.ndarray or torch.Tensor): 두 번째 관절, shape [54, 2]
        save_path (str): 저장할 이미지 경로
    """
    # 텐서일 경우 numpy 배열로 변환
    if not isinstance(joints1, np.ndarray):
        joints1 = joints1.detach().cpu().numpy()
    if not isinstance(joints2, np.ndarray):
        joints2 = joints2.detach().cpu().numpy()

    # 2D 시각화
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("2D Joint Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # 첫 번째 관절 세트 시각화 (파란색)
    x1, y1 = joints1[:, 0], joints1[:, 1]
    ax.scatter(x1, y1, color='blue', s=50, label="Joints Set 1")
    for i in range(len(x1)):
        ax.text(x1[i], y1[i], f'{i}', color='blue', fontsize=8)

    # 두 번째 관절 세트 시각화 (빨간색)
    x2, y2 = joints2[:, 0], joints2[:, 1]
    ax.scatter(x2, y2, color='red', s=50, label="Joints Set 2")
    for i in range(len(x2)):
        ax.text(x2[i], y2[i], f'{i}', color='red', fontsize=8)

    # 전체 관절 좌표 결합
    all_joints = np.concatenate([joints1, joints2], axis=0)

    # x, y축 각각 최소/최대 계산
    x_min, x_max = all_joints[:, 0].min(), all_joints[:, 0].max()
    y_min, y_max = all_joints[:, 1].min(), all_joints[:, 1].max()

    # 마진(10% 비율) 적용
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min

    ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
    ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    ax.set_aspect('equal', adjustable='box')

    plt.legend()
    plt.savefig(save_path)
    plt.close()