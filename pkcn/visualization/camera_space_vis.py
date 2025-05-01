import os
import cv2
import imageio
import argparse
import joblib
import subprocess
import numpy as np
import torch
import glob
from tqdm import tqdm
from smplx import SMPL, SMPLX
from .renderer import Renderer

import cv2
import numpy as np
import os

def save_obj(verts, faces, obj_mesh_name="mesh.obj"):
    # print('Saving:',obj_mesh_name)
    with open(obj_mesh_name, "w") as fp:
        for v in verts:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))

        for f in faces:  # Faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0] + 1, f[1] + 1, f[2] + 1))
            
def merge_videos(video_paths, output_path, mode='horizontal'):
    """
    Merge multiple videos of the same length into one.
    
    Parameters:
        video_paths (list): List of video file paths.
        output_path (str): Output video file path.
        mode (str): 'horizontal' or 'vertical' merging.
    """
    # Open video captures
    caps = [cv2.VideoCapture(vp) for vp in video_paths]
    
    # Get video properties
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check all videos have the same properties
    for cap in caps:
        if (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != frame_width or
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != frame_height or
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != frame_count):
            print("Error: Videos must have the same resolution and frame count.")
            return
    
    # Determine output resolution
    if mode == 'horizontal':
        out_width = frame_width * len(video_paths)
        out_height = frame_height
    else:  # 'vertical'
        out_width = frame_width
        out_height = frame_height * len(video_paths)
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    # Merge frames
    for _ in range(frame_count):
        frames = [cap.read()[1] for cap in caps]
        if None in frames:
            break
        merged_frame = np.hstack(frames) if mode == 'horizontal' else np.vstack(frames)
        out.write(merged_frame)
    
    # Release resources
    for cap in caps:
        cap.release()
    out.release()
    cv2.destroyAllWindows()
    
def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    x, y, w, h = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:, 3]
    cx, cy, h = x + w / 2., y + h / 2., h
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam

def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-framerate', '30000/1001', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.jpg', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

def render(verts, cam, bbox, orig_height, orig_width, orig_img, mesh_face, color, mesh_filename):
    pred_verts, pred_cam, bbox = verts, cam[None, :], bbox[None, :]

    orig_cam = convert_crop_cam_to_orig_img(
        cam=pred_cam,
        bbox=bbox,
        img_width=orig_width,
        img_height=orig_height
    )

    # Setup renderer for visualization
    renderer = Renderer(mesh_face, resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
    renederd_img = renderer.render(
        orig_img,
        pred_verts,
        cam=orig_cam[0],
        color=color,
        mesh_filename=mesh_filename,
        rotate=False
    )

    return renederd_img

def convert_pare_to_full_img_cam(
        pare_cam, 
        bbox_height, 
        bbox_center,
        img_w, 
        img_h, 
        focal_length, 
        crop_res=224
):

    s, tx, ty = pare_cam[0], pare_cam[1], pare_cam[2]
    res = crop_res
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[1] - (img_h / 2.)) / (s * bbox_height)

    cam_t = np.stack([tx + cx, ty + cy, tz], axis=-1)
    return cam_t

def wham_main(args):
    cam_id = args.cam_id
    
    print(">> Visualization Start...")
    output_root = args.output_folder
    output_mesh_folder = os.path.join(output_root, "meshes", f"cam_{cam_id:02d}")
    output_img_folder = os.path.join(output_root, "images", f"cam_{cam_id:02d}")
    output_video_folder = os.path.join(output_root, "videos", f"cam_{cam_id:02d}")
    os.makedirs(output_mesh_folder, exist_ok=True)
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_video_folder, exist_ok=True)
    
    ### Get mesh face ###
    if args.use_smplx :
        faces = SMPLX("model_data/smplx_models").faces
    else :
        faces = SMPL("model_data/smpl_models").faces
    
    ### Data Load ###
    """ 비어있는 Frame에도 
    hmr_results[0] = {"person_id": {"verts": [4, 6890, 3], "cam": [4, 3], "bbox":[4, 4]}}
    hmr_results[1] = {"person_id": {"verts": [4, 6890, 3], "cam": [4, 3], "bbox":[4, 4]}}
    hmr_results[2] = {}
    """
    hmr_results = joblib.load(args.result_folder)
    
    ### Image Load ###
    image_folder = args.image_folder
    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if (x.endswith('.png') or x.endswith('.jpg')) and f'cam_{cam_id+1:06d}' in x
    ])
    orig_height, orig_width = cv2.imread(image_file_names[0]).shape[:2]
    
    color_dict = {0:[255., 85., 127.],
                  1:[86., 170., 0.00],
                  2:[0.00, 170., 255.], 
                  3:[214, 214, 0.00], 
                  4:[0.90, 0.90, 0.80]}
    
    ### Renderer ###
    default_R, default_T = torch.eye(3), torch.zeros(3)
    video_file = os.path.join(output_video_folder, f'cam_{cam_id:06d}.mp4')
    writer = imageio.get_writer(
        video_file, 
        fps=30, mode='I', format='FFMPEG', macro_block_size=1
    )
    
    total_image_len = len(image_file_names)
    print(f"Camera ID : {cam_id}, Total seq : {total_image_len}")

    pbar = tqdm(range(total_image_len), desc=f"{output_mesh_folder}")

    for frame_idx in pbar:
        img_file = image_file_names[frame_idx]
        org_img = cv2.imread(img_file)
        img = org_img[..., ::-1].copy()
        
        for person_id, person_data in hmr_results[frame_idx].items():
            frame_verts = person_data['verts'][cam_id]  # [6890, 3]
            frame_cam = person_data['cam'][cam_id]      # [3]
            frame_bbox = person_data['bbox'][cam_id]    # [4]
            frame_focal = person_data['focal'][cam_id]  # [2]
            mc = color_dict[person_id]
            
            lt_x, lt_y, w, h = frame_bbox
            cx, cy = lt_x + w//2, lt_y + h//2
        
            trans_cam = convert_pare_to_full_img_cam(frame_cam, bbox_height=h, bbox_center=[cx, cy],
                                         img_w=orig_width, img_h=orig_height, focal_length=frame_focal[0], crop_res=512)
            verts = frame_verts + trans_cam.reshape(1, 3)
            
            renderer = Renderer(orig_width, orig_height, frame_focal[0], 'cuda', faces)
            renderer.create_camera(default_R, default_T)
            img = renderer.render_mesh(torch.from_numpy(verts).float().to('cuda'), img, mc)
            
            obj_file = os.path.join(output_mesh_folder, f"{person_id:04d}_frame_{frame_idx:04d}.obj")
            
            #verts -= verts.mean(axis=0, keepdims=True)
            save_obj(verts, faces, obj_file)
            pbar.set_postfix_str(f"{obj_file}")
        
        cv2.imwrite(os.path.join(output_img_folder, f"{frame_idx:06d}.jpg"), img[..., ::-1])
        writer.append_data(img)
        
    writer.close()
    
    ### Save rendererd video ###
    print(">> Visualization Finish...")
    return video_file

def wham_multi_camera(args):
    output_video_list = []
    for i in range(4):
        args.cam_id = i
        video_file = wham_main(args)
        output_video_list.append(video_file)
    
    # merge_videos(output_video_list, "demo_vis/output.mp4", )
    
if __name__ == "__main__":
    """
    python -m visualization.camera_space_vis --result_folder demo_output/quick_demo/model_results/model_results.pth --image_folder demo/images
    python -m visualization.camera_space_vis --result_folder demo_output/quick_video_demo/model_results/model_results.pth --image_folder demo_video/images
    
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='output folder name')
    parser.add_argument('--result_folder', type=str, help='result folder name')
    parser.add_argument('--image_folder', type=str, help='image folder name')
    parser.add_argument('--cam_id', type=int, default=0, help='camera id')
    parser.add_argument('--use_smplx', action="store_true")
    parser.add_argument('--save_obj', action='store_true', help='save results as .obj files.')
    args = parser.parse_args()
    
    # single
    #wham_main(args)
    
    # All camera
    wham_multi_camera(args)
    