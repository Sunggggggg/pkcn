import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.cameras import look_at_rotation

def merge_meshes(meshes):
    if not meshes:
        raise ValueError("At least one mesh is required.")
    
    if len(meshes) == 1:
        return meshes[0]

    all_verts = []
    all_faces = []
    all_verts_features = []
    verts_offset = 0

    for mesh in meshes:
        verts = mesh.verts_packed()
        faces = mesh.faces_packed() + verts_offset
        all_verts.append(verts)
        all_faces.append(faces)
        verts_offset += verts.shape[0]

        if isinstance(mesh.textures, TexturesVertex):
            all_verts_features.append(mesh.textures.verts_features_packed())

    merged_verts = torch.cat(all_verts, dim=0)
    merged_faces = torch.cat(all_faces, dim=0)

    if all_verts_features:
        merged_textures = TexturesVertex(verts_features=[torch.cat(all_verts_features, dim=0)])
    else:
        merged_textures = None

    merged_mesh = Meshes(verts=[merged_verts], faces=[merged_faces], textures=merged_textures).cuda()
    
    return merged_mesh


def checkerboard_geometry(
    length=12.0,
    color0=[0.8, 0.9, 0.9],
    color1=[0.6, 0.7, 0.7],
    tile_width=0.5,
    alpha=1.0,
    up="y",
    c1=0.0,
    c2=0.0,
):
    assert up == "y" or up == "z"
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    radius = length / 2.0
    num_rows = num_cols = max(2, int(length / tile_width))
    vertices = []
    vert_colors = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zs = np.zeros(4)
            if up == "y":
                cur_verts = np.stack([us, zs, vs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 2] += c2
            else:
                cur_verts = np.stack([us, vs, zs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 1] += c2

            cur_faces = np.array(
                [[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64
            )
            cur_faces += 4 * (i * num_cols + j)  # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            vert_colors.append(cur_colors)
            face_colors.append(cur_colors)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
    faces = np.concatenate(faces, axis=0).astype(np.float32)
    face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

    return vertices, faces, vert_colors, face_colors


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    roi_image[mask] = image[mask]
    out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype
    
    K = torch.zeros((K_org.shape[0], 4, 4)
    ).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1
    
    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = (right - left)
    height = (bottom - top)

    new_left = torch.clamp(cx - width/2 * scaleFactor, min=0, max=img_w-1)
    new_right = torch.clamp(cx + width/2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h-1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = torch.stack((new_left.detach(), new_top.detach(),
                        new_right.detach(), new_bottom.detach())).int().float().T
    
    return bbox


class Renderer():
    def __init__(self, width, height, focal_length, device, faces=None):

        self.width = width
        self.height = height
        self.focal_length = focal_length

        self.device = device
        if faces is not None:
            self.faces = torch.from_numpy(
                (faces).astype('int')
            ).unsqueeze(0).to(self.device)

        self.initialize_camera_params()
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer()

    def create_renderer(self):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0],
                    blur_radius=1e-5),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            )
        )

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False)


    def initialize_camera_params(self):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""

        # Extrinsics
        self.R = torch.diag(
            torch.tensor([1, 1, 1])
        ).float().to(self.device).unsqueeze(0)

        self.T = torch.tensor(
            [0, 0, 0]
        ).unsqueeze(0).float().to(self.device)

        # Intrinsics
        self.K = torch.tensor(
            [[self.focal_length, 0, self.width/2],
            [0, self.focal_length, self.height/2],
            [0, 0, 1]]
        ).unsqueeze(0).float().to(self.device)
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)
        self.cameras = self.create_camera()
        
        
    def set_ground(self, length, center_x, center_z):
        device = self.device
        v, f, vc, fc = map(torch.from_numpy, checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y"))
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]


    def update_bbox(self, x3d, scale=2.0, mask=None):
        """ Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """

        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(self,):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, vertices, background, colors=[0.8, 0.8, 0.8]):
        self.update_bbox(vertices[::50], scale=1.2)
        vertices = vertices.unsqueeze(0)
        
        if colors[0] > 1: colors = [c / 255. for c in colors]
        verts_features = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
        verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)
        
        mesh = Meshes(verts=vertices,
                      faces=self.faces,
                      textures=textures,)
        
        materials = Materials(
            device=self.device,
            specular_color=(colors, ),
            shininess=0
            )

        results = torch.flip(
            self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights),
            [1, 2]
        )
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3

        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
        self.reset_bbox()
        return image
    
    
    def render_with_ground(self, verts, faces, colors, cameras, lights):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """
        
        # (B, V, 3), (B, F, 3), (B, V, 3)
        verts, faces, colors = prep_shared_geometry(verts, faces, colors)
        # (V, 3), (F, 3), (V, 3)
        gv, gf, gc = self.ground_geometry
        verts = list(torch.unbind(verts, dim=0)) + [gv]
        faces = list(torch.unbind(faces, dim=0)) + [gf]
        colors = list(torch.unbind(colors, dim=0)) + [gc[..., :3]]
        mesh = create_meshes(verts, faces, colors)

        materials = Materials(
            device=self.device,
            shininess=0
        )
        
        results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            
        return image
    
    def render_meshes_with_ground(self, verts_list, faces, colors_list, cameras, lights):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """
        
        # (B, V, 3), (B, F, 3), (B, V, 3)
        mesh_list = []
        for verts, colors in zip(verts_list, colors_list) :
            _verts, _faces, _colors = prep_shared_geometry(verts, faces, colors)
        
            gv, gf, gc = self.ground_geometry
            _verts = list(torch.unbind(_verts, dim=0)) + [gv]
            _faces = list(torch.unbind(_faces, dim=0)) + [gf]
            _colors = list(torch.unbind(_colors, dim=0)) + [gc[..., :3]]
            mesh = create_meshes(_verts, _faces, _colors)
            mesh_list.append(mesh)

        materials = Materials(
            device=self.device,
            shininess=0
        )
        
        mesh = merge_meshes(mesh_list)
        
        results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            
        return image
    
def prep_shared_geometry(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (F, 3)
    :param colors (B, 4)
    """
    B, V, _ = verts.shape
    F, _ = faces.shape
    colors = colors.unsqueeze(1).expand(B, V, -1)[..., :3]
    faces = faces.unsqueeze(0).expand(B, F, -1)
    return verts, faces, colors


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def get_global_cameras(verts, device, distance=5, position=(-5.0, 5.0, 0.0)):
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets = verts.mean(1)
    
    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    positions = targets - directions
    
    rotation = look_at_rotation(positions, targets, ).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)
    
    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights


def _get_global_cameras(verts, device, min_distance=3, chunk_size=100):
    
    # split into smaller chunks to visualize
    start_idxs = list(range(0, len(verts), chunk_size))
    end_idxs = [min(start_idx + chunk_size, len(verts)) for start_idx in start_idxs]
    
    Rs, Ts = [], []
    for start_idx, end_idx in zip(start_idxs, end_idxs):
        vert = verts[start_idx:end_idx].clone()
        import pdb; pdb.set_trace()

def visualization_meshes_camera_space(output_folder, img_file_list, meshes_list, 
            width, height, focal_length, device, faces):

    default_R, default_T = torch.eye(3), torch.zeros(3)
    renderer = Renderer(width, height, focal_length, device, faces)
    renderer.create_camera(default_R, default_T)
    
    assert len(img_file_list) == len(meshes_list)
    
    pbar = tqdm(enumerate(zip(img_file_list, meshes_list)), desc="Rendering Meshes")
    for frame_idx, (img_file, meshes) in pbar:
        image = cv2.imread(img_file)[:, :, ::-1]
        
        renderered_file = os.path.join(output_folder, f"rendered_{frame_idx:06d}.jpg")
        for mesh in meshes:
            mesh = torch.from_numpy(mesh).float().to(device)
            image = renderer.render_mesh(mesh, image, colors=[0.8, 0.8, 0.8])
        cv2.imwrite(renderered_file, image[..., ::-1])
        pbar.set_postfix_str(f"Rendered {renderered_file}")