import os
import argparse
import numpy as np
import cv2
import trimesh
from trimesh import transformations
import pyrender


def get_camera_position(opt, mesh):
    screen_size = mesh.extents.max()
    znear = 0.05
    zfar = screen_size + znear + opt.depth_range

    cam_pose = np.eye(4)
    cam_pose[0, 3] = mesh.centroid[0] - opt.x_shift * mesh.extents[0]
    cam_pose[1, 3] = mesh.centroid[1] - opt.y_shift * mesh.extents[1]
    cam_pose[2, 3] = mesh.bounds[0, 2] + zfar - opt.z_shift * mesh.extents[2]

    mag = screen_size / 2 * (1 - np.log(opt.scale))
    cam = pyrender.OrthographicCamera(xmag=mag, ymag=mag, znear=znear, zfar=zfar)

    return cam, cam_pose

def get_mesh(opt):
    mesh = trimesh.load(opt.model)

    x_rot = transformations.rotation_matrix(opt.x_angle, [1, 0, 0], model_trimesh.centroid)
    y_rot = transformations.rotation_matrix(opt.y_angle, [0, 1, 0], model_trimesh.centroid)
    z_rot = transformations.rotation_matrix(opt.z_angle, [0, 0, 1], model_trimesh.centroid)
    R = transformations.concatenate_matrices(x_rot, y_rot, z_rot)

    mesh.apply_transform(R)

    return pyrender.Mesh.from_trimesh(mesh)

def render(opt):
    scene = pyrender.Scene(ambient_light=(1, 1, 1), bg_color=(0, 0, 0))
    
    mesh = get_mesh(opt)
    scene.add(mesh)

    cam, cam_pose = get_camera_position(opt, mesh)
    scene.add(cam, pose=cam_pose)

    r = pyrender.OffscreenRenderer(opt.image_size, opt.image_size)
    depth = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.DEPTH_ONLY)

    depth = depth.clip(0, 1)
    if opt.inverse_depth:
        mask = depth != 0
        depth = (255*mask*(1-depth)).astype(np.uint8)
    else:
        depth = (255*depth).astype(np.uint8)

    return depth

def get_args():
    parser = argparse.ArgumentParser('OpenGL render')
    parser.add_argument('--save_folder', default='./output', help='results saving path')
    parser.add_argument('--model', default='./models/logotype_toon.obj', help='path to 3d model')
    parser.add_argument('--image_size', type=int, default=1024, help='size of rendered depth')
    parser.add_argument('--inverse_depth', action='store_true', help='0 value is far, 255 is close')
    parser.add_argument('--scale', type=float, default=0.7, help='scale of logotype size. Value is from interval (0, 1]')
    parser.add_argument('--x_shift', type=float, default=0.0, help='shift logotype along x axis. Value is from interval [-1, 1]')
    parser.add_argument('--y_shift', type=float, default=0.0, help='shift logotype along y axis. Value is from interval [-1, 1]')
    parser.add_argument('--z_shift', type=float, default=0.0, help='shift logotype along z axis. Value is from interval [0, 1]')
    parser.add_argument('--x_angle', type=float, default=0.0, help='rotation angle along x axis. Unit is radian.')
    parser.add_argument('--y_angle', type=float, default=0.0, help='rotation angle along y axis. Unit is radian.')
    parser.add_argument('--z_angle', type=float, default=0.0, help='rotation angle along z axis. Unit is radian.')
    parser.add_argument('--depth_range', type=float, default=11.0, help='distance between objects with different depth. Values from interval [0, 30], 0 means default range')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    depth = render(args)
    
    os.makedirs(args.save_folder, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    name = f'{model_name}_{args.image_size}_{args.scale}_{args.x_shift}_{args.y_shift}_{args.z_shift}_{args.x_angle}_{args.y_angle}_{args.z_angle}_{args.depth_range}_{args.inverse_depth}.png'
    cv2.imwrite(os.path.join(args.save_folder, name), depth)