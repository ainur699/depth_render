import os, sys
import argparse
import numpy as np
import cv2
import trimesh
from trimesh import transformations
import pyrender
from OpenGL.GL import *


def get_mesh(opt):
    mesh = trimesh.load(opt.model)

    x_rot = transformations.rotation_matrix(opt.x_angle, [1, 0, 0], mesh.centroid)
    y_rot = transformations.rotation_matrix(opt.y_angle, [0, 1, 0], mesh.centroid)
    z_rot = transformations.rotation_matrix(opt.z_angle, [0, 0, 1], mesh.centroid)
    R = transformations.concatenate_matrices(x_rot, y_rot, z_rot)

    mesh.apply_transform(R)

    return pyrender.Mesh.from_trimesh(mesh)

def get_camera_position(opt, mesh):
    screen_size = mesh.extents.max()
    znear = 0.001
    depth_size = mesh.extents[2] / (254/255*opt.depth_range+1/255)
    zfar = depth_size + znear

    cam_pose = np.eye(4)
    cam_pose[0, 3] = mesh.centroid[0] - opt.x_shift * mesh.extents[0]
    cam_pose[1, 3] = mesh.centroid[1] - opt.y_shift * mesh.extents[1]
    cam_pose[2, 3] = mesh.bounds[0, 2] + zfar - opt.z_shift * mesh.extents[2]

    mag = screen_size / 2 * (1 - np.log(opt.scale))
    cam = pyrender.OrthographicCamera(xmag=mag, ymag=mag, znear=znear, zfar=zfar)

    return cam, cam_pose

def read_depth(renderer, scene):
    width, height = renderer._main_fb_dims[0], renderer._main_fb_dims[1]

    # Bind framebuffer and blit buffers
    glBindFramebuffer(GL_READ_FRAMEBUFFER, renderer._main_fb_ms)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, renderer._main_fb)
    glBlitFramebuffer(
        0, 0, width, height, 0, 0, width, height,
        GL_COLOR_BUFFER_BIT, GL_LINEAR
    )
    glBlitFramebuffer(
        0, 0, width, height, 0, 0, width, height,
        GL_DEPTH_BUFFER_BIT, GL_NEAREST
    )
    glBindFramebuffer(GL_READ_FRAMEBUFFER, renderer._main_fb)

    # Read depth
    depth_buf = glReadPixels(
        0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT
    )
    depth_im = np.frombuffer(depth_buf, dtype=np.float32)
    depth_im = depth_im.reshape((height, width))
    depth_im = np.flip(depth_im, axis=0)
    inf_inds = (depth_im == 1.0)
    depth_im = 2.0 * depth_im - 1.0
    z_near = scene.main_camera_node.camera.znear
    z_far = scene.main_camera_node.camera.zfar
    noninf = np.logical_not(inf_inds)
    depth_im[noninf] = (z_far + z_near + depth_im[noninf] * (z_far - z_near)) / (2 * z_far)
    depth_im[inf_inds] = 0.0

    return depth_im

def render(opt):
    scene = pyrender.Scene(ambient_light=(1, 1, 1), bg_color=(0, 0, 0))
    
    mesh = get_mesh(opt)
    scene.add(mesh)

    cam, cam_pose = get_camera_position(opt, mesh)
    scene.add(cam, pose=cam_pose)

    r = pyrender.OffscreenRenderer(opt.image_size, opt.image_size)
    _ = r.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES | pyrender.RenderFlags.DEPTH_ONLY)

    # pyrender lib has a bug in getting linear depth with OrthographicCamera projection.
    # fixed it with depth buffer reading function
    depth = read_depth(r._renderer, scene)

    if opt.inverse_depth:
        mask = depth != 0
        depth = (255*mask*(1-depth)).astype(np.uint8)
    else:
        depth = (255*depth).astype(np.uint8)

    return depth

def get_args():
    parser = argparse.ArgumentParser('OpenGL render')
    parser.add_argument('--save_folder', default='./output', help='results saving path')
    parser.add_argument('--model', default='./models/logotype_toon.obj', help='path to 3d model.')
    parser.add_argument('--image_size', type=int, default=1024, help='size of rendered depth')
    parser.add_argument('--inverse_depth', action='store_true', help='closer objects have higher depth value')
    parser.add_argument('--scale', type=float, default=1.0, help='scale of logotype size. Interval is (0, 1]')
    parser.add_argument('--x_shift', type=float, default=0.0, help='shift logotype along x axis. Interval is [-1, 1]')
    parser.add_argument('--y_shift', type=float, default=0.0, help='shift logotype along y axis. Interval is [-1, 1]')
    parser.add_argument('--z_shift', type=float, default=0.0, help='shift logotype along z axis. Interval is [0, 1]')
    parser.add_argument('--x_angle', type=float, default=0.0, help='rotation angle along x axis. Unit is radian.')
    parser.add_argument('--y_angle', type=float, default=0.0, help='rotation angle along y axis. Unit is radian.')
    parser.add_argument('--z_angle', type=float, default=0.0, help='rotation angle along z axis. Unit is radian.')
    parser.add_argument('--depth_range', type=float, default=0.5, help='depth range. Interval is [0, 1], where 1 equals model depth and 0 means model depth is indistinguishable.')

    return parser.parse_args()

def debug_args():
    argv = '--x_angle 0.25 --y_angle 0.15 --inverse_depth --model models/PhotoLab.obj --scale 0.5 --depth_range 0.7'
    argv = argv.split()
    sys.argv.extend(argv)

if __name__ == '__main__':
    #debug_args()
    args = get_args()
    depth = render(args)
    
    os.makedirs(args.save_folder, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    name = f'{model_name}_{args.image_size}_{args.scale}_{args.x_shift}_{args.y_shift}_{args.z_shift}_' \
    f'{args.x_angle}_{args.y_angle}_{args.z_angle}_{args.depth_range}_{args.inverse_depth}.png'
    cv2.imwrite(os.path.join(args.save_folder, name), depth)