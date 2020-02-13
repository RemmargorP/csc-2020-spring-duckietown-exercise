#!/usr/bin/env python3

import time
import sys
import argparse
import math
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='8')
parser.add_argument('--no-pause', action='store_true', help="don't pause on failure")
parser.add_argument('--backwards', action='store_true', help="drive backwards")
parser.add_argument('--render-human', action='store_true', help="render view from robot camera")
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = False,
        draw_bbox = False,
        max_steps = 5000
    )
else:
    env = gym.make(args.env_name)

obs = env.reset()
env.render()

render_mode = 'human' if args.render_human else 'top_down'
route = np.array([
    (2.0, 0, 1.3),
    (1.5, 0, 1.3),
    (1.3, 0, 1.5),
    (1.3, 0, 2.0),
    (1.3, 0, 3.0),
    (1.3, 0, 3.5),
    (1.5, 0, 3.7),
    (2.0, 0, 3.7),
    (3.0, 0, 3.7),
    (3.15, 0, 3.85),
    (3.3, 0, 4.0),
    (3.3, 0, 5.0),
    (3.15, 0, 5.15),
    (3.0, 0, 5.3),
    (2.0, 0, 5.3),
    (1.85, 0, 5.15),
    (1.7, 0, 5.0),
    (1.7, 0, 4.0),
    (1.85, 0, 3.85),
    (2.0, 0, 3.7),
    (3.0, 0, 3.7),
    (3.5, 0, 3.7),
    (3.7, 0, 3.5),
    (3.7, 0, 3.0),
    (3.7, 0, 2.0),
    (3.7, 0, 1.5),
    (3.5, 0, 1.3),
    (3.0, 0, 1.3)
])
route = route * env.road_tile_size

pt_index = np.argmin([np.linalg.norm(pt - env.cur_pos) for pt in route])

while True:
    direction = -1 if args.backwards else 1

    pt_dest = route[pt_index]
    dist = np.linalg.norm(pt_dest - env.cur_pos)
    if dist < 0.13:
        pt_index = (pt_index + direction) % len(route)
        pt_dest = route[pt_index]
        dist = np.linalg.norm(pt_dest - env.cur_pos)

    cur_dir = np.delete(env.get_dir_vec(), 1)
    v = np.delete(pt_dest - env.cur_pos, 1)
    desired_dir = direction * v / np.linalg.norm(v)
    rot = np.arctan2(-cur_dir[0] * desired_dir[1] + cur_dir[1] * desired_dir[0], np.dot(cur_dir, desired_dir))

    # print('pos {}, dest {}, dist {}'.format(np.delete(env.cur_pos, 1), pt_dest, dist))
    # print('cd {}, dd {}, rot {}, ang {}'.format(cur_dir, desired_dir, rot, env.cur_angle))

    speed = direction * 0.3 * (1 - 0.71 * np.clip(abs(rot) - 0.35, 0, 0.4))
    steering = 2 * rot

    env.step([speed, steering])
    env.render(mode=render_mode)
