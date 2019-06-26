#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import pandas as pd
import datetime
import json


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import logging
import random

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue

with open('C:/carla/Settings/settings.json', 'r') as f:
    settings_dict = json.load(f)

FileLocation = None
LidarParams = [0,0,0,0,0,0]

for setting in settings_dict:
    FileLocation = setting['FileLocation']
    LidarParams[0] = setting['Channels']
    LidarParams[1] = setting['PointsPerSecond']
    LidarParams[2] = setting['Range']
    LidarParams[3] = setting['RotationFrequency']
    LidarParams[4] = setting['UpperFOV']
    LidarParams[5] = setting['LowerFOV']
print("Loaded Configuration from settings.json")

currentDT = datetime.datetime.now()
Dataset_path = FileLocation + str(currentDT.strftime("%Y-%m-%d %H-%M-%S"))
Dataset_full_path = FileLocation + str(currentDT.strftime("%Y-%m-%d %H-%M-%S")) + "/{}.csv"
try:
    os.makedirs(Dataset_path, exist_ok=True)
except FileExistsError:
    # directory already exists
    pass

def draw_lidar(surface, lidar, vehicle):
    pointcount = 0
    channels = lidar.channels
    ego_trans = vehicle.get_transform()
    for i in range(0, channels):
        pointcount += lidar.get_point_count(i)
    totalpoints = pointcount * 3
    fullbuffer = np.frombuffer(lidar.raw_data, dtype=np.dtype('f4'), count=-1)
    points = np.frombuffer(lidar.raw_data, dtype=np.dtype('f4'), count=totalpoints)
    labels_ascii = np.frombuffer(lidar.raw_data, dtype=np.dtype('f4'), offset=totalpoints * 4)
    bigstring = ''.join(chr(i) for i in labels_ascii)
    bigstring = bigstring.split(',')
    labels_df = pd.DataFrame(bigstring)
    points = np.reshape(points, (int(points.shape[0] / 3), 3))

    '# putting into PANDAS dataframes'
    if len(points) > 0:
        x = pd.DataFrame(data=np.float16(points), columns=["x", "y", "z"])
        x["labels"] = labels_df
        x.loc[x.index[0], 'ego_X,Y,Z'] = np.float16(ego_trans.location.x)
        x.loc[x.index[1], 'ego_X,Y,Z'] = np.float16(ego_trans.location.y)
        x.loc[x.index[2], 'ego_X,Y,Z'] = np.float16(ego_trans.location.z)
        x.loc[x.index[0], 'ego_rot_P,Y,R'] = np.float16(ego_trans.rotation.pitch)
        x.loc[x.index[1], 'ego_rot_P,Y,R'] = np.float16(ego_trans.rotation.yaw)
        x.loc[x.index[2], 'ego_rot_P,Y,R'] = np.float16(ego_trans.rotation.roll)
        x.loc[x.index[0], 'Lidar_C,P,R,F,uF,lF'] = LidarParams[0]
        x.loc[x.index[1], 'Lidar_C,P,R,F,uF,lF'] = LidarParams[1]
        x.loc[x.index[2], 'Lidar_C,P,R,F,uF,lF'] = LidarParams[2]
        x.loc[x.index[3], 'Lidar_C,P,R,F,uF,lF'] = LidarParams[3]
        x.loc[x.index[4], 'Lidar_C,P,R,F,uF,lF'] = LidarParams[4]
        x.loc[x.index[5], 'Lidar_C,P,R,F,uF,lF'] = LidarParams[5]
        x.to_csv(Dataset_full_path.format(lidar.frame_number), index=None)

def draw_image(surface, image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    image.save_to_disk(Dataset_path+'/{}'.format(image.frame_number))
    surface.blit(image_surface, (0, 0))


'''
    LIDAR VISUALIZATION
    dim = (1280, 720)
    lidar_data = np.array(points[:, :2])
    lidar_data *= min(dim) / 100.0
    lidar_data += (0.5 * dim[0], 0.5 * dim[1])
    lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img_size = (dim[0], dim[1], 3)
    lidar_img = np.zeros((lidar_img_size), dtype=int)
    lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
    lidar_surface = pygame.surfarray.make_surface(lidar_img)
    surface.blit(lidar_surface, (0, 0))
    '''


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


def main():
    actor_list = []
    pygame.init()

    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    print('enabling synchronous mode.')
    settings = world.get_settings()
    settings.synchronous_mode = True
    world.apply_settings(settings)

    try:
        m = world.get_map()
        start_pose = random.choice(m.get_spawn_points())
        waypoint = m.get_waypoint(start_pose.location)
        vehicle_control = carla.VehicleControl()
        blueprint_library = world.get_blueprint_library()

        vehicle = world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.harley*')),
            start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(True)

        camera = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=0.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera)
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', LidarParams[0])
        lidar_bp.set_attribute('range', LidarParams[1])
        lidar_bp.set_attribute('points_per_second', LidarParams[2])
        lidar_bp.set_attribute('rotation_frequency', LidarParams[3])
        lidar_bp.set_attribute('upper_fov', LidarParams[4])
        lidar_bp.set_attribute('lower_fov', LidarParams[5])
        lidar = world.spawn_actor(
            lidar_bp,
            carla.Transform(carla.Location(x=0.5, z=2.8), carla.Rotation(pitch=0)),
            attach_to=vehicle)
        actor_list.append(lidar)


        # Make sync queue for sensor data.
        image_queue1 = queue.Queue()
        image_queue2 = queue.Queue()
        lidar.listen(image_queue1.put)
        camera.listen(image_queue2.put)

        frame = None

        display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        font = get_font()

        clock = pygame.time.Clock()

        while True:
            if should_quit():
                return

            clock.tick()
            world.tick()
            ts = world.wait_for_tick()

            if frame is not None:
                if ts.frame_count != frame + 1:
                    logging.warning('frame skip!')

            frame = ts.frame_count

            while True:
                lidar = image_queue1.get()
                image = image_queue2.get()
                if lidar.frame_number == ts.frame_count:
                    break
                logging.warning(
                    'wrong image time-stampstamp: frame=%d, image.frame=%d',
                    ts.frame_count,
                    lidar.frame_number)

            #waypoint = random.choice(waypoint.next(2))
            vehicle.apply_control(vehicle_control)
            vehicle.set_autopilot(True)


            #vehicle.set_transform(waypoint.transform)

            draw_image(display, image)
            draw_lidar(display, lidar, vehicle)


            text_surface = font.render('% 5d FPS' % clock.get_fps(), True, (255, 255, 255))
            display.blit(text_surface, (8, 10))

            pygame.display.flip()

    finally:
        print('\ndisabling synchronous mode.')
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    main()
