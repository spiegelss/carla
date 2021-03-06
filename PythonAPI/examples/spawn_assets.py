#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
#This script is used for dropping assets along the a lane section at random locations.

import glob
import os
import sys
import pandas as pd

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time
import random


def main():
    actor_list = []

    # In this tutorial script, we are going to add a vehicle to the simulation
    # and let it drive in autopilot. We will also create a camera attached to
    # that vehicle, and save all the images generated by the camera to disk.

    try:
        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # Once we have a client we can retrieve the world that is currently
        # running.
        world = client.get_world()
        m = world.get_map()
        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()
        cone_bp = blueprint_library.find('static.prop.trafficcone01')
        barrier_np = blueprint_library.find('static.prop.streetbarrier')
        bp_list = [cone_bp, barrier_np]
        #Waypoints
        waypoints = m.generate_waypoints(1)
        #lane width is 3.5

        way_points = pd.DataFrame(data=None, columns=['x', 'y', 'z','id', 'type'])
        x = []
        y = []
        z = []
        Road_type = []
        Road_id = []
        for w in waypoints:
            Road_id.append(w.road_id)
            x.append(w.transform.location.x)
            y.append(w.transform.location.y)
            z.append(w.transform.location.z)
            Road_type.append(str(w.lane_change))
        way_points.x = x
        way_points.y = y
        way_points.z = z
        way_points.id = Road_id
        way_points.type = Road_type
        way_points.to_csv("D:/Carla_Lidar/{}.csv".format("Way_pointsT2"), index=None)
        print("Please note the lane type ( Left or Right) and the lane ID to spawn assets along this lane type")
        time.sleep(5)
        for index, rows in way_points.iterrows():
            if rows['type'] == "Left":
                name = 'L - ' + str(rows['id'])
                world.debug.draw_string(carla.Location(x=rows['x'] , y=rows['y']), name, draw_shadow=False,
                                        color=carla.Color(r=0, g=0, b=255), life_time=200.0,
                                        persistent_lines=True)
            elif rows['type'] == "Right":
                name = 'R - ' + str(rows['id'])
                world.debug.draw_string(carla.Location(x=rows['x'], y=rows['y']), name, draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=200.0,
                                        persistent_lines=True)
            elif rows['type'] == "NONE":
                name = 'N - ' + str(rows['id'])
                world.debug.draw_string(carla.Location(x=rows['x'], y=rows['y']), name, draw_shadow=False,
                                        color=carla.Color(r=100, g=0, b=100), life_time=200.0,
                                        persistent_lines=True)
        time.sleep(5)
        chosen_road_ids = []
        if len(chosen_road_ids) != 0:
            for elem in chosen_road_ids:
                xList = []
                yList = []
                coord = pd.DataFrame(data=None, columns=['x', 'y'])
                for index, rows in way_points.iterrows():
                    if rows['id'] == elem:
                        xList.append(rows['x'])
                        yList.append(rows['y'])
                coord.x = xList
                coord.y = yList
                coord = coord.sample(n=3)
                for index,rows in coord.iterrows():
                    transform = carla.Transform(carla.Location(x=rows['x'], y=(rows['y']+3)),
                                                carla.Rotation(yaw=random.choice([0,45, 60,90])))
                    world.debug.draw_string(carla.Location(x=rows['x'], y=rows['y']+3, z=100), 'X', draw_shadow=True,
                                            color=carla.Color(r=255, g=0, b=0), life_time=200.0,
                                            persistent_lines=True)
                    assets = world.spawn_actor(random.choice(bp_list), transform)



    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')


if __name__ == '__main__':

    main()
