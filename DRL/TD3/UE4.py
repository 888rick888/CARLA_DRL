import glob
import os
import sys
import time
import random
import numpy as np
import cv2
import math

try:
    sys.path.append(glob.glob('D:\pzs\CARLA_0.9.12\WindowsNoEditor\PythonAPI\carla\dist\carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(glob.glob('/home/amax/NVME_SSD/Pan.zs/carla0.9.12/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(glob.glob('../carla0.9.12/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
# sys.path.append('..')
# from agents.navigation.basic_agent import BasicAgent

random.seed(1)
np.random.seed(1)

SHOW_PREVIEW = False
SMOOTHING_NUM = 6
MAX_DISTANCE = 8
RECOMMEND_V = 30

REWARD_ANGLE = 1
REWARD_DISTANCE = 1
REWARD_ACCEL = 1

class CarEnv:
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    front_camera = None
    
    def __init__(self, segmentation, state_size):
        # self.client = carla.Client("125.217.226.130",2000)
        # self.client = carla.Client("10.168.3.204",2000)
        self.client = carla.Client("localhost",2000)
        # self.client = carla.Client("125.217.226.196",2000)
        # self.client = carla.Client("125.217.226.228",2010)

        self.client.set_timeout(30.0)
        # self.world = self.client.reload_world() # 重启世界
        print("===  reloading world ......  ===")

        # self.world = self.client.load_world('Town04')
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        self.settings = self.world.get_settings()
        # settings.no_rendering_mode = False
        # self.settings.synchronous_mode = True
        # self.world.apply_settings(settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.EP9 = self.blueprint_library.filter("model3")[0]

        self.steer_t = 0
        self.segmentation = segmentation
        self.im_width = state_size
        self.im_height = state_size
        self.smoothing_list = np.zeros(SMOOTHING_NUM)

    def reset(self):
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)

        self.actor_list = []
        self.sensorlist = []
        self.lane_hist = []
        self.obstacle_list = []
        self.collision_hist = []
        self.vehicle = None
        self.lane_text = ["'s'"]

        # try:
        #     vehicle_to = self.world.get_actors().filter('vehicle.*')
        #     self.client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_to])
        # except: print("---  Can't do  ---")

        # self.world = self.client.reload_world()     #重启世界
        # self.world = self.client.get_world()

        self.EP9.set_attribute('role_name', 'ego')
        # self.EP9.set_attribute('role_name', 'pzs')s
        while self.vehicle == None:
            # self.transform = random.choice(self.world.get_map().get_spawn_points()) #total 155 points
            self.transform = self.world.get_map().get_spawn_points()[self.spawn_point()]        #data clean 2.25
            # self.transform = self.world.get_map().get_spawn_points()[26]        #data clean 2.25
            self.vehicle = self.world.try_spawn_actor(self.EP9, self.transform)
        self.actor_list.append(self.vehicle)
        if self.segmentation:
            self.rgb_cam = self.blueprint_library.find("sensor.camera.semantic_segmentation")

        else: self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb_cam.set_attribute("image_size_x",f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y",f"{self.im_height}")
        self.rgb_cam.set_attribute("fov",f"110")

        transform_sensor = carla.Transform(carla.Location(x=2.5,z=1.7))
        # transform_cam = carla.Transform(carla.Location(x=2.5,z=1.7), carla.Rotation(pitch=-45)) #camera spawn more lane 2.25
        transform_cam = carla.Transform(carla.Location(x=2.5,z=0.9), carla.Rotation(pitch=-45)) #camera spawn more lane 2.25
        self.sensor = self.world.spawn_actor(self.rgb_cam,transform_cam,attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        self.vehicle.apply_control(carla.VehicleControl(throttle=0.8,brake=0.0))
        time.sleep(3)

        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor,transform_sensor,attach_to=self.vehicle)
        self.sensorlist.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        
        lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanesensor = self.world.spawn_actor(lanesensor,transform_sensor,attach_to=self.vehicle)
        self.sensorlist.append(self.lanesensor)
        self.lanesensor.listen(lambda event: self.lane_type(event))

        # obstaclesensor = self.blueprint_library.find('sensor.other.obstacle')
        # self.sensor_obstacle = self.world.spawn_actor(obstaclesensor, transform_sensor, attach_to=self.vehicle)
        # self.sensorlist.append(self.sensor_obstacle)
        # self.sensor_obstacle.listen(lambda event: self.obstacle_type(event))

        while self.front_camera is None:
            time.sleep(0.01)

        # self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.5,brake=0.0))    #添加初始启动动作，测试是否由于车辆启动策略有问题
        
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)
        vector = self.get_V2X()
        return [self.front_camera, vector]

    def collision_data(self,event):
        self.collision_hist.append(event)

    def lane_type(self, event):
        self.lane_hist.append(event)
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.lane_text = text

    def obstacle_type(self,event):
        self.obstacle_list.append(event)
        self.obstacle_actor = event.actor
        self.obstacle_other_actor = event.other_actor
        self.obstacle_distance = event.distance
        # print('obstacle_text_actor', event.actor)
        # print('obstacle_text_other', event.other_actor )
        # print('obstacle_text_distance', event.distance)

    def get_traffic_status(self):
        get_traffic_light = self.vehicle.get_traffic_light()
        get_traffic_light_state = self.vehicle.get_traffic_light_state()
        # get_wheel_steer_angle = self.vehicle.get_wheel_steer_angle(wheel_location=carla.VehicleWheelLocation().FL_Wheel)
        # get_physics_control = self.vehicle.get_physics_control()
        # get_speed_limit = self.vehicle.get_speed_limit()
        # get_control = self.vehicle.get_control()
        # bounding_box  = self.vehicle.bounding_box
        is_at_traffic_light = self.vehicle.is_at_traffic_light()
        return get_traffic_light, get_traffic_light_state, is_at_traffic_light

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("", i3)
            cv2.waitKey(1)
        self.front_camera = i3 / 255
        
    def throttle(self):
        v =  self.vehicle.get_velocity()
        kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if kmh <= 5: return 0.6
        elif kmh > 10: return 0.0
        else: return 0.3

    def step(self,action):
        done = False
        a_steer = float(action[0])
        a_accel = float(action[1])
        brake = 0
        if a_accel < 0:
            brake = abs(a_accel)
        reward = np.zeros(2)

        lane_waypoint = self.map.get_waypoint(self.vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
        vehicle_transform = self.vehicle.get_transform()
        x_distance = lane_waypoint.transform.location.x - vehicle_transform.location.x
        y_distance = lane_waypoint.transform.location.y - vehicle_transform.location.y
        angle = lane_waypoint.transform.rotation.yaw - vehicle_transform.rotation.yaw
        distance = math.sqrt(x_distance**2 + y_distance**2)
        if angle > 180: angle = (angle - 360)
        elif angle < -180:  angle = (angle +360)
        elif angle > 90:  angle = (angle -180)
        elif angle < -90:  angle = (angle +180)
        else: angle = angle
        radian = math.radians(angle-45)
        reward_a = math.cos(radian)-math.sin(radian) #high_value is 1.41 
        reward_d = np.clip(math.exp(distance) - 1, 0 , 2.81)

        self.vehicle.apply_control(carla.VehicleControl(throttle=a_accel, steer=a_steer*self.STEER_AMT, brake=brake))
        self.world.tick()
        # self.world.wait_for_tick()

        reward[0] = REWARD_ANGLE * reward_a - REWARD_DISTANCE * reward_d
        reward[0] -= abs(a_steer) / 3


        v2x_vector = self.get_V2X()
        # reward[1] = - abs(v2x_vector[3] - v2x_vector[0]*3) - v2x_vector[2]*0.5 - min((v2x_vector[1] + v2x_vector[4])/(v2x_vector[1] * v2x_vector[4] + 0.00001), 5)
        reward[1] = - abs(v2x_vector[3]/3 - v2x_vector[0])*2  + 1 - min((v2x_vector[1] + v2x_vector[4])/(v2x_vector[1] * v2x_vector[4] + 0.00001), 5)
        if a_accel < 0:
            reward[1] -= abs(a_accel)
        reward[1] *= REWARD_ACCEL
        # print("the reward of accel is ", -(v2x_vector[3] - v2x_vector[0]*3)*2, - v2x_vector[2]*0.5, - min((v2x_vector[1] + v2x_vector[4])/(v2x_vector[1] * v2x_vector[4] + 0.00001), 5))

        if len(self.collision_hist) != 0:
            done = True
            reward[0] = -10
            reward[1] = -10
        elif self.lane_text[0] == "'Solid'" or self.lane_text[0] == "'SolidSolid'" or self.lane_text[0] == "'Broken'":
            done = True
            self.lane_text = ["'s'"]
            reward[0] = -10
            reward[1] = -3
        elif distance > 1:
            done = True
            reward[0] = -10
            reward[1] = -3
        elif v2x_vector[2] > 1:
            done = True
            reward[0] = -3
            reward[1] = -10
        else:
            done = False

        return [self.front_camera, v2x_vector],reward,done,None

    def des(self):
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        for i in self.actor_list:
            i.destroy()
        for o in self.sensorlist:
            o.destroy()
        # self.actor_list.clear()
        # self.sensorlist.clear()
    
    def spawn_point(self):
        line = [10,15,22,23,26,27,30,31,34,45,46,47,48,49,50,53,66,68,69,71,75,82,83,93,101,104,105,108,116,117,118,124,126,129,130,134,135,147,148,151,152]
        turn = [4,5,32,62,63,65,78,106,120,121,149,150,154]
        light = [11,13,14,24,25,40,45,46,48,49,51,52,64,67,68,70,72,73,74,77,79,80,81,85,86,87,89,90,91,92,94,95,96,98,100,103,104,111,115,122,126,127,134,135,137,138,139,142,145,146,147,148,153]
        line_out = [6,7,8,9,10,15,19,20,22,23,54,60,61,66,69,71,75,82,83,88,100,113,125,131,132,133,143,147,148,151,152]
        line_in = [45,46,47,48,49,56,68,93,101,105,108,116,114,118,124,126,129,130,134,135,31,30,34,45]
        # return random.choice(line+line_to_turn)
        return random.choice(line_out+line_in)
        return random.choice(line_out+line_in)
        
    # def agent_action(self):
        # agent = BasicAgent(self.veh、icle)
        # control = agent.run_step()
        # control.manual_gear_shift = False
        # return control

    def calc_distance(self, target_rear_transform, front_transform, max_vehicle_distance):
        x = (target_rear_transform.location.x-front_transform.location.x)**2
        y = (target_rear_transform.location.y-front_transform.location.y)**2
        return math.sqrt(x+y)
        if max_vehicle_distance > math.sqrt(x+y):
            return True
        else: return False

    def get_V2X(self):
        #将路侧信息转换为向量
        #目前向量长度为5，[自车车速/30， 车车距离/8， 红绿灯0or10， 推荐速度/10， 人车距离/10]
        vehicle_list = self.world.get_actors().filter("*vehicle*")
        walker_list = self.world.get_actors().filter("*walker*")
        lights_list = self.world.get_actors().filter("*traffic_light*")
        max_vehicle_distance = MAX_DISTANCE
        vector = np.zeros(5)

        v =  self.vehicle.get_velocity()
        kmh = int(3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))
        vector[0] = kmh/30
        vector[3] = RECOMMEND_V / 10

        ego_transform = self.vehicle.get_transform()
        ego_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        forward_vector = ego_transform.get_forward_vector()
        extent = self.vehicle.bounding_box.extent.x
        front_transform = ego_transform
        front_transform.location += carla.Location(x=extent * forward_vector.x, y=extent * forward_vector.y)

        for target_walker in walker_list:
            walker_transform = target_walker.get_transform()
            walker_waypoing = self.map.get_waypoint(walker_transform.location)
            if walker_waypoing.road_id != ego_waypoint.road_id or walker_waypoing.lane_id != ego_waypoint.lane_id:
                continue
            distance_walker = self.calc_distance(walker_transform, front_transform, max_vehicle_distance)
            if distance_walker < max_vehicle_distance:
                vector[4] = distance_walker / 10

        for target_light in lights_list:
            # light_transform = target_light.get_transform()
            # light_waypoint = self.map.get_waypoint(light_transform.location)
            light_location = self.get_trafficlight_trigger_location(target_light)
            light_waypoint = self.map.get_waypoint(light_location)
            if light_waypoint.road_id != ego_waypoint.road_id or light_waypoint.lane_id != ego_waypoint.lane_id:
                continue

            ve_dir = forward_vector
            wp_dir = light_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
            if dot_ve_wp < 0:
                continue

            if target_light.state == carla.TrafficLightState.Green :
                continue
            distance_light = self.calc_distance(light_waypoint.transform, front_transform, max_vehicle_distance)
            if distance_light < max_vehicle_distance:
                vector[2] = distance_light / 8

        for target_vehicle in vehicle_list:
            target_transform = target_vehicle.get_transform()
            target_waypoint = self.map.get_waypoint(target_transform.location)
            if target_vehicle.id == self.vehicle.id:
                continue
            if target_waypoint.road_id != ego_waypoint.road_id or target_waypoint.lane_id != ego_waypoint.lane_id:
                continue
                # next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                # if not next_wpt:
                #     continue
                # if target_waypoint.road_id != next_wpt.road_id or target_waypoint.lane_id != next_wpt.lane_id:
                #     continue

            target_forward_vector = target_transform.get_forward_vector()
            target_extent = target_vehicle.bounding_box.extent.x
            target_rear_transform = target_transform
            target_rear_transform.location -= carla.Location(
                x=target_extent * target_forward_vector.x,
                y=target_extent * target_forward_vector.y,
            )

            # if self.calc_distance(target_rear_transform, front_transform, max_vehicle_distance, [0, 90]):
            # if self.within_distance(target_rear_transform, front_transform, max_vehicle_distance):
            distance_vehicle = self.calc_distance(target_rear_transform, front_transform, max_vehicle_distance)
            if distance_vehicle < max_vehicle_distance:
                vector[1] = distance_vehicle / 8
                # return front_transform, target_rear_transform
        return np.array(vector, np.float32)

    def smoothing(self, action):
        for i in range(len(self.smoothing_list)-1):
            self.smoothing_list[i] = self.smoothing_list[i+1]
        self.smoothing_list[-1] = action
        return np.mean(self.smoothing_list)

    def get_trafficlight_trigger_location(self,traffic_light):
        def rotate_point(point, radians):
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y
            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)