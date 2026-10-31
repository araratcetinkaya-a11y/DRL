import airsim
import numpy as np
import math
from gym import spaces


class MultirotorDynamicsAirsim():
    '''
    A simple multirotor dynamics used for vision based navigation
    The controller is AirSim Simple Flight (https://microsoft.github.io/AirSim/simple_flight/)
    API: client.moveByVelocityZAsync(v_x, v_y, v_z, yaw_rate) is used to control
    '''
    # multirotor_airsim.py

    def __init__(self, cfg) -> None:
        
        # config
        self.navigation_3d = cfg.getboolean('options', 'navigation_3d')
        self.using_velocity_state = cfg.getboolean(
            'options', 'using_velocity_state')
        self.dt = cfg.getfloat('multirotor', 'dt')

        # AirSim Client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        kp_xy = 0.6
        ki_xy = 0.0
        kd_xy = 0.0

        kp_z = 2.0
        ki_z = 2.0
        kd_z = 0.0

        vel_gains = airsim.VelocityControllerGains(
            airsim.PIDGains(kp_xy, ki_xy, kd_xy),  # X
            airsim.PIDGains(kp_xy, ki_xy, kd_xy),  # Y
            airsim.PIDGains(kp_z,  ki_z,  kd_z),   # Z
        )

        self.client.setAngleLevelControllerGains(airsim.AngleLevelControllerGains(
            airsim.PIDGains(2.5, 0.0, 0.0),  # Roll
            airsim.PIDGains(2.5, 0.0, 0.0),  # Pitch
            airsim.PIDGains(4, 0, 0.0)   # Yaw 
        ))

        self.client.setVelocityControllerGains(vel_gains)

        # start and goal position
        self.start_position = [0, 0, 0]
        self.start_random_angle = None
        self.goal_position = [0, 0, 0]
        self.goal_distance = None
        self.goal_random_angle = None
        self.goal_rect = None

        # states
        self.x = 0
        self.y = 0
        self.z = 0
        self.v_xy = 0
        self.v_z = 0
        self.yaw = 0
        self.yaw_rate = 0
        self.yaw_rate_cmd_deg = 0
        # cmd
        self.v_xy_sp = 0
        self.v_z_sp = 0
        self.yaw_rate_sp = 0
        self.prev_v_xy_sp = 0
        self.prev_v_z_sp  = 0
        self.prev_yaw_cmd_deg = 0.0
        self.prev_yaw_rate_deg = 0.0

        # --- YENİ (GÖREV B): Son eylemi saklamak için ---
        # Geçici olarak 'None' ayarlandı, action_space'den sonra güncellenecek.
        self.last_action = None

        # action space
        self.acc_xy_max =  cfg.getfloat('multirotor', 'acc_xy_max')
        self.v_xy_max =   cfg.getfloat('multirotor', 'v_xy_max')
        self.v_xy_min =  cfg.getfloat('multirotor', 'v_xy_min')
        self.v_z_max = cfg.getfloat('multirotor', 'v_z_max')
        self.yaw_rate_max_deg = cfg.getfloat('multirotor', 'yaw_rate_max_deg')
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.yaw_accel_max_deg_s2 = 100.0  # Maksimum dönüş ivmesi (100 deg/s^2)
        self.yaw_delta_max_deg = 4
        self.v_xy_delta_max = self.acc_xy_max 
        self.max_vertical_difference = 5
        self.v_z_min     = -self.v_z_max

        if self.navigation_3d:
            # --- YENİ (GÖREV B): Mantıksal sıralama değişikliği ---
            # ÖNCE action_space'i tanımla
            self.action_space = spaces.Box(
                                        low=np.array([-1.0, -1.0, -1.0]),  # [Δv_xy, Δv_z, yaw_rate]
                                        high=np.array([1.0, 1.0, 1.0]),
                                        dtype=np.float32
                                    )
            
            # ŞİMDİ action_dims'i al (bu 3 olacak)
            action_dims = self.action_space.shape[0]

            # SONRA state_feature_length'i GÜNCELLE
            if self.using_velocity_state:
                self.state_feature_length = 6 + action_dims # 6 (eski) + 3 (eylem) = 9
            else:
                self.state_feature_length = 3 + action_dims # 3 (eski) + 3 (eylem) = 6
            # --- YENİ KOD SONU ---

        else:
            # --- YENİ (GÖREV B): Mantıksal sıralama değişikliği ---
            # ÖNCE action_space'i tanımla
            self.action_space = spaces.Box(
                                                low=np.array([-1.0, -1.0]),  # [Δv_xy, yaw_rate]
                                                high=np.array([1.0, 1.0]),
                                                dtype=np.float32)

            # ŞİMDİ action_dims'i al (bu 2 olacak)
            action_dims = self.action_space.shape[0]

            # SONRA state_feature_length'i GÜNCELLE
            if self.using_velocity_state:
                self.state_feature_length = 4 + action_dims # 4 (eski) + 2 (eylem) = 6
            else:
                self.state_feature_length = 2 + action_dims # 2 (eski) + 2 (eylem) = 4
            # --- YENİ KOD SONU ---

        # --- YENİ (GÖREV B): 'last_action'ı doğru boyutta başlat ---
        # Artık self.action_space.shape (2D veya 3D) belli olduğuna göre,
        # 'None' olan değişkeni doğru boyutta bir numpy dizisi ile güncelliyoruz.
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        # --- YENİ KOD SONU ---

    def reset(self):
        self.client.reset()
        # reset goal
        self.update_goal_pose()

        # reset start
        yaw_noise = self.start_random_angle * np.random.random()

        #speed reset
        self.prev_v_xy_sp = 0.0
        self.prev_v_z_sp = 0.0
        self.v_xy_sp = 0
        self.yaw_rate_cmd_deg = 0
        

        # --- YENİ (GÖREV B): 'last_action' hafızasını sıfırla ---
        self.last_action.fill(0.0)
        # --- YENİ KOD SONU ---

       
        # set airsim pose
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.start_position[0]
        pose.position.y_val = self.start_position[1]
        pose.position.z_val = - self.start_position[2]
        pose.orientation = airsim.to_quaternion(0, 0, yaw_noise)
        self.client.simSetVehiclePose(pose, True)

        self.client.simPause(False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # take off
        self.client.moveToZAsync(-self.start_position[2], 2).join()

        self.client.simPause(True)


    def set_action(self, action):
        
        # --- 1. Son Eylemi Kaydet (Reward Hesaplaması İçin) ---
        self.last_action = action
        
        # -------------------------------------------------------
        # 2. Lineer Hız Kontrolü (DELTA / İVME BAZLI) - KORUNDU
        # -------------------------------------------------------
        # action[0] -> [-1, 1]
        # delta -> İvme * dt (Örn: 0.15 m/s artış)
        delta_v_xy = action[0] * self.v_xy_delta_max * 1.1 #Offset var
        
        self.v_xy_sp = np.clip(
            self.prev_v_xy_sp + delta_v_xy,
            self.v_xy_min,
            self.v_xy_max
        )

        # -------------------------------------------------------
        # 3. Açısal Hız Kontrolü (DIRECT RATE CONTROL) - YENİ
        # -------------------------------------------------------
        # action[1] -> [-1, 1]
        # cmd -> [-60, 60] deg/s (Hız komutu)
        # Artık "prev_yaw" veya "modulo" hesabına GEREK YOK.
        
        self.yaw_rate_cmd_deg = action[1] * self.yaw_rate_max_deg * 1.1 #Offset var 

        # -------------------------------------------------------
        # 4. 3D Navigasyon (Hazırlık)
        # -------------------------------------------------------
        if self.navigation_3d:
            # Şimdilik 3D'yi pasif/sabit tutuyoruz
            self.v_z_sp = 0.0 
        else:
            self.v_z_sp = 0.0

        # -------------------------------------------------------
        # 5. Dünya Koordinatlarına Çevirim
        # -------------------------------------------------------
        # Gövde hızını (ileri git), o anki Yaw açısına göre Dünya (NED) hızına çevir
        self.yaw_rad = self.get_attitude()[2]
        vx_world_sp = self.v_xy_sp * math.cos(self.yaw_rad)
        vy_world_sp = self.v_xy_sp * math.sin(self.yaw_rad)

        # -------------------------------------------------------
        # 6. Komutu Uygula
        # -------------------------------------------------------
        
        # 2D Modu
        if len(action) == 2:
            self.client.moveByVelocityZAsync(
                vx_world_sp, 
                vy_world_sp, 
                -self.start_position[2], # Başlangıç irtifasında kal
                self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                # --- KRİTİK DEĞİŞİKLİK: RATE CONTROL ---
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=self.yaw_rate_cmd_deg)
            )

        # 3D Modu
        elif len(action) == 3:
            self.client.moveByVelocityAsync(
                vx_world_sp, 
                vy_world_sp, 
                -self.v_z_sp, 
                self.dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                # --- KRİTİK DEĞİŞİKLİK: RATE CONTROL ---
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=self.yaw_rate_cmd_deg)
            )

        # 7. Simülasyonu İlerlet
        self.client.simContinueForTime(self.dt)

        # -------------------------------------------------------
        # 8. Hafızayı Güncelle
        # -------------------------------------------------------
        self.prev_v_xy_sp = self.v_xy_sp
        
        # DİKKAT: self.prev_yaw_cmd_deg GÜNCELLEMESİ ARTIK YOK!
        # Çünkü açı (angle) tutmuyoruz, anlık hız (rate) veriyoruz.
        # self.prev_yaw_cmd_deg satırını silebilirsin.

    def update_goal_pose(self):
        # if goal is given by rectangular mode
        if self.goal_rect is None:
            distance = self.goal_distance
            noise = np.random.random()
            angle = noise * self.goal_random_angle  # (0~2pi)
            goal_x = distance * math.cos(angle) + self.start_position[0]
            goal_y = distance * math.sin(angle) + self.start_position[1]
        else:
            goal_x, goal_y = self.get_goal_from_rect(self.goal_rect, self.goal_random_angle)
            self.goal_distance = math.sqrt(goal_x*goal_x + goal_y*goal_y)
        self.goal_position[0] = goal_x
        self.goal_position[1] = goal_y
        self.goal_position[2] = self.start_position[2]
        # print('New goal pose: ', self.goal_position)

    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle
    
    def set_goal(self, distance=None, random_angle=0, rect=None):
        if distance is not None:
            self.goal_distance = distance
        self.goal_random_angle = random_angle
        if rect is not None:
            self.goal_rect = rect
            
    def get_goal_from_rect(self, rect_set, random_angle_set):
        rect = rect_set
        random_angle = random_angle_set
        noise = np.random.random()
        angle = random_angle * noise - math.pi   # -pi~pi
        rect = [-128, -128, 128, 128]
        # goal_x = 100*math.sin(angle)
        # goal_y = 100*math.cos(angle)

        if abs(angle) == math.pi/2:
            goal_x = 0
            if angle > 0:
                goal_y = rect[3]
            else:
                goal_y = rect[1]
        if abs(angle) <= math.pi/4:
            goal_x = rect[2]
            goal_y = goal_x*math.tan(angle)
        elif abs(angle) > math.pi/4 and abs(angle) <= math.pi/4*3:
            if angle > 0:
                goal_y = rect[3]
                goal_x = goal_y/math.tan(angle)
            else:
                goal_y = rect[1]
                goal_x = goal_y/math.tan(angle)
        else:
            goal_x = rect[0]
            goal_y = goal_x * math.tan(angle)
        
        return goal_x, goal_y

    def _get_state_feature(self):
        '''
        @description: update and get current uav state and state_norm 
        @param {type} 
        @return: state_norm
                 normalized state range 0.0-1.0
        '''
        
        distance = self.get_distance_to_goal_2d()
        relative_yaw = self._get_relative_yaw()  # return relative yaw -pi to pi 
        relative_pose_z = self.get_position()[2] - self.goal_position[2]  # current position z is positive
        vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) 

        distance_norm = distance / self.goal_distance 
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5) 

        # current speed and angular speed
        velocity = self.get_velocity()
        linear_velocity_xy = velocity[0]
        linear_velocity_norm = (linear_velocity_xy - self.v_xy_min) / (self.v_xy_max - self.v_xy_min) 
        linear_velocity_z = velocity[1]
        linear_velocity_z_norm = (linear_velocity_z / self.v_z_max / 2 + 0.5) 
        angular_velocity_norm = (velocity[2] / self.yaw_rate_max_rad / 2 + 0.5) 
        # state: distance_h, distance_v, relative yaw, velocity_x, velocity_z, velocity_yaw
        self.state_raw = np.array([distance, relative_pose_z,  math.degrees(
            relative_yaw), linear_velocity_xy, linear_velocity_z,  math.degrees(velocity[2])])
        state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm,
                               linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm])
        #state_norm = np.clip(state_norm, 0, 1)
        
        if self.navigation_3d:
            if self.using_velocity_state == False:
                state_norm = state_norm[:3]
        else:
            state_norm = np.array(
                [state_norm[0], state_norm[2], state_norm[3], state_norm[5]])
            if self.using_velocity_state == False:
                state_norm = state_norm[:2]

        
        # --- YENİ (GÖREV B): 'last_action'ı normalize et ve state'e ekle ---
        # self.last_action, [-1, 1] aralığındadır (Adım B3'te ayarlandı).
        # state_norm, [0, 1] aralığındadır. Normalize et.
        last_action_norm = (self.last_action + 1.0) / 2.0  # [-1, 1] -> [0, 2] -> [0, 1]
        
        # state_norm'u (2, 3, 4, 6, vb. elemanlı olabilir) 
        # ve last_action_norm'u (2 veya 3 elemanlı) birleştir.
        state_norm = np.concatenate([state_norm, last_action_norm]).astype(np.float32) # .astype(np.float32) eklendi
        # --- YENİ KOD SONU ---

        self.state_norm = state_norm
        
        return state_norm

    def _get_relative_yaw(self):
        '''
        @description: get relative yaw from current pose to goal in radian
        @param {type} 
        @return: 
        '''
        current_position = self.get_position()
        # get relative angle
        relative_pose_x = self.goal_position[0] - current_position[0]
        relative_pose_y = self.goal_position[1] - current_position[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = self.get_attitude()[2]

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

    def get_position(self):
        position = self.client.simGetVehiclePose().position 
        return [position.x_val, position.y_val, -position.z_val]

    def get_velocity(self):
        states = self.client.getMultirotorState()
        linear_velocity = states.kinematics_estimated.linear_velocity
        angular_velocity = states.kinematics_estimated.angular_velocity

        velocity_xy = math.sqrt(pow(linear_velocity.x_val, 2) + pow(linear_velocity.y_val, 2))
        velocity_z = linear_velocity.z_val
        yaw_rate = angular_velocity.z_val

        return [velocity_xy, -velocity_z, yaw_rate]

    def get_attitude(self):
        self.state_current_attitude = self.client.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(self.state_current_attitude)

    def get_attitude_cmd(self):
        return [0.0, 0.0, self.prev_yaw_cmd_deg]

    def get_distance_to_goal_2d(self):
        return math.sqrt(pow(self.get_position()[0] - self.goal_position[0], 2) + pow(self.get_position()[1] - self.goal_position[1], 2))