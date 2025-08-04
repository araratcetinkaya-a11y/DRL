'''
Author: Aadi Nath Mishra
Email: aadi.mishra@tuhh.de
Date: 2023-10-02 00:00:00
LastEditTime: 2021-10-02 00:00:00
LastEditors: Aadi Nath Mishra
Description: Airsim environment for motion planning of drone in continuous and discrete action space
'''
import math
import numpy as np
import random
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

#import wandb
import airsim
from airsim.utils import *
import time
import cv2

class AirsimDynamicsEnv(gym.Env):
    '''
    Airsim environment

    ip_address: ip address of the airsim server
    image_shape: shape of the image
    continuous: continuous action space or discrete action space
    '''
    # Define the environment
    def __init__(self, ip_address, image_shape, continuous=True):
        super(AirsimDynamicsEnv, self).__init__()

        # core configuration
        self.ip_address           = ip_address
        self.image_shape          = image_shape   # [H, W, C]
        self.navigation_3d        = False
        self.using_velocity_state = True
        self.continuous           = continuous
        self.enable_render        = True

        H, W = self.image_shape[:2]
        self.obs_buffer   = np.empty((H, W, 2), dtype=np.uint8)
        self.state_buffer = np.zeros((H, W),   dtype=np.uint8)

        # initialize AirSim client, drone settings
        self.initialize_drone()

        # ─── Training bookkeeping ───
        self.step_num               = 0
        self.episode_num            = 0
        self.total_timesteps        = 0
        self.cumulated_episode_reward = 0
        self._episode_stats         = []
        self.max_episode_steps      = 2000
        self.timestep               = 0
        self.cumulated_base_reward    = 0.0
        self.cumulated_bonus_reward   = 0.0
        # NEW: keep a running total of each component
        self.cumulated_reward_step     = 0.0
        self.cumulated_reward_distance = 0.0
        self.cumulated_reward_yaw      = 0.0
        self.cumulated_obs_penalty     = 0.0
        self.cumulated_step_penalty    = 0.0

        # ─── Dynamics parameters ───
        self.dt                    = 0.005
        self.max_depth_meters      = 40.0
        self.min_distance_to_obstacles = 0.0

        # ─── Action‐limits (keep all your original values) ───
        self.v_xy_max         =  1.0
        self.v_xy_min         = -1.0
        self.v_z_max          =  3.0
        self.yaw_max          =  1.0
        self.yaw_rate_max_deg = 45.0
        self.yaw_rate_max_rad = math.radians(self.yaw_rate_max_deg)
        self.accept_radius    = 1
        self.hard_crash_margin = 1
        self.soft_crash_margin = 2

        # ─── Internal state placeholders ───
        self.x   = 0.0
        self.y   = 0.0
        self.z   = 0.0
        self.v_xy = 0.0
        self.v_z  = 0.0
        self.yaw  = 0.0

        # ─── Command placeholders ───
        self.v_xy_sp     = 0.0
        self.v_z_sp      = 0.0
        self.yaw_rate_sp = 0.0

        # ─── Start & Goal positions ───
        self.home_position  = [0.0, 0.0, -3]
        self.start_position = list(self.home_position)
        self.goal_position  = np.array([0.0, 0.0, -3])
        # TODO REMOVE these three lines (moved into reset()):
        # self.goal_distance            = self.get_distance_to_goal_2d()
        # self.previous_target_distance = self.goal_distance
        # self.previous_heading         = self.get_relative_yaw()

        # ─── Observation space (image) ───
        # depth + state ⇒ C = image_shape[2] (usually 2)
        self.observation_space = Box(
            low=0, high=255,
            shape=(image_shape[0], image_shape[1], image_shape[2]),
            dtype=np.uint8
        )

        # ─── Action space & feature‐vector length ───
        if not self.continuous:
            # Discrete 2D nav
            self.action_space         = Discrete(9)
            # features = [distance_to_goal, relative_yaw]
            self.state_feature_length = 2

        else:
            if self.navigation_3d:
                # Continuous 3D: [v_xy, v_z, yaw_rate]
                self.action_space = Box(
                    low  = np.array([self.v_xy_min, -self.v_z_max, -self.yaw_rate_max_rad]),
                    high = np.array([self.v_xy_max,  self.v_z_max,  self.yaw_rate_max_rad]),
                    dtype=np.float32,
                    seed=42
                )
                # features = [dist, yaw, speed_xy, v_z, yaw_rate]
                self.state_feature_length = 5 if self.using_velocity_state else 2

            else:

                # Continuous 2D: [v_xy, yaw_rate]
                self.action_space = Box(
                    low  = np.array([self.v_xy_min, -self.yaw_rate_max_rad]),
                    high = np.array([self.v_xy_max,  self.yaw_rate_max_rad]),
                    dtype=np.float32,
                    seed=42
                )
                # features = [distance_to_goal, relative_yaw] (+ speed_xy if velocity‐state enabled)
                
                if self.using_velocity_state:

                   self.state_feature_length = 3

                else:

                  self.state_feature_length = 2


                

    def reset(self,*, seed=None, options=None):
        """
        Reset the environment for a new episode.

        1) (Optional) Seed NumPy for reproducibility
        2) Teleport & randomize start/goal
        3) Bookkeeping & episode info
        4) Initialize distance/yaw baselines
        5) Grab first observation, print info, unpause sim
        """
    # 1) Gym‐style optional seeding
        super().reset(seed=seed)
        if seed is not None:
                
            np.random.seed(seed)

    # 2) Teleport the drone (and sample new start/goal inside reset_drone)
        
        self.reset_drone()

            # 3) Capture stats about the episode that just ended
        info = {
                'cumulative reward': self.cumulated_episode_reward,
                'episode':           self.episode_num,
                'start':             self.start_position.copy(),
                'goal':              self.goal_position.copy(),
            }

    # 4) Advance counters & clear episode reward
        self.episode_num            += 1
        self.step_num                = 0
        self.cumulated_episode_reward = 0
        self.cumulated_base_reward    = 0.0
        self.cumulated_bonus_reward   = 0.0
        self.cumulated_reward_step      = 0.0
        self.cumulated_reward_distance  = 0.0
        self.cumulated_reward_yaw       = 0.0
        self.cumulated_obs_penalty      = 0.0
        self.cumulated_step_penalty     = 0.0

        # 5) Initialize baselines now that drone is in place
        if self.navigation_3d:
                self.goal_distance = self.get_distance_to_goal_3d()
        else:
            self.goal_distance = self.get_distance_to_goal_2d()
            self.previous_target_distance = self.goal_distance
            self.previous_heading         = self.get_relative_yaw()
            self.prev_obs_phi = 0.0
            # 6) Grab first observation & resume sim
            obs, _ = self.get_obs()
            #print(info)
            self.client.simPause(False)

        return obs, info

    def reset_drone(self):
        '''
        Reset drone to initial position
        '''
        self.client.reset()

        # 2) Sample new continuous offsets (floats) within hall bounds
        #    (adjust these ranges to match your 40×24 m hall)


        dx_start = round(np.random.uniform(-2, 2),2)
        dy_start = round(np.random.uniform(-10, 10),2)
        dx_goal  = round(np.random.uniform(29, 34),2)
        dy_goal  = round(np.random.uniform(-10, 10),2)

         
        # 3) Apply offsets relative to home_position
        self.start_position = [
            self.home_position[0] + dx_start,
            self.home_position[1] + dy_start,
            self.home_position[2]
        ]
        self.goal_position = np.array([
            self.home_position[0] + dx_goal,
            self.home_position[1] + dy_goal,
            self.home_position[2]
        ])

         # 4) Build a brand‐new Pose at start_position
        start_pose = airsim.Pose()
        start_pose.position.x_val = self.start_position[0]
        start_pose.position.y_val = self.start_position[1]
        start_pose.position.z_val = self.start_position[2]
        # keep drone level on takeoff
        start_pose.orientation = airsim.to_quaternion(0, 0, 0)

        # 5) Push to simulator & prep the drone
        self.client.simSetVehiclePose(start_pose, ignore_collision=False)
        self.client.simPause(False)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.hoverAsync().join()
        # time.sleep(0.2)   # give it 200 ms to settle
        # then pause until reset() unpauses for step()
        self.client.simPause(True)

        # 6) Reset your own state variables so t=0 obs is clean
        self.x            = self.start_position[0]
        self.y            = self.start_position[1]
        self.z            = self.start_position[2]
        self.v_xy         = 0.0
        self.v_z          = 0.0
        self.yaw          = 0.0
        self.v_xy_sp      = 0.0
        self.v_z_sp       = 0.0
        self.yaw_rate_sp  = 0.0
        

    def close(self):
        self.client.reset()
    
    def step(self, action):
        """
        1) Apply the action for self.dt seconds
        2) Snapshot obs (with error handling)
        3) Compute reward & done/truncation flags
        4) Accumulate & package info
        5) Print end‐of‐episode reason
        6) Record + plot per‐episode stats to W&B
        7) Return obs, reward, done, truncated, info
        """
        import time

        # 0) Sanity check
        assert self.action_space.contains(action), f"{action!r} invalid"

        # Start total timer
        t0 = time.time()

        # 1) Execute the action (advances sim for dt via .join())
        t_a0 = time.time()
        self.do_action(action)
        t_a1 = time.time()

        # 2) Snapshot the new observation, with error handling
        t_o0 = time.time()
        try:
            observation, _ = self.get_obs()
        except Exception as e:
            t_o1 = time.time()
            # handle exception as before...
            # For brevity, omitted exception block here
            raise
        t_o1 = time.time()

        # 3) Step counting and terminal flags
        t_sc0 = time.time()
        self.step_num += 1
        self.total_timesteps += 1
        is_success = self.is_in_desired_pose()
        is_crash   = self.is_crashed()
        done       = is_success or is_crash
        t_sc1 = time.time()

        # 4) Compute reward components
        t_r0 = time.time()
        reward_dict = self.compute_reward_final()
        t_r1 = time.time()

        # 5) Accumulate rewards
        t_acc0 = time.time()
        reward_step     = reward_dict["reward_step"]
        reward_distance = reward_dict["reward_distance"]
        reward_yaw      = reward_dict["reward_yaw"]
        obs_penalty     = reward_dict["obs_penalty"]
        step_penalty    = reward_dict["step_penalty"]
        shaping         = reward_dict["shaping"]
        base_reward     = reward_dict["base_reward"]

        self.cumulated_reward_step     += reward_step
        self.cumulated_reward_distance += reward_distance
        self.cumulated_reward_yaw      += reward_yaw
        self.cumulated_obs_penalty     += obs_penalty
        self.cumulated_step_penalty    += step_penalty
        self.cumulated_base_reward     += base_reward
        self.cumulated_episode_reward  += base_reward
        t_acc1 = time.time()

        # 6) Bonus and truncation logic
        t_t0 = time.time()
        if is_success:
            bonus = 10; terminal_reason = "goal"
        elif is_crash:
            bonus = -10; terminal_reason = "crash"
        elif self.step_num >= self.max_episode_steps:
            bonus = -10; terminal_reason = "max_steps"
        else:
            bonus = 0; terminal_reason = None

        self.cumulated_episode_reward += bonus
        self.cumulated_bonus_reward    += bonus
        reward = base_reward + bonus

        if self.step_num >= self.max_episode_steps:
            truncated = True
            terminal_reason = terminal_reason or "max_steps"
        elif self.cumulated_episode_reward <= -30:
            truncated = True
            terminal_reason = terminal_reason or "low_reward"
        else:
            truncated = False
        t_t1 = time.time()

        # 7) Build info and finalize
        t_i0 = time.time()
        info = {
            'is_success': is_success,
            'is_crash':   is_crash,
            'step_num':   self.step_num,
            'start':      self.start_position.copy(),
            'goal':       self.goal_position.copy(),
        }
        t_i1 = time.time()

        # End total timer
        t1 = time.time()
        total = (t1 - t0)*1000
        a = (t_a1 - t_a0)*1000
        o = (t_o1 - t_o0)*1000
        sc = (t_sc1 - t_sc0)*1000
        r = (t_r1 - t_r0)*1000
        acc = (t_acc1 - t_acc0)*1000
        tt = (t_t1 - t_t0)*1000
        io = (t_i1 - t_i0)*1000

        print(f"step total={total:.1f} ms | action={a:.1f} | obs={o:.1f} | stepcount={sc:.1f} | reward={r:.1f} | accumulate={acc:.1f} | trunc={tt:.1f} | info={io:.1f}")

        return observation, reward, bool(done), bool(truncated), info




    def get_obs(self):
        """
        Tekrar eden state-özelliği kaldırıldı.
        Derinlik + state-kanalı oluşturma ve info döndürme tek geçişte.
        """
        H, W = self.image_shape[:2]

        # 1) Derinlik kanalını al ve yeniden boyutlandır
        depth = self.get_depth_image()  # float32, (h0×w0)
        depth_rs = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        # en yakın engel
        self.min_distance_to_obstacles = float(depth_rs.min())

        # normalize & uint8
        depth_chan = (np.clip(depth_rs, 0, self.max_depth_meters)
                    / self.max_depth_meters * 255
                    ).astype(np.uint8)

        # 2) State feature’u sadece bir kez hesapla
        state_vec = self.get_state_feature()  # uint8 vektör
        n = state_vec.shape[0]
        # hazır bir state-buffer (__init__’te pre-alloc edilebilir)
        state_chan = np.zeros((H, W), dtype=np.uint8)
        state_chan[0, :n] = state_vec         # tek atama, kopya yok

        # 3) Önceden ayırdığın obs_buffer’a in-place yaz
        #    (__init__’te: self.obs_buffer = np.empty((H, W, 2), np.uint8))
        self.obs_buffer[..., 0] = depth_chan
        self.obs_buffer[..., 1] = state_chan

        # 4) Tek bilgi yapısı
        info = {
            'state':             state_vec.astype(np.float32),
            'min_obstacle_dist': self.min_distance_to_obstacles
        }

        return self.obs_buffer, info

            
    def get_obs_image(self):
        
        """
        Constructs a two-channel observation:
        • Channel 0 = depth image (scaled to [0,255] uint8)
        • Channel 1 = state features placed in the first row

        Output shape: (H, W, 2)
        """
        # dimensions
        H, W = self.image_shape[0], self.image_shape[1]

        # — 1) Depth channel —
        depth = self.get_depth_image()  # float32, in meters
        depth_rs = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        # track nearest obstacle for crash logic
        self.min_distance_to_obstacles = float(depth_rs.min())

        # clip & scale into [0,255]
        depth_clipped = np.clip(depth_rs, 0.0, self.max_depth_meters)
        depth_chan = (depth_clipped / self.max_depth_meters * 255.0).astype(np.uint8)

        # — 2) State channel —
        state_feat = self.get_state_feature()           # e.g. [dist, yaw, speed]
        n_feats    = state_feat.shape[0]
        sf_u8      = np.clip(state_feat, 0.0, 255.0).astype(np.uint8)

        state_chan = np.zeros((H, W), dtype=np.uint8)
        state_chan[0, :n_feats] = sf_u8               # place features in top row

        # — 3) Stack into H×W×2 and return —
        return np.stack([depth_chan, state_chan], axis=-1)

    def get_depth_image(self):
        max_retries = 3
        for attempt in range(1, max_retries+1):
            responses = self.client.simGetImages([
                airsim.ImageRequest("1",
                    airsim.ImageType.DepthPerspective,
                    pixels_as_float=False,
                    compress=True)
            ])
            if not responses or responses[0].width == 0:
                time.sleep(0.1); continue
            resp = responses[0]
            try:
                buf = np.frombuffer(resp.image_data_uint8, np.uint8)
                # GRAYSCALE olarak decode
                depth_u8 = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
                # metre cinsine geri ölçekleme
                depth_img = (depth_u8.astype(np.float32)/255.0)*self.max_depth_meters

                if getattr(self, "enable_render", False):
                    cv2.imshow("Depth", depth_u8); cv2.waitKey(1)

                return depth_img
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("get_depth_image failed")



    def initialize_drone(self):

        #self.client = airsim.MultirotorClient(ip=self.ip_address)^
        # ip_address içinde “IP:PORT” formatı bekleyelim
        addr = self.ip_address
        if ":" in addr:
            ip, port_str = addr.split(":")
            port = int(port_str)
        else:
            ip = addr
            port = 41451  # default port
        # Artık hem ip hem port veriyoruz

        print(f"[init] AirSim client connecting to {ip}:{port}")
        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)


    def compute_reward_final(self):
            """
            Compute a scalar reward combining:
            • Progress toward the goal (“reward_step”)
            • Proximity to the goal    (“reward_distance”)
            • Yaw alignment            (“reward_yaw”)
            • Quadratic penalty for getting too close to obstacles (“obs_penalty”)
            • Small per‐step efficiency penalty (“step_penalty”)
            """

            # 1) Compute current distance to goal (2D or 3D)
            if self.navigation_3d:
                current_dist = self.get_distance_to_goal_3d()
            else:
                current_dist = self.get_distance_to_goal_2d()
                prev_dist = self.previous_target_distance

            # 2) Progress reward: thresholded dynamic normalization Dynamic Normalization 
            delta = prev_dist - current_dist 
            step_threshold      = 1 #%% Need to be changed to variable during Tune 
            if delta >= step_threshold:  # e.g. 1.0 m
                reward_step = (delta / self.goal_distance)*3  # ε clamp // 6 ile carptim //2
                #reward_step = (delta / max(prev_dist, 0.1))
            else:
                reward_step = 0.0

            # 3) Distance‐to‐goal reward: potential‐difference shaping
            
            alpha_dist             = 1  #%% Need to be changed to variable during Tune 
            prev_phi_dist          = 1.0 - (prev_dist / self.goal_distance) ** 2
            curr_phi_dist          = 1.0 - (current_dist / self.goal_distance) ** 2
            reward_distance        =  (alpha_dist * curr_phi_dist - prev_phi_dist) *3 # //3 ile carptim

            # 4) Yaw‐alignment reward: potential‐difference + smooth rate penalty
            
            alpha_yaw           = 1         #%% Need to be changed to variable during Tune 
            curr_err = self.get_relative_yaw()
            prev_err = self.previous_heading
            prev_phi_yaw = -abs(prev_err)
            curr_phi_yaw = -abs(curr_err)
            reward_yaw = (alpha_yaw * curr_phi_yaw - prev_phi_yaw)*0.5
            

            # 5) Obstacle‐proximity penalty: potential‐difference shaping
            k_obs           = 1                             #%% Need to be changed to variable during Tune
            thresh_obs      = 0.5                           #%% Need to be changed to variable during Tune
            alpha_obs       = 1                             #%% Need to be changed to variable during Tune
            prev_phi_obs    = self.prev_obs_phi
            d_obs           = self.min_distance_to_obstacles

            if d_obs < thresh_obs:
                gamma_obs = (thresh_obs - d_obs) / thresh_obs
                curr_phi_obs = -k_obs * (gamma_obs ** 2)
            else:
                curr_phi_obs = 0.0

            obs_penalty = alpha_obs * curr_phi_obs - prev_phi_obs

            # 6) Small per‐step efficiency penalty
            base_pen   = 0.005     #%% Need to be changed to variable during Tune 0,05 // 007
            dist_coeff = 1          #%% Need to be changed to variable during Tune
            step_penalty = -base_pen * (1.0 + dist_coeff * (current_dist / self.goal_distance))

            # 7) Combine shaping terms with unique weights
            
            weight_reward_step      = 1 #%% Need to be changed to variable during Tune
            weight_reward_distance  = 1 #%% Need to be changed to variable during Tune 
            weight_yaw_reward       = 1 #%% Need to be changed to variable during Tune 
            
            shaping = weight_reward_step * reward_step \
                    + weight_reward_distance * reward_distance \
                    + weight_yaw_reward * reward_yaw

            # 8) Final reward
            base_reward = shaping + obs_penalty + step_penalty

            # ────────────────────────────────────────────────────────────────────
            # Update stored state _once_ at end
            self.previous_target_distance = current_dist
            self.previous_heading         = curr_err
            self.prev_obs_phi             = curr_phi_obs

            return {
                "reward_step":      reward_step,
                "reward_distance":  reward_distance,
                "reward_yaw":       reward_yaw,
                "obs_penalty":      obs_penalty,
                "step_penalty":     step_penalty,
                "shaping":          shaping,
                "base_reward":      base_reward,
                # **new** raw / intermediate values:
                "delta":            delta,            # raw progress (m)
                "phi_prev_dist":    prev_phi_dist,    # un-weighted prev potential
                "phi_curr_dist":    curr_phi_dist,    # un-weighted curr potential
                "phi_prev_yaw":     prev_phi_yaw,
                "phi_curr_yaw":     curr_phi_yaw,
            }

    def do_action(self, action):
        '''
        Take action

        action: action to do

        return: None
        '''
        if self.continuous:
            
            self.v_xy_sp = np.clip(action[0]*10, 0.0, 8)
            self.yaw_rate_sp = np.clip(action[1], -self.yaw_rate_max_rad, self.yaw_rate_max_rad)

            if self.navigation_3d:
                self.v_z_sp = float(action[1])
            else:
                self.v_z_sp = 0

            self.yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)[2]
            self.yaw_sp = self.yaw + self.yaw_rate_sp * self.dt
            if self.yaw_sp > math.radians(180):
                self.yaw_sp -= math.pi * 2
            elif self.yaw_sp < math.radians(-180):
                self.yaw_sp += math.pi * 2

            vx = self.v_xy_sp * math.cos(self.yaw_sp)
            vy = self.v_xy_sp * math.sin(self.yaw_sp)
            
            #self.client.simPause(False)
            if len(action) == 2:
                
                self.client.moveByVelocityZAsync(
                    vx, vy, self.start_position[2], self.dt,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))
                )    

                '''
                self.client.moveByVelocityZAsync(vx, vy, self.start_position[2], self.dt,
                                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))).join()'''
                # Prevent from swaying
                #self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=0.01).join()

            elif len(action) == 3:
                self.client.moveByVelocityAsync(vx, vy, self.v_z_sp, self.dt,
                                                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(self.yaw_rate_sp))).join()
                
                # Prevent from swaying
                #self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=self.dt).join() 
        else:
            self.move_discrete(action)



    
    ############### HELPER FUNCTIONS ###############
    # Define drone motion functions here

    def move_discrete(self, select_action):
        speed = 0.4

        if select_action == 0:
            vy, vz = (-speed, -speed)
        elif select_action == 1:
            vy, vz = (0, -speed)
        elif select_action == 2:
            vy, vz = (speed, -speed)
        elif select_action == 3:
            vy, vz = (-speed, 0)
        elif select_action == 4:
            vy, vz = (0, 0)
        elif select_action == 5:
            vy, vz = (speed, 0)
        elif select_action == 6:
            vy, vz = (-speed, speed)
        elif select_action == 7:
            vy, vz = (0, speed)
        else:
            vy, vz = (speed, speed)
                
        self.client.moveByVelocityBodyFrameAsync(speed, float(vy), float(vz), duration=self.dt).join()
        self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=self.dt)
      
    def get_relative_yaw(self):
        """
        Get the signed yaw‐error from the drone’s current heading to the goal.

        Returns:
        yaw_error (float): angle in radians ∈ [−π, π],  
                            positive means you must yaw *left* to face the goal,  
                            negative means yaw *right*.
        """
        # 1) Drone’s (x,y) and goal’s (x,y)
        current_x, current_y, _ = self.get_position()
        goal_x,    goal_y,    _ = self.goal_position

        # 2) Bearing from drone → goal
        target_bearing = math.atan2(goal_y - current_y,
                                    goal_x - current_x)

        # 3) Current yaw (roll,pitch,yaw) from quaternion
        _, _, current_yaw = self.get_attitude()

        # 4) Compute error and wrap into [−π, +π]
        yaw_error = target_bearing - current_yaw
        if yaw_error > math.pi:
            yaw_error -= 2 * math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2 * math.pi

        return yaw_error

    def get_position(self):
        '''
        Get current position
        
        return: [x, y, z]
        '''
        position = self.client.simGetVehiclePose().position 
        return [position.x_val, position.y_val, position.z_val]

    def get_velocity(self):
        """
        Get the drone’s current velocity.

        Returns:
        [velocity_xy, velocity_z, yaw_rate] where
            • velocity_xy (float) = √(vx² + vy²) in m/s
            • velocity_z  (float) = vertical speed in m/s (positive up)
            • yaw_rate    (float) = angular velocity around Z in rad/s
        """
        # 1) Query the full multirotor state
        state = self.client.getMultirotorState()
        lin   = state.kinematics_estimated.linear_velocity
        ang   = state.kinematics_estimated.angular_velocity

        # 2) Horizontal speed magnitude
        velocity_xy = math.hypot(lin.x_val, lin.y_val)

        # 3) Vertical speed: invert so positive = up
        velocity_z  = -lin.z_val

        # 4) Yaw‐rate about the body‐Z axis
        yaw_rate    = ang.z_val

        return [velocity_xy, velocity_z, yaw_rate]

    def get_attitude(self):
        '''
        Get current attitude
        '''
        return airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)
    
    def calculate_distance_btw_points(self, p1, p2):
        return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))

    def getEquidistantPoints(self, p1, p2, parts):
        '''
        Get equidistant points between two points
        '''
        l1 = np.linspace(p1[0], p2[0], parts+1)
        l1 = l1[1:]
        l2 = np.linspace(p1[1], p2[1], parts+1)
        l2 = l2[1:]
        return zip(l1,l2)
        
    def get_distance_to_goal_2d(self):
        """
        Get distance to the goal in the XY plane.
        """
        x, y, _     = self.get_position()
        gx, gy, _   = self.goal_position.tolist()
        return math.hypot(gx - x, gy - y)

    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        '''
        Get distance between Point and Line

        pointX, pointY: point position
        lineX1, lineY1, lineX2, lineY2: line position

        return: distance
        '''
        a = lineY2-lineY1
        b = lineX1-lineX2
        c = lineX2*lineY1-lineX1*lineY2
        dis = (math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b, 0.5))

        return dis
    
    def get_distance_to_goal_3d(self):
        '''
        Get distance to goal in 3D
        '''
        current_pose = self.get_position()
        goal_pose = self.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))
    
    def get_state_feature(self):
        """
        Build and return the raw state‐feature vector for embedding in the image:

        • In 2D navigation:
            – [distance_to_goal, relative_yaw]                if using_velocity_state=False
            – [distance_to_goal, relative_yaw, speed_xy]      if using_velocity_state=True

        • In 3D + velocity mode (using_velocity_state=True):
            [distance_to_goal_3d, relative_yaw, speed_xy, vz, yaw_rate]

        All values are normalized into [0,255] and cast to uint8.
        """
        if self.navigation_3d:
            # 3D navigation + velocity
            distance     = self.get_distance_to_goal_3d()
            velocity_xy, velocity_z, yaw_rate = self.get_velocity()
            yaw_error    = self.get_relative_yaw()

            # normalize into [0,255]
            distance_norm   = (distance / self.goal_distance) * 255.0
            yaw_norm        = (yaw_error / (2 * math.pi) + 0.5) * 255.0
            speed_xy_norm   = (velocity_xy / self.v_xy_max) * 255.0
            vz_norm         = ((velocity_z / self.v_z_max) + 1.0) * 127.5
            yaw_rate_norm   = ((yaw_rate   / self.yaw_rate_max_rad) + 1.0) * 127.5

            state = np.array([
                distance_norm,
                yaw_norm,
                speed_xy_norm,
                vz_norm,
                yaw_rate_norm
            ], dtype=np.uint8)

        else:
            # 2D navigation
            distance   = self.get_distance_to_goal_2d()
            yaw_error  = self.get_relative_yaw()

            distance_norm = (distance / self.goal_distance) * 255.0
            yaw_norm      = (yaw_error / (2 * math.pi) + 0.5) * 255.0

            if self.using_velocity_state:
                # include horizontal speed
                velocity_xy, _, _ = self.get_velocity()
                speed_norm = (velocity_xy / self.v_xy_max) * 255.0
                state = np.array([distance_norm, yaw_norm, speed_norm], dtype=np.uint8)
            else:
                state = np.array([distance_norm, yaw_norm], dtype=np.uint8)

        # clip to [0,255] and ensure uint8

        return np.clip(state, 0, 255).astype(np.uint8)

    def is_done(self):
        
        """
        Returns True if the episode has reached a terminal state:
        • success  → drone has reached within accept_radius of goal
        • crash    → collision detected or too‐close “hard crash” margin
        """
        return self.is_in_desired_pose() or self.is_crashed()


    def is_in_desired_pose(self):
        """
        Check whether the drone is “at” the goal.

        Returns:
        bool – True if the current distance to the goal is ≤ accept_radius.
        """
        if self.navigation_3d:
            dist = self.get_distance_to_goal_3d()
        else:
            dist = self.get_distance_to_goal_2d()
        return dist <= self.accept_radius

    def is_crashed(self):
        """
        Returns True if the drone has physically collided (AirSim’s collision flag).
        """
        # Only physical collisions count for “crash” in sim-to-real transfer

        col = self.client.simGetCollisionInfo()

        if col.has_collided:

            return True
    
