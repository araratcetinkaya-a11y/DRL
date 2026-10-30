
import gym
from gym import spaces
import airsim
from configparser import NoOptionError
import keyboard
import msgpackrpc
import time
import torch as th
import numpy as np
import math
import cv2

from .dynamics.multirotor_simple import MultirotorDynamicsSimple
from .dynamics.multirotor_airsim import MultirotorDynamicsAirsim
from .dynamics.fixedwing_simple import FixedwingDynamicsSimple
# from .lgmd.LGMD import LGMD

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

class CameraError(Exception):
    """Raised when simGetImages fails repeatedly."""
    pass

class AirsimGymEnv(gym.Env, QtCore.QThread):
    # pyqt signal for visualization
    action_signal = pyqtSignal(int, np.ndarray)
    state_signal = pyqtSignal(int, np.ndarray)
    attitude_signal = pyqtSignal(int, np.ndarray, np.ndarray)
    reward_signal = pyqtSignal(int, float, float)
    pose_signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    lgmd_signal = pyqtSignal(float, float, np.ndarray)

    def __init__(self) -> None:
        super().__init__()
        np.set_printoptions(formatter={'float': '{: 4.2f}'.format},
                            suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)
        print("init airsim-gym-env.")
        self.model = None
        self.data_path = None
        self.lgmd = None

        # Reward Function Coef
        
        self.SAFE_DISTANCE = 2            # Güvenli bölge (metre)
        self.k_obs_penalty = 0.5            # Engel cezasının sertliği
        self.k_speed_near_obs = 50         # Engele yakın hız cezasının katsayısı
        self.k_jerk_fwd = 0.025              # İleri sarsıntı cezası (küçük başlayın)
        self.k_jerk_yaw = 0.03               # Dönüş sarsıntı cezası
        self.max_safe_speed_near_obs = 3.5

        # --- YENİ (GÖREV A): Son eylemi saklamak için ---
        self.last_action = None

    def set_config(self, cfg):
        """get config from .ini file
        """
        self.cfg = cfg
        self.env_name = cfg.get('options', 'env_name')
        self.dynamic_name = cfg.get('options', 'dynamic_name')
        self.keyboard_debug = cfg.getboolean('options', 'keyboard_debug')
        self.generate_q_map = cfg.getboolean('options', 'generate_q_map')
        self.perception_type = cfg.get('options', 'perception')
        self.k_frameskip = cfg.getint('options', 'k_frameskip', fallback=1)
        if self.k_frameskip > 1:
            print(f"--- Action Repetition Aktif: k_frameskip = {self.k_frameskip} ---")

        # create LGMD agent
        if self.perception_type == 'lgmd':
            self.lgmd = LGMD(type='origin',  p_threshold=50, s_threshold=0, Ki=2, i_layer_size=3, activate_coeff=1, use_on_off=True)
            self.split_out_last = np.array([0, 0, 0, 0, 0])

        print('Environment: ', self.env_name, "Dynamics: ", self.dynamic_name,
              'Perception: ', self.perception_type)

        # set dynamics
        if self.dynamic_name == 'SimpleFixedwing':
            self.dynamic_model = FixedwingDynamicsSimple(cfg)
        elif self.dynamic_name == 'SimpleMultirotor':
            self.dynamic_model = MultirotorDynamicsSimple(cfg)
        elif self.dynamic_name == 'Multirotor':
            self.dynamic_model = MultirotorDynamicsAirsim(cfg)
        else:
            raise Exception("Invalid dynamic_name!", self.dynamic_name)

        # set start and goal position according to different environment
        if self.env_name == 'NH_center':
            start_position = [0, 0, 3]
            goal_rect = [-128, -128, 128, 128]  # rectangular goal pose
            goal_distance = 90
            self.dynamic_model.set_start(
                start_position, random_angle=math.pi*2)
            self.dynamic_model.set_goal(random_angle=math.pi*2, rect=goal_rect)
            self.work_space_x = [-140, 140]
            self.work_space_y = [-140, 140]
            self.work_space_z = [0.5, 20]
            self.max_episode_steps = 1000
        elif self.env_name == 'NH_tree':
            start_position = [110, 180, 5]
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model.set_goal(distance=90, random_angle=0)
            self.work_space_x = [start_position[0],
                                 start_position[0] + goal_distance + 10]
            self.work_space_y = [
                start_position[1] - 30, start_position[1] + 30]
            self.work_space_z = [0.5, 10]
            self.max_episode_steps = 400
        elif self.env_name == 'City':
            start_position = [40, -30, 40]
            goal_position = [280, -200, 40]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 350]
            self.work_space_y = [-300, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 400
        elif self.env_name == 'City_400':
            # note: the start and end points will be covered by update_start_and_goal_pose_random function
            start_position = [0, 0, 50]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-220, 220]
            self.work_space_y = [-220, 220]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 800
        elif self.env_name == 'Tree_200':
            # note: the start and end points will be covered by
            # update_start_and_goal_pose_random function
            start_position = [0, 0, 8]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 600
        elif self.env_name == 'SimpleAvoid':
            start_position = [0, 0, 3]
            goal_distance = 50
            self.dynamic_model.set_start(
                start_position, random_angle=math.pi*2)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=math.pi*2)
            self.work_space_x = [
                start_position[0] - goal_distance - 10, start_position[0] + goal_distance + 10]
            self.work_space_y = [
                start_position[1] - goal_distance - 10, start_position[1] + goal_distance + 10]
            self.work_space_z = [0.5, 50]
            self.max_episode_steps = 1000
        elif self.env_name == 'Forest':
            start_position = [0, 0, 10]
            goal_position = [280, -200, 50]
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model._set_goal_pose_single(goal_position)
            self.work_space_x = [-100, 100]
            self.work_space_y = [-100, 100]
            self.work_space_z = [0, 100]
            self.max_episode_steps = 300
        elif self.env_name == 'Trees':
            start_position = [0, 0, 5]
            goal_distance = 70
            self.dynamic_model.set_start(
                start_position, random_angle=math.pi*2)
            self.dynamic_model.set_goal(
                distance=goal_distance, random_angle=math.pi*2)
            self.work_space_x = [
                start_position[0] - goal_distance - 10, start_position[0] + goal_distance + 10]
            self.work_space_y = [
                start_position[1] - goal_distance - 10, start_position[1] + goal_distance + 10]
            self.work_space_z = [0.5, 50]
            self.max_episode_steps = 500
        else:
            raise Exception("Invalid env_name!", self.env_name)
        
        if self.k_frameskip > 1:
            original_steps = self.max_episode_steps
            self.max_episode_steps = int(self.max_episode_steps / self.k_frameskip)
            print(f"Bölüm uzunluğu (max_episode_steps) {original_steps} fizik adımından {self.max_episode_steps} AJAN adımına güncellendi.")
        # --- YENİ KOD SONU ---

        self.client = self.dynamic_model.client
        try:
            print("Grafik ayarları Rapor 2.1'e göre sıfırlanıyor...")
            self.client.simConsoleExecute("sg.ScalabilityLevel 0")
            self.client.simConsoleExecute("sg.PostProcessQuality 0")
            self.client.simConsoleExecute("sg.ShadowQuality 0")
            self.client.simConsoleExecute("sg.TextureQuality 0")
            self.client.simConsoleExecute("sg.EffectsQuality 0")
            self.client.simConsoleExecute("sg.FoliageQuality 0")
            self.client.simConsoleExecute("r.ScreenPercentage 10") # Ana ekranı %10 çözünürlüğe düşür
            print("Grafik ayarları başarıyla sıfırlandı.")
        except Exception as e:
            print(f"UYARI: Konsol komutları gönderilemedi: {e}")
            
        self.state_feature_length = self.dynamic_model.state_feature_length
        self.cnn_feature_length = self.cfg.getint('options', 'cnn_feature_num')

        # training state
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0

        # other settings
        self.crash_distance = cfg.getint('environment', 'crash_distance')
        self.accept_radius = cfg.getint('environment', 'accept_radius')

        self.max_depth_meters = cfg.getint('environment', 'max_depth_meters')
        self.screen_height = cfg.getint('environment', 'screen_height')
        self.screen_width = cfg.getint('environment', 'screen_width')

        self.trajectory_list = []

        # observation space vector or image
        if self.perception_type == 'vector' or self.perception_type == 'lgmd':
            self.observation_space = spaces.Box(low=0, high=1,
                                                shape=(1,
                                                       self.cnn_feature_length + self.state_feature_length),
                                                dtype=np.float32)
        else:
           self.observation_space = spaces.Box(low=0.0,  # <-- Değişti (float)
                                                high=1.0, # <-- Değişti (float)
                                                shape=(self.screen_height, self.screen_width, 2),
                                                dtype=np.uint8 # <-- En önemli değişiklik (uint8 -> float32)
                                            ) #en başarılı model için lazım bu 
        ''' self.observation_space = spaces.Box(low=0, 
                                                high=255, 
                                                shape=(2, self.screen_height, self.screen_width), 
                                                dtype=np.uint8)'''


        self.action_space = self.dynamic_model.action_space
        print("action_space.low  =", self.action_space.low)
        print("action_space.high =", self.action_space.high)
        print("v_xy_min/max =", self.dynamic_model.v_xy_min, self.dynamic_model.v_xy_max,
            " yaw_rate_max(rad) =", self.dynamic_model.yaw_rate_max_rad)

        self.reward_type = None
        try:
            self.reward_type = cfg.get('options', 'reward_type')
            print('Reward type: ', self.reward_type)
        except NoOptionError:
            self.reward_type = None

    def _rb_reset(self):
        # episode boyunca toplanacak terimler
        self._rb_sums = {
            "r_dist": 0.0, "c_pose": 0.0, "c_obs": 0.0,
            "c_act_fwd": 0.0, "c_act_yaw": 0.0, 
            "c_yaw": 0.0, "r_total": 0.0, "c_spd_obs": 0.0,
            # --- YENİ (GÖREV A) ---
            "c_jerk_fwd": 0.0, "c_jerk_yaw": 0.0
        }

        self._rb_steps = 0
        self._d0 = None          # bölüm başı hedef mesafesi
        self._dT = None          # o andaki hedef mesafesi
        self._min_obs = float("inf")

    def _rb_update_step(self, terms: dict, diag: dict):
        # eksik kalırsa 0 say (KeyError önler)
        
        terms.setdefault("c_spd_obs", 0.0)
        terms.setdefault("c_act_fwd", 0.0) 
        terms.setdefault("c_act_yaw", 0.0)
        # her adımda birikenleri güncelle

        self._rb_steps += 1
        for k, v in terms.items():
            if k in self._rb_sums: # Sadece _rb_sums'da tanımlı anahtarları topla
                self._rb_sums[k] += float(v)

        if "d_now" in diag:
            self._dT = float(diag["d_now"])
            if self._d0 is None:
                self._d0 = float(diag["d_now"])
        if "min_obs_dist" in diag:
            self._min_obs = min(self._min_obs, float(diag["min_obs_dist"])) 

    def _print_ep_summary(self, info: dict):

        if "ep_sums" not in info:
            return
        s = info["ep_sums"]

        reason = s.get("ep/done_reason_str", "unknown")
        elen   = s.get("ep/len", 0)
        prog   = s.get("ep/progress_frac", float("nan"))
        minobs = s.get("ep/min_obs_dist", float("nan"))

        r_tot  = s.get("R_ep/r_total", 0.0)
        r_dist = s.get("R_ep/r_dist",  0.0)
        c_obs  = s.get("R_ep/c_obs",   0.0)
        c_act_fwd = s.get("R_ep/c_act_fwd", 0.0)
        c_act_yaw = s.get("R_ep/c_act_yaw", 0.0)
        # c_step = s.get(...) <- SİLİNDİ
        c_spd_obs = s.get("R_ep/c_spd_obs", 0.0)
        c_jerk_fwd = s.get("R_ep/c_jerk_fwd", 0.0)
        c_jerk_yaw = s.get("R_ep/c_jerk_yaw", 0.0)

        # Log satırı güncellendi ("step" kaldırıldı)
        print(
            f"[EP {self.episode_num:05d}] len={elen:3d}  reason={reason:7s}  "
            f"progress={prog:5.2f}  min_obs={minobs:5.2f}  R_total={r_tot:7.2f} | "
            f"dist={r_dist:6.2f}  - obs={c_obs:6.2f} - fwd={c_act_fwd:6.2f} - "
            f"yaw={c_act_yaw:6.2f} - spd_obs={c_spd_obs:6.2f} - " # YENİ: Satır sonu tire
            # --- YENİ (GÖREV A) ---
            f"j_fwd={c_jerk_fwd:6.2f} - j_yaw={c_jerk_yaw:6.2f}"
        )

    def _print_ep_header(self):
        # Başlıktan "step" kaldırıldı
       print("EP     len  reason   progress  min_obs   R_total |  dist   - obs    - fwd    - yaw    - spd_obs - j_fwd  - j_yaw")

    def reset(self):
        self.dynamic_model.reset()

        if self.last_action is None:
            self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        else:
            self.last_action.fill(0.0)

        # episode sayaçları
        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0.0

        # hedef mesafesi (PBRS için)
        self.dynamic_model.goal_distance = self.dynamic_model.get_distance_to_goal_2d()
        self.previous_distance_from_des_point = self.dynamic_model.goal_distance

        # rollout buffer
        self.trajectory_list = []
        self._rb_reset()

        d0_dist = self.get_distance_to_goal_3d()
        self._d0 = d0_dist
        self._dT = d0_dist

        obs = self.get_obs()
        return obs


    # airsim_env_new_reward_func.py

    def step(self, action):
        # --- Action Repetition değişkenleri ---
        total_reward = 0.0
        obs = None
        done = False
        info = {}

        # Aynı ajanın aksiyonunu k_frameskip kez fizik adımı olarak uygula
        for _ in range(self.k_frameskip):

            # 1) Aksiyonu dinamiğe uygula
            if self.dynamic_name == 'SimpleFixedwing':
                self.dynamic_model.set_action(action, self.step_num)
            else:
                self.dynamic_model.set_action(action)

            position_ue4 = self.dynamic_model.get_position()
            self.trajectory_list.append(position_ue4)

            # 2) Yeni gözlem al
            try:
                obs = self.get_obs()
            except CameraError as e:
                obs_safe = getattr(self, "_last_obs", None)
                if obs_safe is None:
                     obs_safe = np.zeros(self.observation_space.shape, dtype=np.float32) 
                    
                info = {
                    "TimeLimit.truncated": True,
                    "camera_error": True,
                    "msg": str(e),
                }
                obs = obs_safe
                total_reward = -1.0
                done = True
                break
            else:
                self._last_obs = obs

            # 3) done & info hesapla
            done = self.is_done()
            info = {
                'is_success': self.is_in_desired_pose(),
                'is_crash': self.is_crashed(),
                'is_not_in_workspace': self.is_not_inside_workspace(),
                'step_num': self.step_num  # AJAN adımı
            }

            # 4) Ödül hesapla
            if self.dynamic_name == 'SimpleFixedwing':
                reward = self.compute_reward_final_fixedwing(done, action)
                terms, diag = {}, {}
            elif self.reward_type == 'reward_with_action':
                reward = self.compute_reward_with_action(done, action)
                terms, diag = {}, {}
            elif self.reward_type == 'reward_new':
                reward = self.compute_reward_multirotor_new(done, action)
                terms, diag = {}, {}
            elif self.reward_type == 'reward_lqr':
                reward = self.compute_reward_lqr(done, action)
                terms, diag = {}, {}
            elif self.reward_type == 'reward_final':
                reward, terms, diag = self.compute_reward_final(done, action, return_terms=True)
                info["reward_terms"] = terms
                info["diag"] = diag
                self._rb_update_step(terms, diag)
            else:
                reward = self.compute_reward(done, action)
                terms, diag = {}, {}

            # 5) Bölüm toplamlarını güncelle
            total_reward += reward
            self.cumulated_episode_reward += reward

            # 6) AirSim log + PyQt sinyalleri
            self.print_train_info_airsim(action, obs, reward, info)

            if self.cfg.get('options', 'dynamic_name') == 'SimpleFixedwing':
                self.set_pyqt_signal_fixedwing(action, reward, done)
            else:
                self.set_pyqt_signal_multirotor(action, reward)

            # 7) (İsteğe bağlı) klavye debug – bunu istersen bırak
            if self.keyboard_debug:
                action_copy = np.copy(action)
                action_copy[-1] = math.degrees(action_copy[-1])
                state_copy = np.copy(self.dynamic_model.state_raw)

                np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
                print('=============================================================================')
                print('episode', self.episode_num, 'step', self.step_num, 'total step', self.total_step)
                print('action', action_copy)
                print('state', state_copy)
                print('state_norm', self.dynamic_model.state_norm)
                print('reward {:.3f} {:.3f}'.format(reward, self.cumulated_episode_reward))
                print('done', done)
                keyboard.wait('a')

            # 8) Q-map görselleştirme (varsa)
            if self.generate_q_map and (self.cfg.get('options', 'algo') in ['TD3', 'SAC']):
                if self.model is not None:
                    with th.no_grad():
                        obs_copy = obs.copy()
                        if self.perception_type != 'vector':
                            obs_copy = obs_copy.swapaxes(0, 1)
                            obs_copy = obs_copy.swapaxes(0, 2)
                        q_value_current = self.model.critic(
                            th.from_numpy(obs_copy[None]).float().cuda(),
                            th.from_numpy(action[None]).float().cuda()
                        )
                        q_1 = q_value_current[0].cpu().numpy()[0]
                        q_2 = q_value_current[1].cpu().numpy()[0]
                        q_value = min(q_1, q_2)[0]
                        self.visual_log_q_value(q_value, action, reward)

            # 9) Fizik adımı sayacı
            self.total_step += 1

            # 10) Bölüm bitti mi?
            if done:
                term = {
                    "reach": int(self.is_in_desired_pose()),
                    "crash": int(self.is_crashed()),
                    "outside": int(self.is_not_inside_workspace()),
                }
                term["timeout"] = int(1 - max(term.values()))
                info["terminal"] = term

                if term["timeout"] == 1:
                    info["TimeLimit.truncated"] = True

                progress = None
                if self._d0 is not None and self._d0 > 1e-6 and self._dT is not None:
                    progress = (self._d0 - self._dT) / self._d0

                ep_sums = {f"R_ep/{k}": float(v) for k, v in self._rb_sums.items()}
                ep_sums.update({
                    "ep/len": self._rb_steps,  # fizik adımı sayısı
                    "ep/progress_frac": float(progress) if progress is not None else float("nan"),
                    "ep/min_obs_dist": float(self._min_obs),
                    "ep/done_reason_str": max(term, key=term.get),
                })
                info["ep_sums"] = ep_sums

                self._print_ep_summary(info)
                self._rb_reset()
                break  # frameskip döngüsünden çık

        # --- AJAN adımı sonu ---
        self.step_num += 1
        self.last_action = action

        return obs, total_reward, done, info

# ! -------------------------get obs------------------------------------------
    def get_obs(self):
        if self.perception_type == 'vector':
            obs = self.get_obs_vector()
        elif self.perception_type == 'lgmd':
            obs = self.get_obs_lgmd()
        else:
            obs = self.get_obs_image()
            
            if obs.dtype == np.float32 or obs.dtype == np.float64:
                if obs.max() <= 1.0: # Zaten 0-255 arasındaysa çarpmaması için kontrol
                    obs = obs * 255.0
            obs = obs.astype(np.uint8)
            
            # 2. Boyutu (70, 200, 2) -> (2, 70, 200) yap (H, W, C -> C, H, W)
            # Eğer kanal sayısı (2) zaten en sondayysa transpoze et
            if len(obs.shape) == 3 and obs.shape[-1] == 2:
                obs = np.transpose(obs, (2, 0, 1))

        return obs

 
    def get_obs_image(self):
        # Normal mode: get depth image then transfer to matrix with state
        
        # 1. Adım: Artık YÜKSEK HASSASİYETLİ (float32) ham metre verisi alıyoruz
        #          (Bir önceki adımda 'get_depth_image' fonksiyonunu düzelttik)
        image_meters = self.get_depth_image() 
        
        image_resize = cv2.resize(image_meters, (self.screen_width,
                                                self.screen_height))
        
        # min() alırken ham metre verisini kullanmak daha doğrudur
        self.min_distance_to_obstacles = image_resize.min() 

        # 2. DEĞİŞİKLİK: Veriyi [0.0, 1.0] float aralığına normalize et.
        #    Eski kod: ... / self.max_depth_meters * 255
        #    ' * 255 ' YANLIŞTI ve SİLİNDİ. 
        #    Amacımız [0, 255] değil, [0.0, 1.0] elde etmek.
        image_normalized = np.clip(
            image_resize, 0, self.max_depth_meters) / self.max_depth_meters

        # 3. DEĞİŞİKLİK: Veriyi [0.0, 1.0] aralığında ters çevir (Invert).
        #    Eski kod: image_scaled = 255 - image_scaled
        #    Bu, float veride '1.0 - image_normalized' işleminin karşılığıdır.
        #    Böylece (yakın = 1.0) ve (uzak = 0.0) olur.
        image_processed = 1.0 - image_normalized

        # 4. DEĞİŞİKLİK (EN KRİTİK HATA):
        #    image_uint8 = image_scaled.astype(np.uint8) SATIRI TAMAMEN SİLİNDİ.
        #    Veri kaybının (hassasiyet kaybı) ve matematiksel hatanın 
        #    kaynağı bu satırdı. Veri float32 olarak kalmalıdır.

        # 2. get current state (relative_pose, velocity)
        # 5. DEĞİŞİKLİK: state_feature_array'in de float olmasını sağlıyoruz.
        state_feature_array = np.zeros((self.screen_height, self.screen_width), 
                                    dtype=np.float32) # Dtype eklendi
        
        state_feature = self.dynamic_model._get_state_feature()
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        # 3. generate image with state
        # 6. DEĞİŞİKLİK: Artık 'image_uint8' yerine 'image_processed' (float)
        #    kullanarak birleştiriyoruz.
        image_with_state = np.array([image_processed.astype(np.float32), 
                                    state_feature_array.astype(np.float32)])
        
        # Bu eksen değiştirme işlemleri (swapaxes) DRL için yaygındır,
        # (H, W, C) formatına getirir. Aynen kalıyor.
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)

        # Bu 'return' artık (H, W, 2) şeklinde ve np.float32 tipinde
        # bir 'observation' (gözlem) döndürür.
        return image_with_state

    def get_depth_gray_image(self):
        # get depth and rgb image
        # scene vision image in png format
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        ])

        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True),
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
            ])

        # get depth image
        depth_img = airsim.list_to_2d_float_array(
            responses[0].image_data_float,
            responses[0].width, responses[0].height)
        depth_meter = depth_img * 100

        # get gary image
        img_1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
        # reshape array to 4 channel image array H X W X 3
        img_rgb = img_1d.reshape(responses[1].height, responses[1].width, 3)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('test', img_rgb)
        # cv2.waitKey(1)

        return depth_meter, img_gray

  

    def get_depth_image(self):
       
        req = [airsim.ImageRequest(
                "0", 
                airsim.ImageType.DepthPerspective, 
                pixels_as_float=True, 
                compress=False
            )]
        last_err = None
        for attempt in range(3):
            try:
                responses = self.client.simGetImages(req)
                r = responses[0]

                if r.width == 0 or r.height == 0:
                    raise RuntimeError("empty image: zero width/height")
                if len(r.image_data_float) == 0:
                    raise RuntimeError("empty image_data_float")

                # ========================= HATANIN DÜZELTİLDİĞİ YER =========================
                # 'image_data_float' bir 'list'tir, 'bytes' değil.
                # Bu yüzden 'np.frombuffer' hata verir.
                # Sizin orijinal kodunuzdaki 'list_to_2d_float_array' 
                # bu 'list'i işlemek için DOĞRU fonksiyondur.
                
                # HATALI KOD (Benim hatam):
                # depth_img = np.frombuffer(r.image_data_float, dtype=np.float32)
                # depth_img = depth_img.reshape(r.height, r.width)
                
                # DOĞRU KOD:
                depth_img = airsim.list_to_2d_float_array(
                    r.image_data_float, r.width, r.height
                )
                # ==============================================================================

                # 100 ile çarpmayı kaldırma mantığımız hala geçerli, 
                # çünkü 'DepthPerspective' zaten metre verisi verir.
                return depth_img

            except (msgpackrpc.error.RPCError, RuntimeError) as e:
                last_err = e
                if attempt == 0:
                    try:
                        self.client.confirmConnection()
                        if hasattr(self, "vehicle_name"):
                            try:
                                self.client.enableApiControl(True, self.vehicle_name)
                                self.client.armDisarm(True, self.vehicle_name)
                            except Exception:
                                pass
                    except Exception:
                        pass
                time.sleep(0.05)
        raise CameraError(f"simGetImages failed after retries: {last_err}")

        

    def get_obs_vector(self):

        image = self.get_depth_image()  # 0-6550400.0 float 32
        self.min_distance_to_obstacles = image.min()

        image_scaled = np.clip(image, 0, self.max_depth_meters) / self.max_depth_meters * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        image_obs = image_uint8
        split_row = 1
        split_col = 5

        v_split_list = np.vsplit(image_obs, split_row)

        split_final = []
        for i in range(split_row):
            h_split_list = np.hsplit(v_split_list[i], split_col)
            for j in range(split_col):
                split_final.append(h_split_list[j].max())

        img_feature = np.array(split_final) / 255.0

        state_feature = self.dynamic_model._get_state_feature() / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)

        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all

    def get_obs_lgmd(self):
        # get depth and gray image
        depth_meter, img_gray = self.get_depth_gray_image()
        self.min_distance_to_obstacles = depth_meter.min()

        self.lgmd.update(img_gray)

        split_col_num = 5
        s_layer = self.lgmd.s_layer  # (192, 320)
        s_layer_split = np.hsplit(s_layer, split_col_num)  # (192, 109)

        lgmd_out_list = []
        activate_coeff = 0.5
        for i in range(split_col_num):
            s_layer_activated_sum = abs(np.sum(s_layer_split[i]))
            Kf = -(s_layer_activated_sum * activate_coeff) / (192*64)  # 0 - 1
            a = np.exp(Kf)
            lgmd_out_norm = (1 / (1 + a) - 0.5) * 2
            lgmd_out_list.append(lgmd_out_norm)

        # show iamges
        heatmapshow = None
        heatmapshow = cv2.normalize(s_layer, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        cv2.imshow('gray image', img_gray)
        cv2.imshow('depth image', np.clip(depth_meter, 0, 255)/255)
        cv2.imshow('s-layer', heatmapshow)
        cv2.waitKey(1)

        # update LGMD
        split_final = np.array(lgmd_out_list)
        
        filter_coeff = 0.8
        split_final_filter = filter_coeff * split_final + (1-filter_coeff) * self.split_out_last
        self.split_out_last = split_final_filter

        img_feature = np.array(split_final_filter)

        state_feature = self.dynamic_model._get_state_feature() / 255

        feature_all = np.concatenate((img_feature, state_feature), axis=0)

        self.feature_all = feature_all

        feature_all = np.reshape(feature_all, (1, len(feature_all)))

        return feature_all
# ! ---------------------calculate rewards-------------------------------------

    def compute_reward(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                500  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            yaw_speed_cost = 0.1 * \
                abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * \
                    ((abs(action[1]) / self.dynamic_model.v_z_max)**2)
                z_err_cost = 0.05 * \
                    ((abs(
                        self.dynamic_model.state_raw[1]) / self.dynamic_model.max_vertical_difference)**2)
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.1 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    '''
    def compute_reward_final(self, done, action, return_terms=False):

        reward = 0
        reward_reach = 30
        reward_crash = -25
        reward_outside = -10
        
        if self.env_name == 'NH_center':
            distance_reward_coef = 500
        else:
            distance_reward_coef = 50

        if not done:
            # 1 - goal reward
            distance_now = self.get_distance_to_goal_3d()
            d_prev = self.previous_distance_from_des_point
            d_delta = d_prev - distance_now
            reward_distance = distance_reward_coef * (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance   # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            # 2 - Position punishment
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            z = current_pose[2]
            x_g = goal_pose[0]
            y_g = goal_pose[1]
            z_g = goal_pose[2]

            punishment_xy = np.clip(self.getDis(
                x, y, 0, 0, x_g, y_g) / 10, 0, 1)
            punishment_z = 0.5 * np.clip((z - z_g)/5, 0, 1)

            punishment_pose = punishment_xy + punishment_z

            if self.min_distance_to_obstacles < 10:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 5, 0, 1)
            else:
                punishment_obs = 0

            punishment_action = 0

            # add yaw_rate cost
            yaw_speed_cost = abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = ((abs(action[1]) / self.dynamic_model.v_z_max)**2)
                z_err_cost = (
                    (abs(self.dynamic_model.state_raw[1]) / self.dynamic_model.max_vertical_difference)**2)
                punishment_action += (v_z_cost + z_err_cost)

            punishment_action += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            reward = reward_distance - 0.1 * punishment_pose - 0.25 * \
                punishment_obs - 0.1 * punishment_action - 0.15 * yaw_error_cost
            # terms
            terms = {
                "r_total": reward,
                "r_dist": reward_distance,
                "c_pose": 0.1 * punishment_pose,
                "c_obs": 0.25 * punishment_obs,
                "c_act": 0.1 * punishment_action,
                "c_yaw": 0.15 * yaw_error_cost,
                
            }

            # diag
            diag = {
                "d_prev": d_prev,
                "d_now": distance_now,
                "d_delta": d_delta,
                "yaw_error_deg": yaw_error,
                "min_obs_dist": self.min_distance_to_obstacles,
                "crash_distance": self.crash_distance,
                "action_vxy": action[0],
                "action_yawrate": action[-1],
            }

        else:
            # done durumunda sadece sabit ödül
            if self.is_in_desired_pose():
                reward = reward_reach
            elif self.is_crashed():
                reward = reward_crash
            elif self.is_not_inside_workspace():
                reward = reward_outside

            terms = {"r_total": reward, "r_dist": 0.0, "c_pose": 0.0, "c_obs": 0.0, "c_act": 0.0, "c_yaw": 0.0, "c_step": 0.0}
            diag = {}

        if return_terms:
            return reward, terms, diag
        return reward
        '''
    def compute_reward_final(self, done, action, return_terms=False):

        reward = 0.0  # Toplam ödül

        # --- Terminal ödüller ---
        reward_reach = 30
        reward_crash = -25
        reward_outside = -10

        # Ortam katsayısı
        if self.env_name == 'NH_center':
            distance_reward_coef = 500
        else:
            distance_reward_coef = 40

        # ======================================================================
        # 1. SHAPING ÖDÜLLERİ (HER ADIMDA)
        # ======================================================================

        # --- Hedefe yaklaşma (PBRS) ---
        distance_now = self.get_distance_to_goal_3d()
        d_prev = self.previous_distance_from_des_point
        d_delta = d_prev - distance_now

        # goal_distance ile normalize edilmiş shaping
        reward_distance = distance_reward_coef * d_delta / self.dynamic_model.goal_distance
        self.previous_distance_from_des_point = distance_now

        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
       

        reward += reward_distance

        # --- Kısıtlar / cezalar ---

        # KISIT 1: Engeller (kuadratik ceza, SAFE_DISTANCE içi)
        punishment_obs = 0.0
        dist_to_obs = self.min_distance_to_obstacles
        if dist_to_obs < self.SAFE_DISTANCE:
            distance_inside_bubble = self.SAFE_DISTANCE - dist_to_obs
            punishment_obs = self.k_obs_penalty * (distance_inside_bubble ** 2)

        v_xy_real = self.dynamic_model.get_velocity()[0]

        # KISIT 2: Engellere yakınken yüksek hızda gitme (gerçek hız üzerinden)
        punishment_speed_near_obs = 0.0
        if dist_to_obs < self.SAFE_DISTANCE:
            if v_xy_real > self.max_safe_speed_near_obs:
            # Engellere yakınken "güvenli hız" eşiği
                overspeed = v_xy_real - self.max_safe_speed_near_obs
                punishment_speed_near_obs = self.k_speed_near_obs * (overspeed ** 2)

        # KISIT 3: Sarsıntı (Jerk) – aksiyon farkları
        action_jerk_fwd = action[0] - self.last_action[0]
        action_jerk_yaw = action[1] - self.last_action[1]  # artık yaw açısı komutunun jerk'i

        punishment_jerk_fwd = self.k_jerk_fwd * (action_jerk_fwd ** 2)
        punishment_jerk_yaw = self.k_jerk_yaw * (action_jerk_yaw ** 2)

        # --- Tüm shaping cezalarını ekle ---
        reward -= punishment_obs
        reward -= punishment_speed_near_obs
        reward -= punishment_jerk_fwd
        reward -= punishment_jerk_yaw

        # ======================================================================
        # 2. TERMINAL ÖDÜLLER (SADECE done=True İKEN)
        # ======================================================================

        if done:
            if self.is_in_desired_pose():
                reward += reward_reach
            elif self.is_crashed():
                reward += reward_crash
            elif self.is_not_inside_workspace():
                reward += reward_outside
            # TimeLimit sonlanmalarında ekstra ödül yok; sadece shaping ile yetinir.

        # ======================================================================
        # 3. LOG SÖZLÜKLERİ
        # ======================================================================

        yaw_error = self.dynamic_model.state_raw[2]  # (örneğin hedefe göre yaw hata açısı)

        terms = {
            "r_total": reward,
            "r_dist": reward_distance,

            # Uyumluluk için 0'a sabit eski alanlar:
            "c_pose": 0.0,
            "c_act_fwd": 0.0,  # eski forward action cost artık yok
            "c_act_yaw": 0.0,  # eski yaw action cost artık yok
            "c_yaw": 0.0,

            # Aktif cezalar:
            "c_obs": punishment_obs,
            "c_spd_obs": punishment_speed_near_obs,
            "c_jerk_fwd": punishment_jerk_fwd,
            "c_jerk_yaw": punishment_jerk_yaw,
        }

        diag = {
        "d_prev": d_prev,
        "d_now": distance_now,
        "d_delta": d_delta,
        "yaw_error_deg": yaw_error,
        "min_obs_dist": dist_to_obs,
        "crash_distance": self.crash_distance,
        "action_vxy": action[0],
        "action_yaw_cmd": action[1],
        "v_xy_real": v_xy_real,
        }

        if return_terms:
            return reward, terms, diag
        return reward



    def compute_reward_final_fixedwing(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = -10

        if not done:
            # 1 - goal reward
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = 300 * (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance   # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            # 2 - Position punishment
            current_pose = self.dynamic_model.get_position()
            goal_pose = self.dynamic_model.goal_position
            x = current_pose[0]
            y = current_pose[1]
            x_g = goal_pose[0]
            y_g = goal_pose[1]

            punishment_xy = np.clip(self.getDis(
                x, y, 0, 0, x_g, y_g) / 50, 0, 1)
            # punishment_z = 0.5 * np.clip((z - z_g)/5, 0, 1)

            punishment_pose = punishment_xy

            if self.min_distance_to_obstacles < 20:
                punishment_obs = 1 - np.clip((self.min_distance_to_obstacles - self.crash_distance) / 15, 0, 1)
            else:
                punishment_obs = 0

            # action cost
            punishment_action = abs(action[0]) / self.dynamic_model.roll_rate_max

            yaw_error = self.dynamic_model.state_raw[1]
            yaw_error_cost = abs(yaw_error / 90)

            reward = reward_distance - 0.1 * punishment_pose - 0.5 * \
                punishment_obs - 0.1 * punishment_action - 0.1 * yaw_error_cost
            # reward = reward

            # print("r_dist: {:.2f} p_pose: {:.2f} p_obs: {:.2f} p_action: {:.2f}, p_yaw_e: {:.2f}".format(reward_distance, punishment_pose, punishment_obs, punishment_action, yaw_error_cost))
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_test(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -100
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                100  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            yaw_speed_cost = 0.1 * \
                abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * abs(action[1]) / self.dynamic_model.v_z_max
                z_err_cost = 0.05 * \
                    abs(self.dynamic_model.state_raw[1]) / \
                    self.dynamic_model.max_vertical_difference
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.1 * abs(yaw_error / 180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_fixedwing(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -50
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                300  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            # 只有action cost和obs cost
            # 由于没有速度控制，所以前面那个也取消了
            # action_cost = 0
            # obs_cost = 0

            # relative_yaw_cost = abs(
            #     (self.dynamic_model.state_norm[0]/255-0.5) * 2)
            # action_cost = abs(action[0]) / self.dynamic_model.roll_rate_max

            # obs_punish_distance = 15
            # if self.min_distance_to_obstacles < obs_punish_distance:
            #     obs_cost = 1 - (self.min_distance_to_obstacles -
            #                     self.crash_distance) / (obs_punish_distance -
            #                                             self.crash_distance)
            #     obs_cost = 0.5 * obs_cost ** 2
            # reward = reward_distance - (2 * relative_yaw_cost + 0.5 * action_cost + obs_cost)

            action_cost = abs(action[0]) / self.dynamic_model.roll_rate_max

            yaw_error_deg = self.dynamic_model.state_raw[1]
            yaw_error_cost = 0.1 * abs(yaw_error_deg / 180)

            reward = reward_distance - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                yaw_error_deg = self.dynamic_model.state_raw[1]
                reward = reward_reach * (1 - abs(yaw_error_deg / 180))
                # reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_multirotor_new(self, done, action):
        reward = 0
        reward_reach = 100
        reward_crash = -100
        reward_outside = 0

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point -
                               distance_now) / self.dynamic_model.goal_distance * 5
            self.previous_distance_from_des_point = distance_now

            state_cost = 0
            action_cost = 0
            obs_cost = 0

            yaw_error_deg = self.dynamic_model.state_raw[1]

            relative_yaw_cost = abs(yaw_error_deg/180)
            action_cost = abs(action[1]) / self.dynamic_model.yaw_rate_max_rad

            obs_punish_dist = 5
            if self.min_distance_to_obstacles < obs_punish_dist:
                obs_cost = 1 - (self.min_distance_to_obstacles -
                                self.crash_distance) / (obs_punish_dist - self.crash_distance)
                obs_cost = 0.5 * obs_cost ** 2
            reward = - (2 * relative_yaw_cost + 0.5 * action_cost)
        else:
            if self.is_in_desired_pose():
                # 到达之后根据yaw偏差对reward进行scale
                reward = reward_reach * \
                    (1 - abs(self.dynamic_model.state_norm[1]))
                # reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_with_action(self, done, action):
        reward = 0
        reward_reach = 50
        reward_crash = -50
        reward_outside = -10

        step_cost = 0.01  # 10 for max 1000 steps

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now) / \
                self.dynamic_model.goal_distance * \
                10  # normalized to 100 according to goal_distance
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add action cost
            # speed 0-8  cruise speed is 4, punish for too fast and too slow
            v_xy_cost = 0.02 * abs(action[0]-5) / 4
            yaw_rate_cost = 0.02 * \
                abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
            if self.dynamic_model.navigation_3d:
                v_z_cost = 0.02 * abs(action[1]) / self.dynamic_model.v_z_max
                action_cost += v_z_cost
            action_cost += (v_xy_cost + yaw_rate_cost)

            yaw_error = self.dynamic_model.state_raw[2]
            yaw_error_cost = 0.05 * abs(yaw_error/180)

            reward = reward_distance - reward_obs - action_cost - yaw_error_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def compute_reward_lqr(self, done, action):
        # 模仿matlab提供的mix reward的思想设计
        reward = 0
        reward_reach = 10
        reward_crash = -20
        reward_outside = 0

        if not done:
            action_cost = 0
            # add yaw_rate cost
            yaw_speed_cost = 0.2 * \
                ((action[-1] / self.dynamic_model.yaw_rate_max_rad) ** 2)

            if self.dynamic_model.navigation_3d:
                # add action and z error cost
                v_z_cost = 0.1 * ((action[1] / self.dynamic_model.v_z_max)**2)
                z_err_cost = 0.1 * \
                    ((self.dynamic_model.state_raw[1] /
                      self.dynamic_model.max_vertical_difference)**2)
                action_cost += (v_z_cost + z_err_cost)

            action_cost += yaw_speed_cost

            yaw_error_clip = min(
                max(-60, self.dynamic_model.state_raw[2]), 60) / 60
            yaw_error_cost = 1.0 * (yaw_error_clip**2)

            reward = - (action_cost + yaw_error_cost)

            # print('r: {:.2f} y_r: {:.2f} y_e: {:.2f} z_r: {:.2f} z_e: {:.2f}'.format(reward, yaw_speed_cost, yaw_error_cost, v_z_cost, z_err_cost))
        else:
            if self.is_in_desired_pose():
                yaw_error_clip = min(
                    max(-30, self.dynamic_model.state_raw[2]), 30) / 30
                reward = reward_reach * (1 - yaw_error_clip**2)
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

# ! ------------------ is done-----------------------------------------------

    def is_done(self):
        episode_done = False

        is_not_inside_workspace_now = self.is_not_inside_workspace()
        has_reached_des_pose = self.is_in_desired_pose()
        too_close_to_obstable = self.is_crashed()

        # We see if we are outside the Learning Space
        episode_done = is_not_inside_workspace_now or\
            has_reached_des_pose or\
            too_close_to_obstable or\
            self.step_num >= self.max_episode_steps

        return episode_done

    def is_not_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_not_inside = False
        current_position = self.dynamic_model.get_position()

        if current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or \
            current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or \
                current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]:
            is_not_inside = True

        return is_not_inside

    def is_in_desired_pose(self):
        in_desired_pose = False
        if self.get_distance_to_goal_3d() < self.accept_radius:
            in_desired_pose = True

        return in_desired_pose

    def is_crashed(self):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided or self.min_distance_to_obstacles < self.crash_distance:
            is_crashed = True

        return is_crashed
    

# ! ----------- useful functions-------------------------------------------
    def get_distance_to_goal_3d(self):
        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))

    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        '''
        Get distance between Point and Line
        Used to calculate position punishment
        '''
        a = lineY2-lineY1
        b = lineX1-lineX2
        c = lineX2*lineY1-lineX1*lineY2
        dis = (math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b, 0.5))

        return dis
# ! -----------used for plot or show states------------------------------------------------------------------

    def print_train_info_airsim(self, action, obs, reward, info):
        # if self.perception_type == 'split' or self.perception_type == 'lgmd':
        #     feature_all = self.feature_all
        # elif self.perception_type == 'vector':
        #     feature_all = self.feature_all
        # else:
        #     if self.cfg.get('options', 'algo') == 'TD3' or self.cfg.get('options', 'algo') == 'SAC':
        #         feature_all = self.model.actor.features_extractor.feature_all
        #     elif self.cfg.get('options', 'algo') == 'PPO':
        #         feature_all = self.model.policy.features_extractor.feature_all

        # self.client.simPrintLogMessage('feature_all: ', str(feature_all))

        msg_train_info = "EP: {} Step: {} Total_step: {}".format(
            self.episode_num, self.step_num, self.total_step)

        self.client.simPrintLogMessage('Train: ', msg_train_info)
        self.client.simPrintLogMessage('Action: ', str(action))
        self.client.simPrintLogMessage('Reward: ', "{:4.4f} total: {:4.4f}".format(
            reward, self.cumulated_episode_reward))

        # --- Reward terms: tek, kısa satır (GÜNCELLENDİ, "step" kaldırıldı) ---
        terms = info.get("reward_terms", {})
        terms_line = (
            f"dist={terms.get('r_dist',0.0):.2f}  "
            f"obs={terms.get('c_obs',0.0):.2f}  "
            f"fwd={terms.get('c_act_fwd',0.0):.2f}  " 
            f"yaw={terms.get('c_act_yaw',0.0):.2f}  " 
            f"spd_obs={terms.get('c_spd_obs',0.0):.2f}  "
            # --- YENİ (GÖREV A) ---
            f"j_fwd={terms.get('c_jerk_fwd',0.0):.2f}  "
            f"j_yaw={terms.get('c_jerk_yaw',0.0):.2f}  "
            # --- YENİ KOD SONU ---
            f"total={terms.get('r_total',0.0):.2f}"
        )
        self.client.simPrintLogMessage('Terms: ', terms_line)

        # ... (Diag1 ve Diag2 logları aynı kalır) ...
        diag = info.get("diag", {})
        diag_line1 = (
            f"d_prev={diag.get('d_prev', float('nan')):.2f}  "
            f"d_now={diag.get('d_now', float('nan')):.2f}  "
            f"d_delta={diag.get('d_delta', float('nan')):.2f}  "
            f"yaw_deg={diag.get('yaw_error_deg', float('nan')):.2f}"
        )
        diag_line2 = (
            f"min_obs={diag.get('min_obs_dist', float('nan')):.2f}  "
            f"crash_d={diag.get('crash_distance', float('nan')):.2f}  "
            f"vxy={diag.get('action_vxy', float('nan')):.2f}  "
            f"yaw_cmd={diag.get('action_yaw_cmd', float('nan')):.2f}"
        )
        self.client.simPrintLogMessage('Diag1: ', diag_line1)
        self.client.simPrintLogMessage('Diag2: ', diag_line2)
        
        self.client.simPrintLogMessage('Min_depth: ', str(self.min_distance_to_obstacles))

    def set_pyqt_signal_fixedwing(self, action, reward, done):
        """
        emit signals for pyqt plot
        """
        step = int(self.total_step)
        # action: v_xy, v_z, roll

        action_plot = np.array([10, 0, math.degrees(action[0])])

        state = self.dynamic_model.state_raw  # distance, relative yaw, roll

        # state out 6: d_xy, d_z, yaw_error, v_xy, v_z, roll
        # state in  3: d_xy, yaw_error, roll
        state_output = np.array([state[0], 0, state[1], 10, 0, state[2]])

        self.action_signal.emit(step, action_plot)
        self.state_signal.emit(step, state_output)

        # other values
        self.attitude_signal.emit(step, np.asarray(self.dynamic_model.get_attitude(
        )), np.asarray(self.dynamic_model.get_attitude_cmd()))
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position), np.asarray(
            self.dynamic_model.start_position), np.asarray(self.dynamic_model.get_position()), np.asarray(self.trajectory_list))

        # lgmd_signal = pyqtSignal(float, float, np.ndarray)  min_dist, lgmd_out, lgmd_split
        self.lgmd_signal.emit(self.min_distance_to_obstacles, 0,  self.feature_all[:-1])

    def set_pyqt_signal_multirotor(self, action, reward):
        step = int(self.total_step)

        # transfer 2D state and action to 3D
        state = self.dynamic_model.state_raw
        if self.dynamic_model.navigation_3d:
            action_output = action
            state_output = state
        else:
            cmd_v_xy = self.dynamic_model.v_xy_sp
            cmd_yaw_rate = math.radians(self.dynamic_model.yaw_rate_cmd_deg)
            action_output = np.array([cmd_v_xy, 0, cmd_yaw_rate])
            state_output = np.array([state[0], 0, state[2], state[3], 0, state[5]])

        self.action_signal.emit(step, action_output)
        self.state_signal.emit(step, state_output)

        # other values
        self.attitude_signal.emit(step, np.asarray(self.dynamic_model.get_attitude(
        )), np.asarray(self.dynamic_model.get_attitude_cmd()))
        self.reward_signal.emit(step, reward, self.cumulated_episode_reward)
        self.pose_signal.emit(np.asarray(self.dynamic_model.goal_position), np.asarray(
            self.dynamic_model.start_position), np.asarray(self.dynamic_model.get_position()), np.asarray(self.trajectory_list))

    def visual_log_q_value(self, q_value, action, reward):
        '''
        Create grid map (map_size = work_space)
        Log Q value and the best action in grid map
        At any grid position, record:
        1. Q value
        2. action 0
        3. action 1
        4. steps
        5. reward
        Save image every 10k steps
        Used only for 2D explanation
        '''

        # create init array if not exist
        map_size_x = self.work_space_x[1] - self.work_space_x[0]
        map_size_y = self.work_space_y[1] - self.work_space_y[0]
        if not hasattr(self, 'q_value_map'):
            self.q_value_map = np.full((9, map_size_x+1, map_size_y+1), np.nan)

        # record info
        position = self.dynamic_model.get_position()
        pose_x = position[0]
        pose_y = position[1]

        index_x = int(np.round(pose_x) + self.work_space_x[1])
        index_y = int(np.round(pose_y) + self.work_space_y[1])

        # check if index valid
        if index_x in range(0, map_size_x) and index_y in range(0, map_size_y):
            self.q_value_map[0, index_x, index_y] = q_value
            self.q_value_map[1, index_x, index_y] = action[0]
            self.q_value_map[2, index_x, index_y] = action[-1]
            self.q_value_map[3, index_x, index_y] = self.total_step
            self.q_value_map[4, index_x, index_y] = reward
            self.q_value_map[5, index_x, index_y] = q_value
            self.q_value_map[6, index_x, index_y] = action[0]
            self.q_value_map[7, index_x, index_y] = action[-1]
            self.q_value_map[8, index_x, index_y] = reward
        else:
            print(
                'Error: X:{} and Y:{} is outside of range 0~mapsize (visual_log_q_value)')

        # save array every record_step steps
        record_step = self.cfg.getint('options', 'q_map_save_steps')
        if (self.total_step+1) % record_step == 0:
            if self.data_path is not None:
                np.save(
                    self.data_path + '/q_value_map_{}'.format(self.total_step+1), self.q_value_map)
                # refresh 5 6 7 8 to record period data
                self.q_value_map[5, :, :] = np.nan
                self.q_value_map[6, :, :] = np.nan
                self.q_value_map[7, :, :] = np.nan
                self.q_value_map[8, :, :] = np.nan