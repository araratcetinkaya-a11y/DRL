# Gerekli Kütüphaneler
import airsim
import time
import numpy as np
import cv2
import msgpackrpc
import math
import gym
from gym import spaces

# PyTorch (Gerçek Eğitim Yükü için)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Stable Baselines 3 (Sadece kritik bileşenler)
try:
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.policies import BaseFeaturesExtractor
    from stable_baselines3.common.torch_layers import create_mlp
except ImportError:
    print("\n!!! HATA: 'stable-baselines3' kütüphanesi bulunamadı.")
    print("Test script'inin çalışması için lütfen 'pip install stable-baselines3' komutunu çalıştırın.")
    print("Bu, 'ReplayBuffer' ve 'BaseFeaturesExtractor' sınıfları için gereklidir.")
    exit()

# --- 1. SİZİN CNN AĞINIZ ---
class CNN_GAP_new(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 132, state_feature_dim=4):
        super(CNN_GAP_new, self).__init__(observation_space, features_dim)
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim # 128
        self.feature_all = None
        # Matematik (70, 200) girişi için doğrulandı:
        self.conv1 = nn.Conv2d(1, 8, 5, 2)  # (70,200) -> (33,98)
        self.conv2 = nn.Conv2d(8, 8, 5, 2)  # (33,98) -> (15,47)
        self.conv3 = nn.Conv2d(8, 8, 3, 2)  # (15,47) -> (7,23)
        self.pool = nn.MaxPool2d(2, 3)     # (7,23) -> (2,8)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Gözlem (B, 2, 70, 200) şeklinde gelir
        depth_img = observations[:, 0:1, :, :]
        depth_img_norm = (depth_img - 0.5) * 2
        # PyTorch CNN katmanları
        self.layer_1_out = F.relu(self.conv1(depth_img_norm))
        self.layer_2_out = F.relu(self.conv2(self.layer_1_out))
        self.layer_3_out = F.relu(self.conv3(self.layer_2_out))
        self.layer_small = self.pool(self.layer_3_out)
        self.flatten = torch.flatten(self.layer_small, start_dim=1)
        # Durum vektörünü al
        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        # CNN özelliklerini ve durumu birleştir
        x = torch.cat((self.flatten, state_feature), dim=1)
        self.feature_all = x
        return x

# --- 2. SAC AĞLARININ SİMÜLASYONU ---
class Actor(nn.Module):
    """SAC Actor Ağı"""
    def __init__(self, features_dim, action_dim, net_arch):
        super().__init__()
        self.policy_net = nn.Sequential(*create_mlp(features_dim, action_dim, net_arch, nn.Tanh))
    def forward(self, obs): return self.policy_net(obs)

class Critic(nn.Module):
    """SAC Critic Ağı"""
    def __init__(self, features_dim, action_dim, net_arch):
        super().__init__()
        self.q_net = nn.Sequential(*create_mlp(features_dim + action_dim, 1, net_arch, nn.Tanh))
    def forward(self, obs, actions):
        return self.q_net(torch.cat([obs, actions], dim=1))

# --- 3. TEST ORTAMI SINIFI (Tüm Bileşenler) ---
class FullSystemTest:
    """Tüm AirSim, Gözlem, Fizik ve Eğitim bileşenlerini yöneten ana sınıf"""
    def __init__(self, config):
        self.cfg = config
        print("AirSim'e bağlanılıyor...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        print("Bağlantı başarılı.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Kullanılan Cihaz: {self.device}")

        # Gözlem (T2)
        self.screen_height = config['screen_height']
        self.screen_width = config['screen_width']
        self.max_depth_meters = config['max_depth_meters']
        self.state_feature_dim = config['state_feature_dim']
        
        # Fizik (T1)
        self.dt = config['dt']
        self.start_position = [0, 0, 15] # 15m irtifa
        self.v_xy_delta_max = config['acc_xy_max'] * config['dt']
        self.yaw_delta_max_deg = config['yaw_rate_max_deg'] * config['dt'] # (Hata düzeltmesi: 10.0* kaldırıldı)
        self.v_xy_min = config['v_xy_min']
        self.v_xy_max = config['v_xy_max']
        self.yaw_rate_max_deg = config['yaw_rate_max_deg']
        self.prev_v_xy_sp = 0
        self.prev_yaw_rate_deg = 0

        # Eğitim (T3)
        self.features_dim = config['features_dim']
        self.action_dim = 2 
        self.net_arch = config['net_arch']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        
        self.obs_shape = (2, self.screen_height, self.screen_width)
        self.action_shape = (self.action_dim,)
        self.replay_buffer = ReplayBuffer(
            config['buffer_size'],
            spaces.Box(low=0, high=1, shape=self.obs_shape),
            spaces.Box(low=-1, high=1, shape=self.action_shape),
            self.device,
            handle_timeout_termination=False
        )
        
        self.feature_extractor = CNN_GAP_new(
            spaces.Box(low=0, high=1, shape=self.obs_shape),
            features_dim=self.features_dim,
            state_feature_dim=self.state_feature_dim
        ).to(self.device)
        self.actor = Actor(self.features_dim, self.action_dim, self.net_arch).to(self.device)
        self.critic = Critic(self.features_dim, self.action_dim, self.net_arch).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # Başlangıç pozisyonuna git ve duraklat
        pose = self.client.simGetVehiclePose()
        pose.position.z_val = -self.start_position[2] # Z= -15m
        self.client.simSetVehiclePose(pose, True)
        self.client.simPause(True)
        print(f"Drone {self.start_position[2]}m irtifada, simülasyon duraklatıldı.")

    def get_depth_image(self):
        """T2 (Gözlem) - Sadece Görüntü Alma"""
        req = [airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)]
        responses = self.client.simGetImages(req)
        r = responses[0]
        if r.width == 0: raise RuntimeError("Görüntü 0")
        return airsim.list_to_2d_float_array(r.image_data_float, r.width, r.height)

    def get_obs(self):
        """T2 (Gözlem) - Görüntü + İşleme"""
        image_meters = self.get_depth_image() 
        image_resize = cv2.resize(image_meters, (self.screen_width, self.screen_height))
        image_normalized = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters
        image_processed = 1.0 - image_normalized
        state_feature = np.zeros(self.state_feature_dim, dtype=np.float32)
        state_feature_array = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
        state_feature_array[0, 0:self.state_feature_dim] = state_feature
        image_with_state = np.array([image_processed, state_feature_array], dtype=np.float32)
        return image_with_state.swapaxes(0, 2).swapaxes(0, 1)

    def get_z_pos(self):
        """T4 (Güvenlik Kontrolü)"""
        return self.client.simGetVehiclePose().position.z_val

    def set_action_mevcut_mimari(self, action):
        """T1 (Fizik) - Sizin mevcut mimariniz"""
        delta_v_xy = action[0] * self.v_xy_delta_max 
        delta_yaw_rate_deg = action[1] * self.yaw_delta_max_deg
        self.v_xy_sp = np.clip(self.prev_v_xy_sp + delta_v_xy, self.v_xy_min, self.v_xy_max)
        self.yaw_rate_sp_deg = np.clip(self.prev_yaw_rate_deg + delta_yaw_rate_deg, -self.yaw_rate_max_deg, self.yaw_rate_max_deg)
        self.v_z_sp = 0.0
        self.yaw_rad = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)[2]
        vx_world_sp = self.v_xy_sp * math.cos(self.yaw_rad)
        vy_world_sp = self.v_xy_sp * math.sin(self.yaw_rad)
        self.client.moveByVelocityZAsync(
            vx_world_sp, vy_world_sp, -self.start_position[2], 1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=self.yaw_rate_sp_deg)
        )
        # !!! ANA DARBOĞAZ (T1) BURADA !!!
        self.client.simContinueForTime(self.dt) 
        self.prev_v_xy_sp = self.v_xy_sp
        self.prev_yaw_rate_deg = self.yaw_rate_sp_deg

    def set_action_onerilen_mimari(self, action):
        """T1 (Fizik) - Önerilen mimari"""
        delta_v_xy = action[0] * self.v_xy_delta_max 
        delta_yaw_rate_deg = action[1] * self.yaw_delta_max_deg
        self.v_xy_sp = np.clip(self.prev_v_xy_sp + delta_v_xy, self.v_xy_min, self.v_xy_max)
        self.yaw_rate_sp_deg = np.clip(self.prev_yaw_rate_deg + delta_yaw_rate_deg, -self.yaw_rate_max_deg, self.yaw_rate_max_deg)
        self.v_z_sp = 0.0
        self.yaw_rad = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation)[2]
        vx_world_sp = self.v_xy_sp * math.cos(self.yaw_rad)
        vy_world_sp = self.v_xy_sp * math.sin(self.yaw_rad)
        self.client.moveByVelocityZAsync(
            vx_world_sp, vy_world_sp, -self.start_position[2], 1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=self.yaw_rate_sp_deg)
        )
        # 'simContinueForTime' buradan çıkarıldı
        self.prev_v_xy_sp = self.v_xy_sp
        self.prev_yaw_rate_deg = self.yaw_rate_sp_deg

    def run_training_step(self):
        """T3 (Eğitim Yükü)"""
        # 1. Veri Hazırlama (CPU -> GPU)
        replay_data = self.replay_buffer.sample(self.batch_size)
        
        # 2. Asıl Eğitim (GPU İşi)
        with torch.no_grad():
            sahte_target_q = torch.randn(self.batch_size, 1).to(self.device)
        features = self.feature_extractor(replay_data.observations)
        current_q_values = self.critic(features, replay_data.actions)
        critic_loss = F.mse_loss(current_q_values, sahte_target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        features_pi = self.feature_extractor(replay_data.observations)
        actions_pi = self.actor(features_pi)
        q_values_pi = self.critic(features_pi, actions_pi)
        actor_loss = -q_values_pi.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss.item(), critic_loss.item()
        
# --- TESTİ BAŞLATAN FONKSİYONLAR (DÜZELTİLMİŞ) ---

def fill_buffer(env, buffer, num_steps):
    """Testin başlayabilmesi için buffer'ı sahte veriyle doldurur"""
    print(f"Replay Buffer dolduruluyor ({num_steps} adım)...")
    # (Hata düzeltmesi: Gözlem alanı shape'i (C, H, W) olmalı)
    sahte_obs = np.random.rand(env.obs_shape[0], env.obs_shape[1], env.obs_shape[2]).astype(np.float32)
    for _ in range(num_steps):
        buffer.add(sahte_obs, sahte_obs, np.random.rand(env.action_dim), 0.5, False, [])
    print("Buffer doldurma tamamlandı.")

def run_test(test_function_name_str, config, num_steps=1000):
    """Belirtilen test fonksiyonunu (isime göre) çalıştırır ve profillemesini yapar"""
    
    env = FullSystemTest(config)
    fill_buffer(env, env.replay_buffer, config['batch_size'])
    
    print(f"\n--- TEST BAŞLIYOR: {test_function_name_str} ---")
    print(f"--- (Ayarlar: dt={config['dt']}, train_freq={config['train_freq']}, batch_size={config['batch_size']}) ---")
    
    test_action = np.array([0.5, 0.1])
    
    z_pozisyonlari = []
    # Z-BUG DÜZELTMESİ: Başlangıç Z'sini güvenli bir şekilde al
    env.client.simContinueForTime(0.01) # Simülasyonun "oturması" için (dt'den küçük)
    z_baslangic = env.get_z_pos()
    z_pozisyonlari.append(z_baslangic)
    print(f"Güvenilir başlangıç irtifası ayarlandı: {z_baslangic:.2f}m")

    total_t1 = 0
    total_t2 = 0
    total_t3 = 0
    total_loop = 0
    
    t_test_start = time.time()

    for i in range(num_steps):
        t_loop_start = time.time()
        
        # --- T3: EĞİTİM (DÜZELTİLMİŞ MANTIK) ---
        t = time.time()
        # 'i' (adım sayısı) 'train_freq'e bölünebiliyorsa T3'ü çalıştır
        if (i > 0) and (i % env.cfg['train_freq']) == 0:
            # gradient_steps=256 olduğu için 256 kez çalıştır
            for _ in range(env.cfg['gradient_steps']):
                env.run_training_step()
        t3 = time.time() - t
        # ------------------------------------
        
        # --- T1: FİZİK ---
        t = time.time()
        # (Test 1 veya Test 2'yi burada çağır)
        if test_function_name_str == "test_mevcut_mimari":
             env.set_action_mevcut_mimari(test_action)
        else:
             # 'test_onerilen_mimari'yi taklit et
             env.set_action_onerilen_mimari(test_action) # Komutu gönder (hızlı)
             env.client.simContinueForTime(env.dt) # Bekle (yavaş)
        t1 = time.time() - t
        
        # --- T2: GÖZLEM ---
        t = time.time()
        obs = env.get_obs()
        t2 = time.time() - t
        
        # --- T4: Z-KONTROL ---
        z_pos = env.get_z_pos()
        z_pozisyonlari.append(z_pos)
        
        total_t1 += t1
        total_t2 += t2
        total_t3 += t3
        total_loop += (time.time() - t_loop_start)
        
        if (i + 1) % 100 == 0:
            print(f"Adım {i+1}/{num_steps} | Z-İrtifa: {z_pos:.2f} (Başlangıç: {z_baslangic:.2f})")

    t_test_end = time.time()
    
    # --- Sonuçları Raporla ---
    print("\n--- TEST BİTTİ ---")
    avg_t1 = (total_t1 / num_steps) * 1000
    avg_t2 = (total_t2 / num_steps) * 1000
    avg_t3 = (total_t3 / num_steps) * 1000 # Bu artık T3 yükünün ortalamasıdır
    avg_loop = (total_loop / num_steps) * 1000
    
    print(f"Ortalama Adım Süresi (Loop Time): {avg_loop:.2f} ms")
    print(f"  -> T1 (Fizik/Hareket): {avg_t1:.2f} ms")
    print(f"  -> T2 (Gözlem/Görüntü): {avg_t2:.2f} ms")
    print(f"  -> T3 (Eğitim/Donma): {avg_t3:.2f} ms  <-- (T3 yükünün 1 adıma düşen ortalaması)")
    
    total_test_time = t_test_end - t_test_start
    final_fps = num_steps / total_test_time
    print(f"\nToplam Süre: {total_test_time:.2f} saniye ({num_steps} için)")
    print(f"ORTALAMA FPS (T1+T2+T3): {final_fps:.2f}")
    
    # Z-BUG DÜZELTMESİ: Kaymayı, ilk güvenilir adımdan (indeks 1) itibaren hesapla
    # (Hata düzeltmesi: indeks 1 değil, 20. adımdan itibaren alalım)
    if len(z_pozisyonlari) > 20:
        z_ilk_adim_sonrasi = z_pozisyonlari[20] # 20. adımdaki Z
        z_son_adim = z_pozisyonlari[-1]
        z_kaymasi = abs(z_ilk_adim_sonrasi - z_son_adim)
        print(f"İrtifa Kayması (Z-Drift): {z_kaymasi:.4f} metre (Güvenilir ölçüm)")
    else:
        print("İrtifa kayması hesaplanamadı (yeterli adım yok).")
    
    env.client.simPause(False)
    env.client.armDisarm(False)

# --- ANA ÇALIŞTIRMA BÖLÜMÜ (İsteğiniz Üzere Değiştirildi) ---
if __name__ == "__main__":
    
    # Tüm çözümleri içeren nihai config
    CONFIG = {
        # Görüntü
        "screen_height": 70,
        "screen_width": 200,
        "max_depth_meters": 15.0,
        # Fizik (T1 ÇÖZÜMÜ)
        "dt": 0.04, # 100ms'den 20ms'ye
        "acc_xy_max": 2.0,
        "v_xy_max": 5.0,
        "v_xy_min": 0.5,
        "yaw_rate_max_deg": 60.0,
        # Ağ
        "state_feature_dim": 4, 
        "features_dim": 132,  
        "net_arch": [64, 32, 16], 
        # Eğitim (T3 ÇÖZÜMÜ)
        "learning_rate": 3e-4,
        "buffer_size": 100000,
        "batch_size": 512,
        "train_freq": 512, # 256 adımda BİR eğit
        "gradient_steps": 256 # 1:1 oranı
    }
    
    # HANGİ TESTİ ÇALIŞTIRMAK İSTİYORSUNUZ?
    # (Sadece birini seçin)
    
    # --- Test 1: Sizin mevcut mimariniz ---
    run_test("test_mevcut_mimari", CONFIG, num_steps=2048)
    
    # --- Test 2: Benim önerdiğim mimari ---
    #run_test("test_onerilen_mimari", CONFIG, num_steps=1000)