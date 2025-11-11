# Gerekli Kütüphaneler
import airsim
import time
import numpy as np
import cv2
import msgpackrpc  # 'pip install msgpack-rpc-python'
import math
import gym
from gym import spaces

# --- PyTorch (Gerçek Eğitim Yükü için) ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Stable Baselines 3 (Sadece kritik bileşenler) ---
# (Bu script'in çalışması için 'pip install stable-baselines3' gereklidir)
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
# (Sağladığınız 'CNN_GAP_new' sınıfı)
class CNN_GAP_new(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 132, state_feature_dim=4):
        super(CNN_GAP_new, self).__init__(observation_space, features_dim)
        
        self.feature_num_state = state_feature_dim
        self.feature_num_cnn = features_dim - state_feature_dim # 132 - 4 = 128
        self.feature_all = None

        # Matematik (70, 200) girişi için doğrulandı:
        self.conv1 = nn.Conv2d(1, 8, 5, 2)  # (70,200) -> (33,98)
        self.conv2 = nn.Conv2d(8, 8, 5, 2)  # (33,98) -> (15,47)
        self.conv3 = nn.Conv2d(8, 8, 3, 2)  # (15,47) -> (7,23)
        self.pool = nn.MaxPool2d(2, 3)     # (7,23) -> (2,8)
        # Flattened dim = 8 (kanal) * 2 (H) * 8 (W) = 128

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Gözlem (B, 2, 70, 200) şeklinde gelir
        depth_img = observations[:, 0:1, :, :]
        depth_img_norm = (depth_img - 0.5) * 2

        self.layer_1_out = F.relu(self.conv1(depth_img_norm))
        self.layer_2_out = F.relu(self.conv2(self.layer_1_out))
        self.layer_3_out = F.relu(self.conv3(self.layer_2_out))
        self.layer_small = self.pool(self.layer_3_out)
        
        self.flatten = torch.flatten(self.layer_small, start_dim=1)
        state_feature = observations[:, 1, 0, 0:self.feature_num_state]
        
        x = torch.cat((self.flatten, state_feature), dim=1)
        self.feature_all = x
        return x

# --- 2. SAC AĞLARININ SİMÜLASYONU ---
# (SB3'ün SACPolicy'sini taklit eden basit yapılar)
class Actor(nn.Module):
    def __init__(self, features_dim, action_dim, net_arch):
        super().__init__()
        # SB3'ün 'create_mlp' fonksiyonunu kullanarak uyumlu bir ağ oluştur
        self.policy_net = nn.Sequential(
            *create_mlp(features_dim, action_dim, net_arch, nn.Tanh)
        )
    def forward(self, obs):
        return self.policy_net(obs)

class Critic(nn.Module):
    def __init__(self, features_dim, action_dim, net_arch):
        super().__init__()
        # SB3'ün 'create_mlp' fonksiyonunu kullanarak uyumlu bir ağ oluştur
        self.q_net = nn.Sequential(
            *create_mlp(features_dim + action_dim, 1, net_arch, nn.ReLU) # Genellikle ReLU
        )
    def forward(self, obs, actions):
        # SB3 gibi, gözlem ve aksiyonu birleştirir
        obs_actions = torch.cat([obs, actions], dim=1)
        return self.q_net(obs_actions)

# --- 3. TEST ORTAMI SINIFI (Tüm Bileşenler) ---
class FullSystemTest:

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

        # --- A. Gözlem (T2) için ayarlar ---
        self.screen_height = config['screen_height']
        self.screen_width = config['screen_width']
        self.max_depth_meters = config['max_depth_meters']
        self.state_feature_dim = config['state_feature_dim']
        
        # --- B. Fizik (T1) için ayarlar ---
        self.dt = config['dt']
        self.navigation_3d = False # .ini'den
        self.start_position = [0, 0, 5] # Sabit irtifa
        
        self.v_xy_delta_max = config['acc_xy_max'] * config['dt']
        self.yaw_delta_max_deg = (config['yaw_rate_max_deg'] * 10.0) * config['dt'] # 100 deg/s^2 varsayımı
        self.v_xy_min = config['v_xy_min']
        self.v_xy_max = config['v_xy_max']
        self.yaw_rate_max_deg = config['yaw_rate_max_deg']
        self.prev_v_xy_sp = 0
        self.prev_yaw_rate_deg = 0

        # --- C. Eğitim (T3) için ayarlar ---
        self.features_dim = config['features_dim']
        self.action_dim = 2 # (2D Navigasyon)
        self.net_arch = config['net_arch']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        
        self.obs_shape = (2, self.screen_height, self.screen_width)
        self.action_shape = (self.action_dim,)
        
        # Gerçek SB3 ReplayBuffer'ı (SB3 kurulmuş olmalı)
        self.replay_buffer = ReplayBuffer(
            config['buffer_size'],
            spaces.Box(low=0, high=1, shape=self.obs_shape),
            spaces.Box(low=-1, high=1, shape=self.action_shape),
            self.device, # Veriyi otomatik olarak GPU'ya taşı
            handle_timeout_termination=False
        )
        
        # Gerçek Modelleri Kur
        self.feature_extractor = CNN_GAP_new(
            spaces.Box(low=0, high=1, shape=self.obs_shape),
            features_dim=self.features_dim,
            state_feature_dim=self.state_feature_dim
        ).to(self.device)
        
        self.actor = Actor(self.features_dim, self.action_dim, self.net_arch).to(self.device)
        self.critic = Critic(self.features_dim, self.action_dim, self.net_arch).to(self.device)
        
        # Gerçek Optimizer'lar
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        # Başlangıç pozisyonuna git ve simülasyonu BAŞLAT (artık pause(True) yok)
        # simContinueForTime kullandığımız için simülasyonun akması gerekiyor
        print("Dronu 5m irtifaya alınıyor...")
        self.client.moveToZAsync(-self.start_position[2], 2).join()
        # self.client.simPause(True) # simContinueForTime'ın çalışması için pause ETMİYORUZ
        print("İrtifa tamam. Test başlıyor.")

    # --- T2: GÖZLEM FONKSİYONU ---
    def get_obs(self):
        # 1. Görüntü
        req = [airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)]
        responses = self.client.simGetImages(req)
        r = responses[0]
        if r.width == 0: raise RuntimeError("Görüntü 0")
        image_meters = airsim.list_to_2d_float_array(r.image_data_float, r.width, r.height)
        
        image_resize = cv2.resize(image_meters, (self.screen_width, self.screen_height))
        image_normalized = np.clip(image_resize, 0, self.max_depth_meters) / self.max_depth_meters
        image_processed = 1.0 - image_normalized # Yakın = 1.0

        # 2. Durum (State) - Test için sahte (gerçekçi yük için)
        state_feature = np.zeros(self.state_feature_dim, dtype=np.float32)
        state_feature_array = np.zeros((self.screen_height, self.screen_width), dtype=np.float32)
        state_feature_array[0, 0:self.state_feature_dim] = state_feature
        
        # 3. Birleştirme (H, W, C=2)
        image_with_state = np.array([image_processed, state_feature_array], dtype=np.float32)
        image_with_state = image_with_state.swapaxes(0, 2).swapaxes(0, 1)
        return image_with_state

    # --- T4: GÜVENLİK KONTROLÜ ---
    def get_z_pos(self):
        return self.client.simGetVehiclePose().position.z_val

    # --- T1: FİZİK FONKSİYONU ("Mevcut Mimari") ---
    def set_action_mevcut_mimari(self, action):
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

        # !!! ANA DARBOĞAZ (Hipotez) BURADA !!!
        self.client.simContinueForTime(self.dt) # 100ms kilitlenme

        self.prev_v_xy_sp = self.v_xy_sp
        self.prev_yaw_rate_deg = self.yaw_rate_sp_deg

    # --- T3: GERÇEK EĞİTİM YÜKÜ ---
    def run_training_step(self):
        # 1. Veri Hazırlama (CPU İşi -> SB3 Buffer'ı GPU'ya taşır)
        replay_data = self.replay_buffer.sample(self.batch_size)
        
        # 2. Asıl Eğitim (GPU İşi - 'nvidia-smi %97')
        
        # Critic Loss
        with torch.no_grad():
            sahte_target_q = torch.randn(self.batch_size, 1).to(self.device)

        features = self.feature_extractor(replay_data.observations)
        current_q_values = self.critic(features, replay_data.actions)
        critic_loss = F.mse_loss(current_q_values, sahte_target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward() # GPU Yükü 1
        self.critic_optimizer.step()
        
        # Actor Loss
        # (SB3'te actor loss için critic'in gradyanları dondurulur, 
        # ancak yükü taklit etmek için biz tekrar hesaplıyoruz)
        features_pi = self.feature_T(replay_data.observations)
        actions_pi = self.actor(features_pi)
        q_values_pi = self.critic(features_pi, actions_pi)
        actor_loss = -q_values_pi.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward() # GPU Yükü 2
        self.actor_optimizer.step()

# --- FAZ TESTİNİ ÇALIŞTIRAN ANA FONKSİYON ---
def run_phase_test(config, num_total_steps=1500):
    
    try:
        env = FullSystemTest(config)
    except Exception as e:
        print(f"HATA: AirSim'e bağlanılamadı. Simülatör çalışıyor mu?")
        print(f"Hata detayı: {e}")
        return

    learning_starts = config['learning_starts']
    train_freq = config['train_freq']
    
    # Süreleri ve Z pozisyonunu takip etmek için listeler
    warmup_times_T1 = [] # Fizik
    warmup_times_T2 = [] # Gözlem
    
    training_times_T1 = [] # Fizik
    training_times_T2 = [] # Gözlem
    training_times_T3 = [] # Eğitim
    
    z_pozisyonlari = []
    z_baslangic = env.get_z_pos()
    z_pozisyonlari.append(z_baslangic)

    print(f"\n--- FAZ 1 (ISINMA) BAŞLIYOR ---")
    print(f"{learning_starts} adım boyunca SADECE veri toplanacak (T1 + T2)...")
    
    t_test_start = time.time()

    for i in range(num_total_steps):
        
        if i == learning_starts:
            print(f"\n--- FAZ 2 (EĞİTİM) BAŞLIYOR ---")
            print(f"Kalan adımlarda veri toplanacak VE eğitim yapılacak (T1 + T2 + T3)...")

        # Rastgele bir eylemle (hareket ederek) buffer'ı doldur
        action = np.random.rand(env.action_dim) * 2 - 1 

        # --- T1: FİZİK ---
        t = time.time()
        env.set_action_mevcut_mimari(action)
        t1 = time.time() - t
        
        # --- T2: GÖZLEM ---
        t = time.time()
        try:
            obs = env.get_obs()
        except Exception as e:
            print(f"HATA: {i}. adımda get_obs() başarısız oldu: {e}")
            continue
        t2 = time.time() - t
        
        # --- T4: Z-KONTROL ---
        z_pozisyonlari.append(env.get_z_pos())
        
        # Buffer'a (Isınma aşamasında da) ekle
        # (obs, next_obs aynı, test için önemsiz)
        env.replay_buffer.add(obs, obs, action, 0.5, False, [])

        # --- T3: EĞİTİM ---
        t3 = 0.0 # Varsayılan (Isınma fazı)
        if i >= learning_starts:
            if (i % train_freq) == 0:
                try:
                    t = time.time()
                    env.run_training_step()
                    t3 = time.time() - t
                except Exception as e:
                    print(f"HATA: {i}. adımda run_training_step() başarısız oldu: {e}")
                    # Bu genelde buffer'ın yeterince dolmamasından (örn: batch_size > i)
                    # veya bir CUDA hatasından kaynaklanır.
                    print("Test durduruluyor.")
                    break
        
        # Süreleri kaydet
        if i < learning_starts:
            warmup_times_T1.append(t1)
            warmup_times_T2.append(t2)
        else:
            training_times_T1.append(t1)
            training_times_T2.append(t2)
            if t3 > 0: # Sadece eğitim yapılan adımları kaydet
                training_times_T3.append(t3)

    t_test_end = time.time()
    
    # --- Raporlama ---
    print("\n" + "="*50)
    print("--- TEST SONUÇLARI ---")
    print("="*50)
    
    # FAZ 1 Raporu (ISINMA)
    avg_w_t1 = np.mean(warmup_times_T1) * 1000
    avg_w_t2 = np.mean(warmup_times_T2) * 1000
    avg_warmup_total_ms = avg_w_t1 + avg_w_t2
    warmup_fps = 1000.0 / avg_warmup_total_ms
    
    print(f"\nFAZ 1 (ISINMA / Sadece Veri Toplama) Raporu:")
    print(f"  Ortalama Adım Süresi: {avg_warmup_total_ms:.2f} ms")
    print(f"    -> T1 (Fizik/Hareket): {avg_w_t1:.2f} ms")
    print(f"    -> T2 (Gözlem/Görüntü): {avg_w_t2:.2f} ms")
    print(f"  >> ISINMA FPS (Sadece T1+T2): {warmup_fps:.2f} FPS")
    
    # FAZ 2 Raporu (EĞİTİM)
    if len(training_times_T1) > 0:
        avg_t_t1 = np.mean(training_times_T1) * 1000
        avg_t_t2 = np.mean(training_times_T2) * 1000
        # T3 sadece train_freq'de bir çalışır, ortalaması alınmalı
        # (config['train_freq'] == 1 olduğu için bu ortalama doğrudur)
        avg_t_t3 = np.mean(training_times_T3) * 1000 if len(training_times_T3) > 0 else 0.0
        
        avg_training_total_ms = avg_t_t1 + avg_t_t2 + avg_t_t3
        training_fps = 1000.0 / avg_training_total_ms
        
        print(f"\nFAZ 2 (EĞİTİM) Raporu (train_freq={train_freq}):")
        print(f"  Ortalama Adım Süresi: {avg_training_total_ms:.2f} ms")
        print(f"    -> T1 (Fizik/Hareket): {avg_t_t1:.2f} ms")
        print(f"    -> T2 (Gözlem/Görüntü): {avg_t_t2:.2f} ms")
        print(f"    -> T3 (Eğitim/Donma): {avg_t_t3:.2f} ms")
        print(f"  >> EĞİTİM FPS (T1+T2+T3): {training_fps:.2f} FPS")
    
    total_steps = len(warmup_times_T1) + len(training_times_T1)
    total_test_time = t_test_end - t_test_start
    final_avg_fps = total_steps / total_test_time
    
    print(f"\n--- GENEL SONUÇ ---")
    print(f"Toplam Süre: {total_test_time:.2f} saniye ({total_steps} adım için)")
    print(f"ORTALAMA FPS (Tüm Test): {final_avg_fps:.2f}")
    
    z_kaymasi = abs(z_baslangic - z_pozisyonlari[-1])
    print(f"İrtifa Kayması (Z-Drift): {z_kaymasi:.4f} metre (0'a yakın olmalı)")
    
    env.client.simPause(False)
    env.client.armDisarm(False)

# --- ANA ÇALIŞTIRMA BÖLÜMÜ ---
if __name__ == "__main__":
    
    # Sizin .ini dosyanızdan gelen değerler
    # (learning_starts hariç)
    CONFIG = {
        # Görüntü
        "screen_height": 70,
        "screen_width": 200,
        "max_depth_meters": 15.0,
        # Fizik
        "dt": 0.1,
        "acc_xy_max": 2.0,
        "v_xy_max": 5.0,
        "v_xy_min": 0.5,
        "yaw_rate_max_deg": 60.0,
        # Ağ
        "state_feature_dim": 4, 
        "features_dim": 132,  
        "net_arch": [64, 32, 16], # .ini'den
        # Eğitim (Sizin 4 FPS senaryonuz)
        "learning_rate": 3e-4,
        "buffer_size": 100000,
        "batch_size": 256,
        "train_freq": 1,
        "gradient_steps": 1,
        
        # --- TEST İÇİN DEĞİŞTİRİLEN DEĞER ---
        "learning_starts": 1000 # Fazları ayırabilmek için 1 yerine 1000
    }

    # Testi çalıştır (Sadece mevcut mimari)
    run_phase_test(CONFIG, num_total_steps=1500) # 1000 ısınma, 500 eğitim