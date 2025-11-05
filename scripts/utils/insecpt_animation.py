import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- AYARLAR ---
BASE_PATH = r"C:\Users\Studentenaccount\Desktop\Ararat Cetinkaya MA\DRL-New\UAV_Navigation_DRL_AirSim\best_eval\SimpleAvoid\2025_10_26_04_09_Multirotor_CNN_GAP_SAC\eval_50_NH_center_Multirotor"
#"C:\Users\Studentenaccount\Desktop\Ararat Cetinkaya MA\DRL-New\UAV_Navigation_DRL_AirSim\best_eval\SimpleAvoid\2025_10_26_04_09_Multirotor_CNN_GAP_SAC\eval_50_SimpleAvoid_Multirotor"

FILE_TO_INSPECT = 'obs_eval.npy'
# Loglarınıza göre EP 7 bir çarpışmaydı. Python'da bu index 6'dır.
EPISODE_TO_INSPECT = 6 

FRAME_PAUSE_TIME = 0.05 
# --- AYARLAR SONU ---

def load_data(path, file_name):
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        print(f"HATA: Dosya bulunamadı! -> {full_path}")
        return None
    try:
        data = np.load(full_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"HATA: '{file_name}' dosyası yüklenirken bir hata oluştu: {e}")
        return None

def visualize_episode_dual_channel(observations_data, episode_index):
    print(f"\nBölüm {episode_index} için çift kanallı DÜZELTİLMİŞ animasyon başlatılıyor...")
    
    if observations_data is None or episode_index >= len(observations_data):
        print(f"Hata: Bölüm {episode_index} için gözlem verisi bulunamadı.")
        return

    try:
        episode_data = [step for step in observations_data[episode_index] 
                        if step is not None and hasattr(step, 'shape')]
        
        if not episode_data:
            print(f"Bölüm {episode_index} için geçerli adım bulunamadı.")
            return
            
    except Exception as e:
        print(f"Bölüm {episode_index} verisi işlenirken hata: {e}")
        return

    num_steps = len(episode_data)
    print(f"Bölüm {episode_index} toplam {num_steps} adımdan oluşuyor. Oynatılıyor...")

    plt.figure(figsize=(15, 7))
    plt.ion() 

    for i, step_observation in enumerate(episode_data):
        try:
            obs_np = np.array(step_observation, dtype=float)
            
            if obs_np.ndim != 3 or obs_np.shape[2] != 2:
                print(f"  Adım {i} atlanıyor (format hata: {obs_np.shape})")
                continue

            channel_0 = obs_np[:, :, 0]
            channel_1 = obs_np[:, :, 1]
            
            plt.clf() 
            plt.suptitle(f'Bölüm {episode_index} - Adım {i} / {num_steps-1}', fontsize=16)

            # --- DÜZELTME 1 ---
            # 'origin='lower'' SİLİNDİ. (Varsayılan 'upper' kullanılacak)
            ax1 = plt.subplot(1, 2, 1)
            im1 = ax1.imshow(channel_0, cmap='viridis') 
            ax1.set_title('Kanal 1 (Index 0) - Düzeltilmiş')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            
            # --- DÜZELTME 2 ---
            # 'origin='lower'' SİLİNDİ. (Varsayılan 'upper' kullanılacak)
            ax2 = plt.subplot(1, 2, 2)
            im2 = ax2.imshow(channel_1, cmap='viridis')
            ax2.set_title('Kanal 2 (Index 1) - Düzeltilmiş')
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.pause(FRAME_PAUSE_TIME) 

        except Exception as e:
            print(f"  Adım {i} çizilirken hata: {e}")
            
    plt.ioff()
    print(f"Bölüm {episode_index} animasyonu tamamlandı.")
    plt.show() 


def main_animator():
    print(f"'{FILE_TO_INSPECT}' dosyası yükleniyor...")
    data = load_data(BASE_PATH, FILE_TO_INSPECT)
    if data is None:
        return
        
    visualize_episode_dual_channel(data, EPISODE_TO_INSPECT)


if __name__ == "__main__":
    main_animator()