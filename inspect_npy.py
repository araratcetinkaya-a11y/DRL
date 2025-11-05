import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- AYARLAR ---
BASE_PATH = r"C:\Users\Studentenaccount\Desktop\Ararat Cetinkaya MA\DRL-New\UAV_Navigation_DRL_AirSim\best_eval\SimpleAvoid\2025_10_26_04_09_Multirotor_CNN_GAP_SAC\eval_50_SimpleAvoid_Multirotor"
FILE_TO_INSPECT = 'obs_eval.npy'

# Loglarınıza göre EP 7 bir çarpışmaydı. Python'da bu index 6'dır.
EPISODE_TO_INSPECT = 6 

# Animasyon hızı (saniye cinsinden)
FRAME_PAUSE_TIME = 0.05 
# --- AYARLAR SONU ---

def load_data(path, file_name):
    """Veriyi yüklemek için yardımcı fonksiyon."""
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
    """
    Belirli bir bölümdeki (episode_index) tüm adımların
    iki kanalını da yan yana art arda çizer (animasyon).
    """
    print(f"\nBölüm {episode_index} için çift kanallı animasyon başlatılıyor...")
    
    if observations_data is None or episode_index >= len(observations_data):
        print(f"Hata: Bölüm {episode_index} için gözlem verisi bulunamadı.")
        return

    try:
        # O bölüme ait tüm geçerli (None olmayan) adımları al
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
    print("Oynatma penceresini kapattığınızda program sonlanacaktır.")

    plt.figure(figsize=(15, 7)) # İki resim için geniş bir figür
    plt.ion() # Matplotlib'i interaktif moda al

    for i, step_observation in enumerate(episode_data):
        try:
            obs_np = np.array(step_observation, dtype=float)
            
            # Veri formatını kontrol et
            if obs_np.ndim != 3 or obs_np.shape[2] != 2:
                print(f"  Adım {i} atlanıyor (format hatası: {obs_np.shape}, (H, W, 2) bekleniyordu)")
                continue

            # İki kanalı ayır
            channel_0 = obs_np[:, :, 0]
            channel_1 = obs_np[:, :, 1]
            
            plt.clf() # Önceki grafiği (frame'i) temizle
            
            # Ana başlık (her adımda güncellenir)
            plt.suptitle(f'Bölüm {episode_index} - Adım {i} / {num_steps-1}', fontsize=16)

            # Alt Grafik 1: KANAL 1 (Index 0)
            ax1 = plt.subplot(1, 2, 1)
            im1 = ax1.imshow(channel_0, cmap='viridis', origin='lower', vmin=0, vmax=1) # Skalayı 0-1 arası kilitle
            ax1.set_title('Kanal 1 (Index 0)')
            ax1.axis('off') # Eksenleri kapat
            
            # Alt Grafik 2: KANAL 2 (Index 1)
            ax2 = plt.subplot(1, 2, 2)
            im2 = ax2.imshow(channel_1, cmap='viridis', origin='lower', vmin=0, vmax=1) # Skalayı 0-1 arası kilitle
            ax2.set_title('Kanal 2 (Index 1)')
            ax2.axis('off') # Eksenleri kapat
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ana başlık için yer bırak
            plt.pause(FRAME_PAUSE_TIME) # 0.05 saniye bekle (video efekti için)

        except Exception as e:
            print(f"  Adım {i} çizilirken hata: {e}")
            
    plt.ioff() # İnteraktif modu kapat
    print(f"Bölüm {episode_index} animasyonu tamamlandı.")
    print("Son kareyi görmek için pencereyi açık tutun. Kapatınca program biter.")
    plt.show() # Son kareyi ekranda tut


def main_animator():
    print(f"'{FILE_TO_INSPECT}' dosyası yükleniyor...")
    data = load_data(BASE_PATH, FILE_TO_INSPECT)
    if data is None:
        return
        
    visualize_episode_dual_channel(data, EPISODE_TO_INSPECT)


if __name__ == "__main__":
    main_animator()