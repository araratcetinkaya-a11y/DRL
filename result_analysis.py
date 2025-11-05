import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json 

"""
BU BETİK NASIL KULLANILIR:
1.  Aşağıdaki `BASE_PATH` değişkeninin değerini, analiz etmek istediğiniz
    değerlendirme (eval) klasörünün yolu ile değiştirin.
2.  GEREKLİ AYARLARI (INDEX'LERİ) kontrol edin.
3.  Betiği çalıştırın.

GEREKSİNİMLER:
    pip install numpy matplotlib
"""

# --- KULLANICI AYARLARI ---

BASE_PATH = r"C:\Users\Studentenaccount\Desktop\Ararat Cetinkaya MA\DRL-New\UAV_Navigation_DRL_AirSim\best_eval\SimpleAvoid\2025_10_26_04_09_Multirotor_CNN_GAP_SAC\eval_50_SimpleAvoid_Multirotor"
YAW_COMMAND_INDEX = 1 
DEPTH_CHANNEL_INDEX = 0
SUCCESS_FLAG_INDEX = 2

# --- ANALİZ FONKSİYONLARI (DÜZELTME 7 - Otomatik Bölüm Sayısı) ---

def analyze_trajectories(trajectories_data, num_episodes_to_run):
    """
    Tüm değerlendirme bölümlerinin (episode) yörüngelerini
    aynı 2D grafikte çizer. Sadece 'num_episodes_to_run' kadar bölümü analiz eder.
    """
    print("Yörünge analizi yapılıyor...")
    
    plt.figure(figsize=(10, 10))
    # num_episodes, artık trajectories_data.shape[0] değil, parametre olarak gelen değer.
    num_episodes = num_episodes_to_run 
    
    markers_created = False 
    start_marker, end_marker = None, None 

    for i in range(num_episodes): # Sadece 'results'ta bulunan bölüm sayısı kadar döner
        
        try:
            clean_data = [step for step in trajectories_data[i] 
                          if step is not None and hasattr(step, '__len__') and len(step) >= 2]
            
            if not clean_data: 
                 print(f"  Bölüm {i} geçerli veri içermiyor, atlanıyor.")
                 continue
            
            ep_traj = np.array(clean_data, dtype=float)
            
        except Exception as e: 
            print(f"  Bölüm {i} verisi dönüştürülemedi ({e}), atlanıyor.")
            continue
            
        if ep_traj.shape[0] < 2: 
            print(f"  Bölüm {i} çok kısa, atlanıyor.")
            continue
            
        x_coords = ep_traj[:, 0]
        y_coords = ep_traj[:, 1]
        
        if not markers_created:
            start_marker, = plt.plot(x_coords[0], y_coords[0], 'go', markersize=8, label='Başlangıç')
            end_marker, = plt.plot(x_coords[-1], y_coords[-1], 'rs', markersize=8, label='Bitiş')
            markers_created = True 
        else:
            plt.plot(x_coords[0], y_coords[0], 'go', markersize=8)
            plt.plot(x_coords[-1], y_coords[-1], 'rs', markersize=8)
            
        plt.plot(x_coords, y_coords, alpha=0.6, label=f'Bölüm {i}' if num_episodes < 10 else None)
    
    plt.title(f'Toplam {num_episodes} Bölümün Yörünge Analizi', fontsize=16)
    plt.xlabel('X Pozisyonu (m)', fontsize=12)
    plt.ylabel('Y Pozisyonu (m)', fontsize=12)
    plt.grid(True)
    plt.axis('equal') 
    
    if markers_created:
        plt.legend([start_marker, end_marker], ['Başlangıç', 'Bitiş'])
    else:
        print("\nUYARI: Hiçbir geçerli yörünge verisi bulunamadı veya çizilemedi.")
    
    print("Yörünge analizi tamamlandı.")

def analyze_actions(actions_data, yaw_index, num_episodes_to_run):
    """
    Tüm bölümlerin 'yaw rate' komutlarını alt grafiklerde çizer.
    Sadece 'num_episodes_to_run' kadar bölümü analiz eder.
    """
    print("Eylem (Action) analizi yapılıyor...")
    
    # num_episodes, artık actions_data.shape[0] değil, parametre olarak gelen değer.
    num_episodes = num_episodes_to_run
    
    cols = min(10, int(np.ceil(np.sqrt(num_episodes))))
    rows = int(np.ceil(num_episodes / cols))
    
    plt.figure(figsize=(15, 12))
    
    valid_episodes_found = 0 
    
    for i in range(num_episodes): # Sadece 'results'ta bulunan bölüm sayısı kadar döner
        
        try:
            clean_data = [step for step in actions_data[i] 
                          if step is not None and hasattr(step, '__len__') and len(step) > yaw_index]
            
            if not clean_data:
                 print(f"  Bölüm {i} geçerli eylem verisi içermiyor, atlanıyor.")
                 continue
            
            ep_actions = np.array(clean_data, dtype=float)
            
        except Exception as e:
            print(f"  Bölüm {i} eylem verisi dönüştürülemedi ({e}), atlanıyor.")
            continue
            
        if ep_actions.shape[0] < 1:
            continue
            
        try:
            yaw_commands = ep_actions[:, yaw_index]
            valid_episodes_found += 1
        except IndexError:
            print(f"HATA: YAW_COMMAND_INDEX = {yaw_index} hatalı!")
            print(f"Eylem verinizin şekli: {ep_actions.shape}. 'yaw' komutunuzun doğru indeksini girin.")
            return

        ax = plt.subplot(rows, cols, i + 1)
        ax.plot(yaw_commands)
        ax.set_title(f'Bölüm {i}')
        ax.set_ylim(-1.1, 1.1) 
        ax.grid(True)
    
    if valid_episodes_found == 0:
        print("\nUYARI: Hiçbir geçerli eylem verisi bulunamadı veya çizilemedi.")
        
    plt.suptitle(f'Eylem (Yaw Rate Komutu) Analizi (Tüm {num_episodes} Bölüm)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    print("Eylem analizi tamamlandı.")

def load_data(path, file_name):
    """Veriyi yüklemek için yardımcı fonksiyon."""
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        print(f"HATA: Dosya bulunamadı! -> {full_path}")
        print(f"Kontrol et: 'load_data' fonksiyonu '{file_name}' arıyor.")
        return None
    
    try:
        data = np.load(full_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"HATA: '{file_name}' dosyası yüklenirken bir hata oluştu: {e}")
        return None

# --- DÜZELTİLMİŞ FONKSİYON (Çarpışma Analizi) ---
# BU FONKSİYON ARTIK 'trajectories' VERİSİNİ KULLANIYOR
def visualize_crash_observations(observations, trajectories, depth_channel_index):
    """
    Başarısız olan bölümlerin (trajectories verisine göre) 
    son adımındaki derinlik görüntülerini (observations verisinden) gösterir.
    """
    print("\nÇarpma/Başarısızlık anındaki gözlemler analiz ediliyor (Düzeltilmiş Yöntem)...")

    if observations is None or trajectories is None:
        print("Gözlem veya Yörünge verileri eksik. Analiz atlanıyor.")
        return
    
    # Bölüm sayısını 'trajectories' verisinden alıyoruz (Düzeltme)
    num_episodes = trajectories.shape[0]
    
    if observations.shape[0] < num_episodes:
        print(f"UYARI: Gözlem sayısı ({observations.shape[0]}) yörünge sayısından ({num_episodes}) az. Gözlem sayısı baz alınacak.")
        num_episodes = observations.shape[0]

    failed_episode_indices = []
    
    print(f"'trajectories.npy' verisi yüklendi. {num_episodes} bölüm sonucu aranıyor.")

    try:
        # Başarısız bölümleri bulmak için 'trajectories' verisini kullan
        for i in range(num_episodes):
            ep_traj_data = trajectories[i]
            if ep_traj_data is None or len(ep_traj_data) < 1:
                print(f"  Bölüm {i} yörünge verisi yok, atlanıyor.")
                continue
            
            # Değerlendirme betiği (thread_evaluation.py) bölüm sonuna 
            # başarı/başarısızlık durumunu bir sayı olarak ekliyordu:
            # 1 = success, 2 = crash, 3 = other (time out vb.)
            
            # Son elemanın ne olduğunu kontrol et
            last_element = ep_traj_data[-1] 
            
            is_success = False
            is_crash = False

            # Bu kontrol, thread_evaluation.py'deki kaydetme mantığına dayanır:
            # traj_list.append(1), traj_list.append(2) veya traj_list.append(3)
            if isinstance(last_element, (int, float)):
                if last_element == 1:
                    is_success = True
                elif last_element == 2:
                    is_crash = True
            
            # Eğer son eleman bir sayı değilse, eski (hatalı) veridir, atla
            if not is_success and not is_crash:
                 # Belki de son eleman sayı değildir, ondan öncekine bak
                 if len(ep_traj_data) > 1 and isinstance(ep_traj_data[-2], (list, np.ndarray)):
                     # Bu, bölümün sonunun bir pozisyon verisi olduğu anlamına gelir
                     # Bu durumda, 'crash' durumunu bilemeyiz (eski/farklı bir kayıt formatı)
                     # Şimdilik bu formatı atlayalım.
                     pass
            
            if is_crash: # Sadece 'crash' (2) olanları al
                 failed_episode_indices.append(i)
                 
    except Exception as e:
        print(f"HATA: 'trajectories.npy' verisi işlenirken hata: {e}")
        print("Veri formatı beklenenden farklı olabilir. Analiz atlanıyor.")
        return

    if not failed_episode_indices:
        print("Hiç 'crash' (durum kodu 2) olarak işaretlenmiş bölüm bulunamadı.")
        # Not: Eğer 'time out' (durum kodu 3) olanları da görmek isterseniz 
        # yukarıdaki 'if is_crash:' satırını 'if not is_success:' olarak değiştirebilirsiniz.
        return
    
    print(f"Başarısız (crash) olarak işaretlenen bölümler bulundu: {failed_episode_indices}")

    num_to_display = min(len(failed_episode_indices), 16) 
    if num_to_display == 0: return 

    cols = int(np.ceil(np.sqrt(num_to_display)))
    rows = int(np.ceil(num_to_display / cols))
    
    plt.figure(figsize=(15, 12))
    display_count = 0

    for ep_idx in failed_episode_indices:
        if display_count >= num_to_display:
            break

        try:
            # Gözlem verisini al
            clean_obs_data = [step for step in observations[ep_idx] 
                              if step is not None and hasattr(step, '__len__') and len(step) > 0]
            if not clean_obs_data:
                print(f"  Bölüm {ep_idx} için gözlem verisi yok veya bozuk, atlanıyor.")
                continue
            
            # Başarısız bölümün SON gözlemini al
            last_step_idx = len(clean_obs_data) - 1 
            last_observation = np.array(clean_obs_data[last_step_idx], dtype=float)

            # Görüntüyü çizdir
            if last_observation.ndim == 3 and last_observation.shape[-1] > depth_channel_index:
                depth_image_to_plot = last_observation[:, :, depth_channel_index]
            elif last_observation.ndim == 2:
                depth_image_to_plot = last_observation
            else:
                print(f"  Bölüm {ep_idx} için son gözlem (derinlik görüntüsü) beklendiği gibi değil ({last_observation.shape}), atlanıyor.")
                continue

            ax = plt.subplot(rows, cols, display_count + 1)
            ax.imshow(depth_image_to_plot, cmap='viridis', origin='lower') 
            ax.set_title(f'Bölüm {ep_idx} Son Görüş (Crash)')
            ax.axis('off') 
            display_count += 1

        except Exception as e:
            print(f"  Bölüm {ep_idx} için son gözlem görselleştirilemedi ({e}), atlanıyor.")
            continue
            
    if display_count > 0:
        plt.suptitle('Başarısız Bölümlerden Önceki Görüntüler (Düzeltilmiş)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        print("Hiçbir başarısız bölüm için gözlem görselleştirilemedi.")
    print("Çarpma/Başarısızlık anındaki gözlemler analizi tamamlandı.")


def visualize_episode_steps(observations_data, episode_index, depth_channel_index):
    """
    Belirli bir bölümdeki (episode_index) tüm adımların
    gözlemlerini (görüntülerini) art arda çizer.
    """
    print(f"\nBölüm {episode_index} için adım adım görselleştirme başlatılıyor...")
    
    if observations_data is None or episode_index >= len(observations_data):
        print(f"Hata: Bölüm {episode_index} için gözlem verisi bulunamadı.")
        return

    try:
        # O bölüme ait tüm adımları temizle (None vb. kaldır)
        episode_data = [step for step in observations_data[episode_index] 
                        if step is not None and hasattr(step, '__len__') and len(step) > 0]
        
        if not episode_data:
            print(f"Bölüm {episode_index} için geçerli adım bulunamadı.")
            return
            
    except Exception as e:
        print(f"Bölüm {episode_index} verisi işlenirken hata: {e}")
        return

    num_steps = len(episode_data)
    print(f"Bölüm {episode_index} toplam {num_steps} adımdan oluşuyor. Oynatılıyor...")

    plt.figure(figsize=(10, 6))
    plt.ion() # Matplotlib'i interaktif moda al

    for i, step_observation in enumerate(episode_data):
        try:
            step_obs_np = np.array(step_observation, dtype=float)
            
            # Görüntüyü al (visualize_crash_observations'daki mantığın aynısı)
            depth_image_to_plot = None
            if step_obs_np.ndim == 3 and step_obs_np.shape[-1] > depth_channel_index:
                depth_image_to_plot = step_obs_np[:, :, depth_channel_index]
            elif step_obs_np.ndim == 2:
                depth_image_to_plot = step_obs_np
            else:
                print(f"  Adım {i} atlanıyor (format hatası: {step_obs_np.shape})")
                continue

            plt.clf() # Önceki grafiği temizle
            plt.imshow(depth_image_to_plot, cmap='viridis', origin='lower')
            plt.title(f'Bölüm {episode_index} - Adım {i} / {num_steps-1}')
            plt.axis('off')
            plt.pause(0.05) # 0.05 saniye bekle (video efekti için)

        except Exception as e:
            print(f"  Adım {i} çizilirken hata: {e}")
            
    plt.ioff() # İnteraktif modu kapat
    print(f"Bölüm {episode_index} görselleştirmesi tamamlandı.")

# --- ANA KOD (Güncellendi) ---



def main():
    if not os.path.isdir(BASE_PATH):
        print(f"HATA: Klasör bulunamadı! -> {BASE_PATH}")
        print("Lütfen BASE_PATH değişkenini doğru klasör yolu ile güncelleyin.")
        return

    # 1. Verileri BİR KEZ yükle
    trajectories = load_data(BASE_PATH, 'traj_eval.npy')
    actions = load_data(BASE_PATH, 'action_eval.npy')
    obs = load_data(BASE_PATH, 'obs_eval.npy')
    # --- KANIT KODU BAŞLANGICI ---
    if obs is not None:
        print("\n--- KANIT BÖLÜMÜ: 'obs_eval.npy' İÇERİK ANALİZİ ---")
        try:
            # İlk bölümün verisini al
            ilk_bolum_verisi = obs[0]
            
            # Bu bölümdeki ilk adımı (ilk gözlemi) al
            # Bazı adımlar None olabilir, None olmayan ilk adımı bul
            ilk_gercek_gozlem = None
            for adim in ilk_bolum_verisi:
                if adim is not None and hasattr(adim, 'shape'):
                    ilk_gercek_gozlem = adim
                    break
            
            if ilk_gercek_gozlem is not None:
                print(f"Başarılı: İlk bölümün ilk gözlemi yüklendi.")
                print(f"  Veri Tipi (type): {type(ilk_gercek_gozlem)}")
                print(f"  Veri Şekli (shape): {ilk_gercek_gozlem.shape}")
                print(f"  Veri 'dtype' (dtype): {ilk_gercek_gozlem.dtype}")
                
                if len(ilk_gercek_gozlem.shape) == 2 or len(ilk_gercek_gozlem.shape) == 3:
                    print("\n  SONUÇ: Veri bir GÖRÜNTÜ (Yükseklik, Genişlik, ...). Bu, CNN'in GİRDİSİDİR.")
                elif len(ilk_gercek_gozlem.shape) == 1:
                    print(f"\n  SONUÇ: Veri bir VEKTÖR. Boyut: {ilk_gercek_gozlem.shape[0]}. Bu, muhtemelen CNN'in ÇIKTISIDIR.")
                else:
                    print("\n  SONUÇ: Veri formatı beklenmiyor.")
            else:
                print("Hata: İlk bölümde geçerli bir gözlem adımı bulunamadı.")
                
        except Exception as e:
            print(f"Hata: 'obs' verisi analiz edilirken bir istisna oluştu: {e}")
        print("--- KANIT BÖLÜMÜ SONU ---\n")
    # --- KANIT KODU SONU ---
    states = load_data(BASE_PATH, 'state_eval.npy')
    # Bu dosya sadece 4 elemanlı özet istatistikleri içerir
    results_summary = load_data(BASE_PATH, 'results.npy') 

    # *** ASIL DÜZELTME BURADA ***
    # 2. Bölüm sayısını 'results.npy' (özet) yerine 'traj_eval.npy' (tam veri) dosyasından al
    
    num_episodes_to_run = 0
    if trajectories is not None:
        num_episodes_to_run = trajectories.shape[0] # <-- DOĞRU YÖNTEM
        print(f"'traj_eval.npy' dosyasına göre {num_episodes_to_run} bölüm analiz edilecek.")
    elif actions is not None:
        num_episodes_to_run = actions.shape[0] # <-- Alternatif
        print(f"'action_eval.npy' dosyasına göre {num_episodes_to_run} bölüm analiz edilecek.")
    else:
        print("HATA: 'traj_eval.npy' veya 'action_eval.npy' yüklenemedi. Analiz durduruluyor.")
        return
        
    if results_summary is not None:
        print(f"Özet istatistikler ('results.npy') yüklendi: {results_summary}")
    
    # 3. Yörüngeleri Analiz Et
    if trajectories is not None and trajectories.shape[0] >= num_episodes_to_run:
        print(f"Yüklendi: 'traj_eval.npy', Şekil: {trajectories.shape}")
        analyze_trajectories(trajectories, num_episodes_to_run) # num_episodes_to_run gönderiliyor
    else:
        print(f"'traj_eval.npy' analiz edilemedi.")

    # 4. Eylemleri Analiz Et
    if actions is not None and actions.shape[0] >= num_episodes_to_run:
        print(f"Yüklendi: 'action_eval.npy', Şekil: {actions.shape}")
        analyze_actions(actions, YAW_COMMAND_INDEX, num_episodes_to_run) # num_episodes_to_run gönderiliyor
    else:
        print(f"'action_eval.npy' analiz edilemedi.")

    # 5. Yüklenmiş verileri kullanarak Gözlemleri Analiz Et
    if obs is not None and obs.shape[0] >= num_episodes_to_run:
        print(f"Yüklendi: 'obs_eval.npy', Şekil: {obs.shape}")
    if states is not None and states.shape[0] >= num_episodes_to_run:
        print(f"Yüklendi: 'state_eval.npy', Şekil: {states.shape}")
    
    # UYARI: 'visualize_crash_observations' fonksiyonu, 'results.npy' dosyasını
    # her bölüm için ayrı veri (1/0) bekleyerek okumaya çalışıyor.
    # Ancak 'results.npy' sadece özet (4 eleman) içeriyor.
    # Bu nedenle bu fonksiyon çarpma görüntülerini bulamayabilir.
    # Çarpma analizi için 'visualize_crash_observations' fonksiyonunun 
    # 'traj_eval.npy' dosyasını okuyacak şekilde yeniden yazılması gerekir.
    # Şimdilik, özet dosyasını gönderiyoruz:
    visualize_crash_observations(obs, trajectories, DEPTH_CHANNEL_INDEX)

    visualize_episode_steps(obs, episode_index=7, depth_channel_index=DEPTH_CHANNEL_INDEX)

    # 6. Tüm grafikleri göster
    print("\nGrafik pencereleri açılıyor...")
    plt.show()

if __name__ == "__main__":
    main()