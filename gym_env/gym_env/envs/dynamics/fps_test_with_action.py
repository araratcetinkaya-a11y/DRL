import airsim
import time
import math
import numpy as np

# --- Test Ayarları ---
TEST_DURATION = 15      # saniye - Her bir testin SİMÜLASYON süresi
STABILIZE_WAIT = 3      # saniye - Bekleme süresi kısaltıldı
TARGET_Z = -15.0        # metre - Test irtifası
SAMPLE_PERIOD = 0.5     # saniye - Veri yazdırma sıklığı (GERÇEK zaman)
LOOP_STEP_DURATION = 0.001 # saniye - 'set_action' döngüsünün adımı (self.dt taklidi)

# --- YÖNTEM B: DELTA KONTROL LİMİTLERİ ---
V_XY_MIN = 0.5
V_XY_MAX = 5.0
V_XY_ACCEL_MAX = 5.0    
V_XY_DELTA_MAX = V_XY_ACCEL_MAX * LOOP_STEP_DURATION

YAW_RATE_MAX = 40.0     
YAW_ACCEL_MAX = 100.0   
YAW_DELTA_MAX = YAW_ACCEL_MAX * LOOP_STEP_DURATION 


# Global değişken, başlığın sadece bir kez yazdırılmasını sağlar
g_header_printed = False

def get_rpy_degrees(client):
    """Drone'un Roll, Pitch, Yaw açılarını derece cinsinden alır."""
    state = client.getMultirotorState()
    q = state.kinematics_estimated.orientation
    pitch, roll, yaw = airsim.utils.to_eularian_angles(q)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

def reset_drone_state(client):
    """
    Drone'u sıfırlar ve simülasyonu 'sürekli çalışma' moduna alır.
    """
    global g_header_printed
    g_header_printed = False 

    print(f"  Drone (0, 0, {TARGET_Z}) konumuna sıfırlanıyor...")
    
    # 1. Simülasyonu duraklat (Adımlama modu için hazırlık)
    client.simPause(True) 
    
    client.reset()
    client.enableApiControl(True)
    client.armDisarm(True)
    
    pose = airsim.Pose(
        airsim.Vector3r(0, 0, TARGET_Z), 
        airsim.to_quaternion(0, 0, 0) # R=0, P=0, Y=0 (Kuzey'e baksın)
    )
    client.simSetVehiclePose(pose, ignore_collision=True)
    
    # 2. Kalkış için simülasyonu GEÇİCİ olarak başlat
    client.simPause(False) 
    client.moveToZAsync(TARGET_Z, 2).join() # Sadece irtifayı garantile
    client.simPause(True) # <-- Tekrar duraklat
    
    print(f"  {STABILIZE_WAIT} saniye boyunca stabilizasyon bekleniyor (adımlayarak)...")
    
    start_wait = time.time()
    last_print = 0
    
    # 3. Beklerken 'simContinueForTime' ile adımla
    while True:
        elapsed_wait = time.time() - start_wait
        if elapsed_wait >= STABILIZE_WAIT:
            break
            
        # Hover komutu gönder ve 0.1s simülasyon zamanı kadar bekle
        client.hoverAsync()
        
        # --- HATA DÜZELTMESİ: .join() SİLİNDİ ---
        # simContinueForTime ZATEN bekleyen bir fonksiyondur.
        client.simContinueForTime(0.1) 
        
        if (elapsed_wait - last_print) >= SAMPLE_PERIOD:
            last_print = elapsed_wait
            state = client.getMultirotorState()
            pos_z = state.kinematics_estimated.position.z_val
            print(f"    ...beklerken Z pozisyonu: {pos_z:<8.2f}")
    
    print("  Drone stabilizasyonu tamamlandı. Test başlıyor.")
    # 4. 'reset' sonunda simülasyon PAUSED (DURAKLATILMIŞ) modda kalır.
    #    'run_test_case' fonksiyonu onu adımlayacaktır.

# --- run_test_case_static (Atlandı) ---
def run_test_case_static(client, vx, vy, drivetrain, yaw_mode, test_name):
    print(f"\n--- 🚀 TEST BAŞLIYOR: {test_name} ---")
    print("   (Statik test - DRL döngüsü taklit edilmiyor, atlanıyor)")
    pass


# --- YENİ TEST FONKSİYONU (YÖNTEM B + simContinueForTime) ---
def run_test_case_delta_control_HIGH_FPS(client, target_v_xy, target_yaw_rate, test_name):
    """
    (Senaryo 6 ve 7 - YÜKSEK FPS VERSİYONU)
    'set_action' mantığını 'simContinueForTime' kullanarak taklit eder.
    Bu, 10x ClockSpeed ile ~50-90 FPS hızında çalışmalıdır.
    """
    global g_header_printed
    
    print(f"\n--- 🚀 YÜKSEK FPS TESTİ BAŞLIYOR: {test_name} ---")
    print(f"  Dinamik Komutlar (Yöntem B: Delta Kontrolü Taklidi):")
    print(f"    Hedef V_xy: {target_v_xy} m/s | Hedef Yaw Hızı: {target_yaw_rate} deg/s")
    print(f"    Loop Adım Süresi (dt): {LOOP_STEP_DURATION}s (Simülasyon Zamanı)")
    print("-" * 120)
    print(f"{'Gerçek Zaman':<12} | {'Sim Zamanı':<10} | {'Adım #':<6} | {'Vxy(Hedef)':<10} | {'YawHız(Hedef)':<13} | {'Yaw(Gerçek)':<12} | {'Pos Z':<8}")
    print("-" * 120)
    g_header_printed = True
    
    # Simülasyonun 'reset_drone_state' tarafından PAUSED modda bırakıldığını varsayıyoruz
    
    start_real_time = time.time()  # GERÇEK zamanı ölç
    start_state = client.getMultirotorState() # Mevcut (duraklatılmış) durumu al
    start_sim_time_ns = start_state.timestamp
    
    last_print_real_time = start_real_time
    
    # set_action'ın hafızasında tutacağı dahili değişkenler
    current_v_xy_sp = 0.0
    current_yaw_rate_sp = 0.0
    step_count = 0
    elapsed_sim_time = 0.0 # Simülasyon zamanını manuel takip et
    
    while True:
        # --- 'set_action' (Yöntem B) Mantığı Başlangıcı ---
        
        # 1. & 2. HEDEFLERİ GÜNCELLE (Hesaplama)
        # (Bu kısım aynı, simülasyon çağrısı yapmıyor)
        if current_v_xy_sp < target_v_xy:
            current_v_xy_sp += V_XY_DELTA_MAX
            current_v_xy_sp = min(current_v_xy_sp, target_v_xy) 
        elif current_v_xy_sp > target_v_xy:
            current_v_xy_sp -= V_XY_DELTA_MAX
            current_v_xy_sp = max(current_v_xy_sp, target_v_xy)
        current_v_xy_sp = np.clip(current_v_xy_sp, 0.0, V_XY_MAX)
        if target_v_xy > 0 and current_v_xy_sp < V_XY_MIN:
            current_v_xy_sp = V_XY_MIN
            
        if current_yaw_rate_sp < target_yaw_rate:
            current_yaw_rate_sp += YAW_DELTA_MAX
            current_yaw_rate_sp = min(current_yaw_rate_sp, target_yaw_rate)
        elif current_yaw_rate_sp > target_yaw_rate:
            current_yaw_rate_sp -= YAW_DELTA_MAX
            current_yaw_rate_sp = max(current_yaw_rate_sp, target_yaw_rate)
        current_yaw_rate_sp = np.clip(current_yaw_rate_sp, -YAW_RATE_MAX, YAW_RATE_MAX)
            
        # 3. DURUMU OKU (Mevcut Yön)
        # Duraklatılmış simülasyondan son durumu okuyoruz
        _roll, _pitch, current_yaw_deg = get_rpy_degrees(client) 
        current_yaw_rad = math.radians(current_yaw_deg)
        
        # 4. HESAPLA (Gövde -> Dünya Çevirisi)
        vx_world = current_v_xy_sp * math.cos(current_yaw_rad)
        vy_world = current_v_xy_sp * math.sin(current_yaw_rad)
        
        # 5. KOMUT GÖNDER (YENİ MANTIK - '.join()' YOK)
        # Komutu sıraya al (simülasyon hala duraklatılmış)
        client.moveByVelocityZAsync(
            vx=vx_world, 
            vy=vy_world, 
            z=TARGET_Z, 
            duration=1.0, 
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=current_yaw_rate_sp)
        ) # .join() SİLİNDİ
        
        # --- 'set_action' Mantığı Bitişi ---

        # 6. SİMÜLASYONU 'dt' KADAR İLERLET VE BEKLE
        # Bu, .join()'un yerini alır. Simülasyonu başlatır,
        # 0.1s (sim zamanı) çalıştırır ve TEKRAR DURAKLATIR.
        
        # --- HATA DÜZELTMESİ: .join() SİLİNDİ ---
        client.simContinueForTime(LOOP_STEP_DURATION) 
        
        step_count += 1
        now_real_time = time.time()
        
        # Simülasyon zamanını manuel olarak artır
        elapsed_sim_time += LOOP_STEP_DURATION

        # Test süresi (Simülasyon zamanına göre) dolduysa çık
        if elapsed_sim_time >= TEST_DURATION:
            break
        
        # 7. Veri Yazdır (Gerçek zamana göre)
        if (now_real_time - last_print_real_time) >= SAMPLE_PERIOD:
            last_print_real_time = now_real_time
            
            # Durum okumayı sadece yazdırma yaparken yap (hız için)
            now_state = client.getMultirotorState() 
            pos_z = now_state.kinematics_estimated.position.z_val
            
            print(f"{now_real_time - start_real_time:<12.2f} | {elapsed_sim_time:<10.1f} | {step_count:<6} | {current_v_xy_sp:<10.2f} | {current_yaw_rate_sp:<13.2f} | {current_yaw_deg:<12.2f} | {pos_z:<8.2f}")
    
    # --- Döngü Bitişi ---
    
    end_real_time = time.time()
    
    # Simülasyonu tekrar çalışır moda al
    client.simPause(False) 
    client.hoverAsync() # Test bitince drone'u durdur
    
    # --- FPS HESAPLAMASI ---
    total_real_time = end_real_time - start_real_time
    if total_real_time > 0:
        calculated_fps = step_count / total_real_time
    else:
        calculated_fps = float('inf')
        
    print("-" * 120)
    print(f"--- YÜKSEK FPS TESTİ SONUÇLARI ---")
    print(f"Toplam {step_count} simülasyon adımı atıldı.")
    print(f"Geçen toplam GERÇEK süre: {total_real_time:.2f} saniye.")
    print(f"Hesaplanan Ortalama SPS (FPS): {calculated_fps:.2f}")
    print("-" * 120)

def main_test_loop():
    client = airsim.MultirotorClient()
    
    try:
        print("Simülasyona bağlanılıyor...")
        client.confirmConnection()
        client.enableApiControl(True)
        client.armDisarm(True)
        
        state = client.getMultirotorState() 
        
        if state.landed_state == airsim.LandedState.Landed:
            print("Drone yerdeyken havalanıyor...")
            # Kalkış için simülasyonu geçici olarak başlat
            client.simPause(False)
            client.takeoffAsync().join()
            client.simPause(True) # Tekrar duraklat
            time.sleep(1)

        print("\n--- KONTROLCÜ TEMİZLEME (PRIME) BAŞLIYOR ---")
        print(f"Kontrolcünün Z hedefini {TARGET_Z} olarak ayarlamak için 'moveToPosition'...")
        client.simPause(False) # Geçici başlat
        client.moveToPositionAsync(0, 0, TARGET_Z, 5).join()
        client.simPause(True) # Tekrar duraklat
        print("Kontrolcü temizlendi. 'hoverAsync' artık doğru Z hedefine sahip olmalı.")
        print("--- KONTROLCÜ TEMİZLEME (PRIME) BİTTİ ---\n")

        # --- YENİ SENARYO 6: MİNİMUM DÖNÜŞ ÇAPI ---
        reset_drone_state(client)
        run_test_case_delta_control_HIGH_FPS( # <-- YENİ HIZLI FONKSİYONU ÇAĞIR
            client=client,
            target_v_xy = V_XY_MIN,      # 0.5 m/s (Min Hız)
            target_yaw_rate = YAW_RATE_MAX, # 40.0 deg/s (Maks Dönüş)
            test_name=f"Senaryo 6: MİNİMUM DÖNÜŞ ÇAPI (V_xy={V_XY_MIN}, Yaw Hızı={YAW_RATE_MAX})"
        )
        
        # --- YENİ SENARYO 7: MAKSİMUM LİMİTLER STRES TESTİ ---
        reset_drone_state(client)
        run_test_case_delta_control_HIGH_FPS( # <-- YENİ HIZLI FONKSİYONU ÇAĞIR
            client=client,
            target_v_xy = V_XY_MAX,      # 5.0 m/s (Maks Hız)
            target_yaw_rate = YAW_RATE_MAX, # 40.0 deg/s (Maks Dönüş)
            test_name=f"Senaryo 7: MAKSİMUM LİMİTLER (V_xy={V_XY_MAX}, Yaw Hızı={YAW_RATE_MAX})"
        )
        
        print("\n--- ✅ TÜM TESTLER BİTTİ ---")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
    
    finally:
        print("\nGörev sonu temizliği yapılıyor (hızlı mod)...")
        try:
            client.simPause(False)
            if client.isApiControlEnabled():
                client.armDisarm(False)
                client.enableApiControl(False)
                print("API kontrolü bırakıldı.")
        except Exception as e:
            print(f"Temizlik sırasında hata: {e}")

if __name__ == "__main__":
    main_test_loop()