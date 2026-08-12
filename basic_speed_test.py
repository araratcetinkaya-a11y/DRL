import airsim
import time
import math
import numpy as np

# ==========================================
# TEST AYARLARI
# ==========================================
TEST_CFG = {
    'target_speed': 20.0,        # Test edilecek hedef hız (m/s)
    'target_yaw_rate': 60.0,    # Test edilecek hedef dönüş hızı (deg/s)
    'duration': 6.0,            # Her testin süresi (saniye)
    'altitude': -15.0,          # Uçuş irtifası
    'dt': 0.1                   # Veri okuma sıklığı
}

def get_drone_data(client):
    state = client.getMultirotorState()
    vel = state.kinematics_estimated.linear_velocity
    ang_vel = state.kinematics_estimated.angular_velocity
    
    # XY düzlemindeki toplam hız
    speed_xy = math.sqrt(vel.x_val**2 + vel.y_val**2)
    yaw_rate = math.degrees(ang_vel.z_val)
    
    return speed_xy, yaw_rate

def run_step_test():
    print(f"--- STEP RESPONSE (BASAMAK TEPKİSİ) TESTİ ---")
    print(f"Hedeflenen Sabit Hız: {TEST_CFG['target_speed']} m/s")
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    try:
        # 1. KALKIŞ VE KONUMLANMA
        print("\n1. Kalkış yapılıyor ve pozisyon alınıyor...")
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.moveToZAsync(TEST_CFG['altitude'], 5).join()
        # --- BURAYI EKLE: Önce yüzünü Kuzey'e (0) dönsün ---
        print("Yön Kuzey'e (0 derece) çevriliyor...")
        client.rotateToYawAsync(0).join()  # Drone 0 dereceye döner ve biter.
        # ----------------------------------------------------
        
        # Hızı sıfırla (Hover)
        print("Hız sıfırlanıyor (Stabilizasyon)...")
        client.moveByVelocityAsync(0, 0, 0, 2).join()
        time.sleep(1)

        # ====================================================
        # TEST 1: LİNEER HIZ (STEP INPUT)
        # ====================================================
        print(f"\n>>> TEST 1 BAŞLIYOR: {TEST_CFG['target_speed']} m/s emri gönderildi! <<<")
        print(f"{'Time':<6} | {'Komut':<8} | {'Gerçek_Hız':<10} | {'Durum'}")
        print("-" * 40)
        
        start_time = time.time()
        
        while (time.time() - start_time) < TEST_CFG['duration']:
            loop_start = time.time()
            elapsed = loop_start - start_time
            
            # --- STEP INPUT: Direkt sabit hız gönderiyoruz ---
            # İvmelenme hesabı yok, direkt "5 m/s git" diyoruz.
            client.moveByVelocityZAsync(
                TEST_CFG['target_speed'], # Vx
                0,                        # Vy
                TEST_CFG['altitude'],     # Z
                1,                        # Duration
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0)
            ).join()
            
            # Veri Oku
            real_speed, _ = get_drone_data(client)
            
            # Hıza ulaştı mı kontrolü (Tolerans %5)
            status = ""
            if real_speed >= TEST_CFG['target_speed'] * 0.95:
                status = "HEDEFE ULASILDI"
            elif real_speed >= TEST_CFG['target_speed'] * 0.63:
                status = "Time Constant (Tau)"
            
            print(f"{elapsed:05.2f}  | {TEST_CFG['target_speed']:<8.1f} | {real_speed:<10.2f} | {status}")
            
            # Döngü hızı ayarı
            remainder = TEST_CFG['dt'] - (time.time() - loop_start)
            if remainder > 0: time.sleep(remainder)

        # ====================================================
        # ARA BOŞLUK: DURMA
        # ====================================================
        print("\n--- Frenleme Testi (Hız 0 komutu) ---")
        start_time = time.time()
        while (time.time() - start_time) < 4.0:
            client.moveByVelocityZAsync(0, 0, TEST_CFG['altitude'], 1)
            real_speed, _ = get_drone_data(client)
            print(f"Frenleme | Komut: 0.0 | Gerçek: {real_speed:.2f}")
            time.sleep(0.1)

        # ====================================================
        # TEST 2: YAW HIZI (STEP INPUT)
        # ====================================================
        print(f"\n>>> TEST 2 BAŞLIYOR: Yaw Rate {TEST_CFG['target_yaw_rate']} deg/s emri! <<<")
        print(f"{'Time':<6} | {'Komut':<8} | {'Gerçek_Yaw':<10}")
        print("-" * 40)
        
        start_time = time.time()
        while (time.time() - start_time) < TEST_CFG['duration']:
            loop_start = time.time()
            elapsed = loop_start - start_time
            
            client.moveByVelocityZAsync(
                0, 0, TEST_CFG['altitude'], 1,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=TEST_CFG['target_yaw_rate'])
            )
            
            _, real_yaw_rate = get_drone_data(client)
            print(f"{elapsed:05.2f}  | {TEST_CFG['target_yaw_rate']:<8.1f} | {real_yaw_rate:<10.2f}")
            
            remainder = TEST_CFG['dt'] - (time.time() - loop_start)
            if remainder > 0: time.sleep(remainder)

    except Exception as e:
        print(f"Hata: {e}")
    finally:
        print("\nTest bitti. Drone serbest bırakılıyor.")
        client.simPause(False)
        client.moveByVelocityAsync(0,0,0,1)
        client.enableApiControl(False)

if __name__ == "__main__":
    run_step_test()