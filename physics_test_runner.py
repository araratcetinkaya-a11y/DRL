import airsim
import numpy as np
import math
import time

# ==========================================
# 1. TEST KONFİGÜRASYONU (BURADAN DEĞİŞTİRİN)
# ==========================================
TEST_CFG = {
    'dt': 0.06,                  # Simülasyon adım süresi (saniye)
    'acc_xy_max': 2,             # Maksimum Lineer İvme (m/s^2)
    'v_xy_max': 15,               # Maksimum Lineer Hız (m/s)
    'v_xy_min': 0.0,             # Minimum Hız
    'yaw_rate_max_deg': 100,    # Maksimum Dönüş Hızı (derece/s)
    'yaw_accel_deg_s2': 20,      # Dönüş İvmesi (derece/s^2)
    'start_z': -15.0,            # Hedef irtifa (AirSim'de yukarı negatiftir)
}

print("-" * 50)
print(f"TEST BAŞLATILIYOR...")
print(f"DT (Adım Süresi): {TEST_CFG['dt']} s")
print(f"Max Lineer Hız  : {TEST_CFG['v_xy_max']} m/s")
print(f"Max Lineer İvme : {TEST_CFG['acc_xy_max']} m/s^2")
print(f"Max Yaw Hızı    : {TEST_CFG['yaw_rate_max_deg']} deg/s")
print("-" * 50)

airsim.types

class PhysicsTester:
    """
    Orijinal 'set_action' mantığını birebir kopyalayan test sınıfı.
    """
    def __init__(self, cfg):
        self.dt = cfg['dt']

        # AirSim Bağlantısı
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Fizik Parametreleri
        self.acc_xy_max = cfg['acc_xy_max']
        self.v_xy_max = cfg['v_xy_max']
        self.v_xy_min = cfg['v_xy_min']
        self.yaw_rate_max_deg = cfg['yaw_rate_max_deg']

        # Delta Hesaplamaları
        self.yaw_accel_max_deg_s2 = cfg['yaw_accel_deg_s2']
        self.yaw_delta_max_deg = self.yaw_accel_max_deg_s2 
        self.v_xy_delta_max = self.acc_xy_max 

        # Değişkenler
        self.start_position = [0, 0, 0]  # Referans için
        self.v_xy_sp = 0
        self.prev_v_xy_sp = 0
        self.yaw_rate_sp_deg = 0
        self.prev_yaw_rate_deg = 0
        self.v_z_sp = 0  # 2D test ettiğimiz için 0 varsayıyoruz
        self.target_z = cfg['start_z']  # Sabit irtifa

    def reset_and_takeoff(self, target_z):
        print(f"Resetleniyor ve {target_z} metreye (NED) kalkış yapılıyor...")

        # Simülasyon koşsun (pause kapalı)
        self.client.simPause(False)

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Hızlı kalkış
        self.client.moveToZAsync(target_z, 5).join()

        # Değişkenleri sıfırla
        self.prev_v_xy_sp = 0.0
        self.prev_yaw_rate_deg = 0.0
        self.v_xy_sp = 0.0

        print("Kalkış tamamlandı. Stabilizasyon bekleniyor...")
        time.sleep(1)

        # Bundan sonrası için simülasyonu pause moduna al
        self.client.simPause(True)
        print("Simülasyon pause moduna alındı, dt adımları simContinueForTime(dt) ile ilerleyecek.")

    def get_data(self):
        # Gerçek verileri okuma
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        vel = state.kinematics_estimated.linear_velocity
        ang_vel = state.kinematics_estimated.angular_velocity

        # Hız vektörünün büyüklüğü (XY düzlemi)
        real_v_xy = math.sqrt(vel.x_val ** 2 + vel.y_val ** 2)

        # Yaw (Radyan -> Derece)
        orientation = airsim.to_eularian_angles(state.kinematics_estimated.orientation)
        real_yaw_deg = math.degrees(orientation[2])

        return {
            'x': pos.x_val,
            'y': pos.y_val,
            'z': pos.z_val,
            'real_v_xy': real_v_xy,
            'real_yaw_rate': math.degrees(ang_vel.z_val),
            'real_yaw': real_yaw_deg
        }

    def stop(self):
        print("Test bitti. Simülasyon serbest bırakılıyor...")

        # 1) Zamanı tekrar akıtmaya başla
        self.client.simPause(False)

        # 2) Drone'u durdur
        self.client.moveByVelocityAsync(0, 0, 0, 1).join()

        # 3) Motorları kapat (isteğe bağlı ama temiz)
        self.client.armDisarm(False)

        # 4) API kontrolünü bırak
        self.client.enableApiControl(False)

        print("Kontrol AirSim'e devredildi.")

    def set_action(self, action):
        # Güvenlik kontrolü
        if len(action) < 2:
            raise ValueError(f"action uzunluğu 2 olmalı, ancak {len(action)} geldi: {action}")

        # 1. Aksiyonları çöz (Delta)
        delta_v_xy = action[0] * self.v_xy_delta_max
        delta_yaw_rate_deg = action[1] * self.yaw_delta_max_deg

        # 2. Yeni hedef hızları hesapla
        self.v_xy_sp = np.clip(
            self.prev_v_xy_sp + delta_v_xy,
            self.v_xy_min,
            self.v_xy_max
        )

        self.yaw_rate_sp_deg = np.clip(
            self.prev_yaw_rate_deg + delta_yaw_rate_deg,
            -self.yaw_rate_max_deg,
            self.yaw_rate_max_deg
        )

        # 3. Mevcut yönü oku
        pose = self.client.simGetVehiclePose()
        yaw_rad = airsim.to_eularian_angles(pose.orientation)[2]

        # 4. Gövde hızını Dünya hızına çevir
        vx_world_sp = self.v_xy_sp * math.cos(yaw_rad)
        vy_world_sp = self.v_xy_sp * math.sin(yaw_rad)

        # 5. Komutu uygula
        self.client.moveByVelocityZAsync(
            vx_world_sp,
            vy_world_sp,
            self.target_z,  # sabit irtifa member'dan
            self.dt,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=self.yaw_rate_sp_deg)
        )

        # 6. Simülasyonu dt kadar ilerlet (pause modundaysa)
        self.client.simContinueForTime(self.dt)

        # 7. Hafızaya al
        self.prev_v_xy_sp = self.v_xy_sp
        self.prev_yaw_rate_deg = self.yaw_rate_sp_deg


# ==========================================
# TEST SENARYOSU ÇALIŞTIRICI
# ==========================================
def run_test():
    tester = PhysicsTester(TEST_CFG)

    try:
        tester.reset_and_takeoff(TEST_CFG['start_z'])

        sim_time = 0.0

        # -------------------------------------------------
        # FAZ 0: STABİLİZASYON
        # -------------------------------------------------
        print("\n--- FAZ 0: Stabilizasyon (2s) ---")
        header0 = f"{'Time(s)':<8} | {'Act_lin':<8} | {'Tgt_Spd':<8} | {'Real_Spd':<8} | {'X':<8} | {'Y':<8} | {'Z':<8} | {'YawRate':<8}"
        print(header0)
        print("-" * len(header0))

        for _ in range(int(2.0 / TEST_CFG['dt'])):
            action = [0.0, 0.0]
            tester.set_action(action)
            data = tester.get_data()
            sim_time += TEST_CFG['dt']
            print(f"{sim_time:05.2f}    | "
                  f"{action[0]:<8.1f} | "
                  f"{tester.v_xy_sp:<8.2f} | "
                  f"{data['real_v_xy']:<8.2f} | "
                  f"{data['x']:<8.2f} | "
                  f"{data['y']:<8.2f} | "
                  f"{data['z']:<8.2f} | "
                  f"{data['real_yaw_rate']:.2f}")

        
        # -------------------------------------------------
        # FAZ 1: LİNEER HIZLANMA
        # -------------------------------------------------
        print("\n--- FAZ 1: Hızlanma Testi (4s - Action 1.0) ---")
        header1 = f"{'Time(s)':<8} | {'Act_lin':<8} | {'Tgt_Spd':<8} | {'Real_Spd':<8} | {'V_theory':<8} | {'Err':<8} | {'X':<8} | {'Y':<8} | {'Z':<8} | {'YawRate':<8}"
        print(header1)
        print("-" * len(header1))

        t_phase = 0.0
        acc = TEST_CFG['acc_xy_max']
        v_max = TEST_CFG['v_xy_max']

        for _ in range(int(20.0 / TEST_CFG['dt'])):
            action = [1.0, 0.0]
            tester.set_action(action)
            data = tester.get_data()
            sim_time += TEST_CFG['dt']
            t_phase += TEST_CFG['dt']

            v_theory = min(acc * t_phase, v_max)
            err = data['real_v_xy'] - v_theory

            print(f"{sim_time:05.2f}    | "
                  f"{action[0]:<8.1f} | "
                  f"{tester.v_xy_sp:<8.2f} | "
                  f"{data['real_v_xy']:<8.2f} | "
                  f"{v_theory:<8.2f} | "
                  f"{err:<8.2f} | "
                  f"{data['x']:<8.2f} | "
                  f"{data['y']:<8.2f} | "
                  f"{data['z']:<8.2f} | "
                  f"{data['real_yaw_rate']:.2f}")

        # FAZ 1B: KADEMELİ LİNEER HIZLANMA (RAMPALI ACTION)
        # -------------------------------------------------
        print("\n--- FAZ 1B: Kademeli Hızlanma (4s - Action ramp) ---")
        header1b = f"{'Time(s)':<8} | {'Act_lin':<8} | {'Tgt_Spd':<8} | {'Real_Spd':<8} | {'X':<8} | {'Y':<8} | {'Z':<8} | {'YawRate':<8}"
        print(header1b)
        print("-" * len(header1b))

        # Faz başında zamanı ve setpointleri sıfırla
        t_phase = 0.0
        tester.prev_v_xy_sp = 0.0
        tester.v_xy_sp = 0.0

        for _ in range(int(4.0 / TEST_CFG['dt'])):
            # 0–1 s: 0.1, 1–2 s: 0.3, 2–3 s: 0.6, 3–4 s: 1.0
            if t_phase < 1.0:
                act_lin = 0.1
            elif t_phase < 2.0:
                act_lin = 0.3
            elif t_phase < 3.0:
                act_lin = 0.6
            else:
                act_lin = 1.0

            action = [act_lin, 0.0]
            tester.set_action(action)
            data = tester.get_data()

            sim_time += TEST_CFG['dt']
            t_phase += TEST_CFG['dt']

            print(f"{sim_time:05.2f}    | "
                  f"{action[0]:<8.2f} | "
                  f"{tester.v_xy_sp:<8.2f} | "
                  f"{data['real_v_xy']:<8.2f} | "
                  f"{data['x']:<8.2f} | "
                  f"{data['y']:<8.2f} | "
                  f"{data['z']:<8.2f} | "
                  f"{data['real_yaw_rate']:.2f}")


        # -------------------------------------------------
        # FAZ 2: LİNEER FRENLEME
        # -------------------------------------------------
        print("\n--- FAZ 2: Frenleme Testi (4s - Action -1.0) ---")
        header2 = header1  # aynı kolonlar
        print(header2)
        print("-" * len(header2))

        t_phase = 0.0
        v0 = tester.v_xy_sp  # fren başlangıcındaki hedef hız

        for _ in range(int(4.0 / TEST_CFG['dt'])):
            action = [-1.0, 0.0]
            tester.set_action(action)
            data = tester.get_data()
            sim_time += TEST_CFG['dt']
            t_phase += TEST_CFG['dt']

            v_theory = max(v0 - acc * t_phase, 0.0)
            err = data['real_v_xy'] - v_theory

            print(f"{sim_time:05.2f}    | "
                  f"{action[0]:<8.1f} | "
                  f"{tester.v_xy_sp:<8.2f} | "
                  f"{data['real_v_xy']:<8.2f} | "
                  f"{v_theory:<8.2f} | "
                  f"{err:<8.2f} | "
                  f"{data['x']:<8.2f} | "
                  f"{data['y']:<8.2f} | "
                  f"{data['z']:<8.2f} | "
                  f"{data['real_yaw_rate']:.2f}")

                # -------------------------------------------------
        
        # -------------------------------------------------
        # FAZ 3: AÇISAL HIZ (YAW) TESTİ
        # -------------------------------------------------
        print("\n--- FAZ 3: Yaw Testi (3s - Action [0, 1.0]) ---")
        tester.prev_v_xy_sp = 0

        yaw_header = f"{'Time(s)':<8} | {'Act_yaw':<8} | {'Tgt_YawRate':<11} | {'Real_YawRate':<13} | {'Yaw(deg)':<8} | {'X':<8} | {'Y':<8} | {'Z':<8}"
        print(yaw_header)
        print("-" * len(yaw_header))

        for _ in range(int(3.0 / TEST_CFG['dt'])):
            action = [0.0, 1.0]
            tester.set_action(action)
            data = tester.get_data()
            sim_time += TEST_CFG['dt']

            print(f"{sim_time:05.2f}    | "
                  f"{action[1]:<8.1f} | "
                  f"{tester.yaw_rate_sp_deg:<11.2f} | "
                  f"{data['real_yaw_rate']:<13.2f} | "
                  f"{data['real_yaw']:<8.2f} | "
                  f"{data['x']:<8.2f} | "
                  f"{data['y']:<8.2f} | "
                  f"{data['z']:<8.2f}")

    except KeyboardInterrupt:
        print("\nTest kullanıcı tarafından durduruldu!")
    except Exception as e:
        print(f"\nTest sırasında hata oluştu: {e}")
    finally:
        tester.stop()


if __name__ == "__main__":
    run_test()
