# test_airsim_physics.py
import os
import sys
import argparse
import time
import configparser
import numpy as np
import airsim

# ---- PATH AYARI ----
# Bu scriptin olduğu klasör
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dıştaki gym_env klasörünü sys.path'e ekle:
#  .../UAV_Navigation_DRL_AirSim/gym_env
GYM_ENV_PARENT = os.path.join(BASE_DIR, "gym_env")
if GYM_ENV_PARENT not in sys.path:
    sys.path.append(GYM_ENV_PARENT)

# Artık paket olarak import edebiliriz
from gym_env.envs.dynamics.multirotor_airsim import MultirotorDynamicsAirsim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to your .ini config file"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help="Number of physics steps to run in the forward-acceleration test"
    )
    args = parser.parse_args()

    # 1) Config yükle
    cfg = configparser.ConfigParser()
    read_files = cfg.read(args.config)
    if not read_files:
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # 2) Multirotor dinamiğini başlat
    dyn = MultirotorDynamicsAirsim(cfg)
    client: airsim.MultirotorClient = dyn.client
    dt = dyn.dt

    print(f"[INFO] dt = {dt:.4f} s")

    # 3) Drone'u Z ≈ 15 m yüksekliğe çıkar
    target_alt = 15.0  # "yükseklik" (pozitif yukarı)
    target_z_ned = -target_alt  # AirSim NED: yukarı = negatif Z

    print(f"[INFO] Moving to altitude z ≈ {target_alt} m (NED Z = {target_z_ned})")

    # Simülasyonu çalışır hale getir
    client.simPause(False)
    client.enableApiControl(True)
    client.armDisarm(True)

    # Yüksekliğe git
    client.moveToZAsync(
        target_z_ned,
        2.0  # m/s vertical speed
    ).join()

    # 2–3 saniye stabilize olmasını bekle
    client.hoverAsync().join()
    time.sleep(3.0)

    # Z konumunu oku ve yaz
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    print(f"[INFO] Stabilized position: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}")

    # 4) MultirotorDynamics içindeki start_position'ı bu yüksekliğe ayarla
    # Böylece set_action sabit bu irtifayı hedefler: moveByVelocityZAsync(..., -start_position[2], ...)
    dyn.start_position[2] = target_alt

    # Hız ve yaw-rate geçmişlerini sıfırla (temiz başlangıç)
    dyn.prev_v_xy_sp = 0.0
    dyn.prev_v_z_sp = 0.0
    dyn.prev_yaw_rate_deg = 0.0

    # Simülasyonu step-step ilerletmek için pause moduna al
    client.simPause(True)

    # 5) Sabit action ile ileri hızlanma testi
    #
    # navigation_3d = False ise action = [Δv_xy, yaw_rate_delta]
    # burada sadece ileri hızlanmayı test ediyoruz: action = [1.0, 0.0]
    action = np.array([1.0, 0.0], dtype=np.float32)

    print("\n[TEST] Forward acceleration test (constant action = [1.0, 0.0])")
    print("step\t t[s]\t X[m]\t Y[m]\t Z[m]")

    for step in range(args.steps):
        # Bir fizik adımı uygula
        dyn.set_action(action)

        # State oku
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position

        t = (step + 1) * dt
        print(f"{step:4d}\t {t:5.2f}\t {pos.x_val:6.2f}\t {pos.y_val:6.2f}\t {pos.z_val:6.2f}")

    print("\n[INFO] Test finished. You can now inspect the trajectory / speeds in logs or in AirSim.")


if __name__ == "__main__":
    main()
