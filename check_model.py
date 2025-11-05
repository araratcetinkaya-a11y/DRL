import sys
import os

# NOT: Bu betik kök dizinde olduğu için, 'scripts.utils' gibi
# importlar sorunsuz çalışacaktır.
from stable_baselines3.common.save_util import load_from_zip_file

# Lütfen model dosyanızın TAM YOLUNU buraya yazın.
# ÖNEMLİ: Sonuna '.zip' uzantısını eklemeyi unutmayın.
MODEL_PATH = r"C:\Users\Studentenaccount\Desktop\Ararat Cetinkaya MA\DRL-New\UAV_Navigation_DRL_AirSim\logs\SimpleAvoid\2025_08_31_10_46_Multirotor_CNN_GAP_SAC\models\model_sb3.zip"


print(f"Model dosyası inceleniyor:\n{MODEL_PATH}\n")

if not os.path.exists(MODEL_PATH):
    print("HATA: Model dosyası bu yolda bulunamadı. Yolu kontrol edin.")
    sys.exit()

try:
    # Modeli yüklemeden sadece içindeki verileri (metadata) oku
    # Bu komut, modeli eğitirken kullanılan ortamın 
    # özelliklerini (action_space, observation_space) içeren
    # 'data' adında bir sözlük döndürür.
    data, params, pytorch_variables = load_from_zip_file(MODEL_PATH)

    # 'data' sözlüğünden 'action_space' anahtarını çek
    saved_action_space = data['action_space']

    print("--- MODEL DOSYASINDA KAYITLI BİLGİ BULUNDU ---")
    print(f"Kayıtlı Eylem Uzayı (Action Space): {saved_action_space}")
    print(f"  Low (Alt Sınırlar): {saved_action_space.low}")
    print(f"  High (Üst Sınırlar): {saved_action_space.high}")
    print(f"  Shape (Boyut): {saved_action_space.shape}")
    print("-------------------------------------------------")

except ModuleNotFoundError as e:
    print(f"HATA: ModuleNotFoundError oluştu: {e}")
    print("Model eğitilirken kullanılan bir modül (örn: scripts.utils) bulunamıyor.")
    print("Bu betiği proje kök dizininden (UAV_Navigation_DRL_AirSim) çalıştırdığınızdan emin olun.")
except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}")