import airsim
import time
import numpy as np
import cv2
import msgpackrpc # 'pip install msgpack-rpc-python' gerekebilir

# --- Kodunuzun çalışması için gerekli tanımlamalar ---

# 1. Kodunuzda olduğunu varsaydığım özel hata sınıfı
class CameraError(Exception):
    pass

# 2. Test için 'dynamic_model' ve 'vehicle_name' simülasyonu
class MockDynamicModel:
    """_get_state_feature() fonksiyonunu taklit eder."""
    def _get_state_feature(self):
        # state_feature_length'e uygun sahte bir state vektörü
        return np.zeros(10, dtype=np.float32)

class FPSEnvironment:
    """
    Tüm DÜZELTİLMİŞ fonksiyonlarınızı içeren 
    ve FPS testi için kullanılacak ana test sınıfı.
    """
    def __init__(self):
        # Test ayarları (Kendi .ini dosyanızdaki değerlerle eşleşmeli)
        self.screen_width = 70
        self.screen_height = 200
        self.max_depth_meters = 15.0  # .ini dosyanızdan gelen değer (float)
        self.state_feature_length = 10 # _get_state_feature'a uygun olmalı
        
        # Taklit edilen (Mocked) parçalar
        self.dynamic_model = MockDynamicModel()
        
        # Gerçek AirSim bağlantısı
        print("AirSim'e bağlanılıyor...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        
        # 'vehicle_name' kullanıyorsanız, hata yakalama mantığınız için
        # self.vehicle_name = "Drone1" # (Kendi drone adınız)
        
        print(f"Bağlantı başarılı. Test ayarları:")
        print(f"  Çözünürlük: ({self.screen_width}, {self.screen_height})")
        print(f"  Maks. Derinlik: {self.max_depth_meters}m")

    # --- 1. DÜZELTİLMİŞ FONKSİYONUNUZ ---
    # (DepthPerspective kullanan)
    def get_depth_image(self):
        req = [airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)]
        last_err = None
        for attempt in range(3):
            try:
                responses = self.client.simGetImages(req)
                r = responses[0]

                if r.width == 0 or r.height == 0:
                    raise RuntimeError("empty image: zero width/height")
                if len(r.image_data_float) == 0:
                    raise RuntimeError("empty image_data_float")

                # ========================= HATANIN DÜZELTİLDİĞİ YER =========================
                # 'image_data_float' bir 'list'tir, 'bytes' değil.
                # Bu yüzden 'np.frombuffer' hata verir.
                # Sizin orijinal kodunuzdaki 'list_to_2d_float_array' 
                # bu 'list'i işlemek için DOĞRU fonksiyondur.
                
                # HATALI KOD (Benim hatam):
                # depth_img = np.frombuffer(r.image_data_float, dtype=np.float32)
                # depth_img = depth_img.reshape(r.height, r.width)
                
                # DOĞRU KOD:
                depth_img = airsim.list_to_2d_float_array(
                    r.image_data_float, r.width, r.height
                )
                # ==============================================================================

                # 100 ile çarpmayı kaldırma mantığımız hala geçerli, 
                # çünkü 'DepthPerspective' zaten metre verisi verir.
                return depth_img

            except (msgpackrpc.error.RPCError, RuntimeError) as e:
                last_err = e
                if attempt == 0:
                    try:
                        self.client.confirmConnection()
                        if hasattr(self, "vehicle_name"):
                            try:
                                self.client.enableApiControl(True, self.vehicle_name)
                                self.client.armDisarm(True, self.vehicle_name)
                            except Exception:
                                pass
                    except Exception:
                        pass
                time.sleep(0.05)
        raise CameraError(f"simGetImages failed after retries: {last_err}")

    # --- 2. DÜZELTİLMİŞ FONKSİYONUNUZ ---
    # (Bu fonksiyonda hata yoktu, aynen kalıyor)
    def get_obs_image(self):
        image_meters = self.get_depth_image() 
        image_resize = cv2.resize(image_meters, (self.screen_width,
                                                 self.screen_height))
        
        self.min_distance_to_obstacles = image_resize.min() 

        image_normalized = np.clip(
            image_resize, 0, self.max_depth_meters) / self.max_depth_meters
        
        # Yakın = 1.0, Uzak = 0.0
        image_processed = 1.0 - image_normalized

        state_feature_array = np.zeros((self.screen_height, self.screen_width), 
                                       dtype=np.float32)
        
        state_feature = self.dynamic_model._get_state_feature()
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        image_with_state = np.array([image_processed.astype(np.float32), 
                                     state_feature_array.astype(np.float32)])
        
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)

        return image_with_state

# --- FPS TESTİNİ ÇALIŞTIRMA BÖLÜMÜ ---
# (Bu kısım da hatasızdı, aynen kalıyor)
def run_fps_test():
    try:
        env = FPSEnvironment()
    except Exception as e:
        print(f"HATA: AirSim'e bağlanılamadı. Simülatör çalışıyor mu?")
        print(f"Hata detayı: {e}")
        return

    num_frames_to_test = 300
    frame_count = 0
    
    print(f"\n--- FPS Testi Başlıyor ---")
    print(f"{num_frames_to_test} adet tam 'gözlem' (observation) alınacak...")
    print("Bu test, 'get_obs_image' fonksiyonunun (resize, clip, vb. dahil) tamamını ölçer.")
    
    # Isınma turu
    try:
        _ = env.get_obs_image()
    except Exception as e:
        print(f"Isınma turunda hata: {e}")
        return

    # --- Gerçek Test Başlangıcı ---
    start_time = time.time()

    for i in range(num_frames_to_test):
        try:
            observation = env.get_obs_image()
            frame_count += 1
            
            if (i + 1) % 50 == 0:
                print(f"  {i + 1} / {num_frames_to_test} gözlem alındı...")
                
        except Exception as e:
            print(f"HATA: {i}. karede 'get_obs_image' başarısız oldu: {e}")
            break

    # --- Test Bitişi ---
    end_time = time.time()

    if frame_count > 0:
        total_time = end_time - start_time
        fps = frame_count / total_time
        
        print("\n--- TEST SONUÇLARI ---")
        print(f"Toplam {frame_count} gözlem ('observation') toplandı.")
        print(f"Geçen toplam süre: {total_time:.2f} saniye")
        print(f"DRL Ajanının Alabildiği Pratik Gözlem FPS: {fps:.2f}")
    else:
        print("\nTest başarısız, hiç kare alınamadı.")

if __name__ == "__main__":
    run_fps_test()