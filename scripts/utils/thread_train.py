from .custom_policy_sb3 import CNN_FC, CNN_GAP, CNN_GAP_BN, No_CNN, CNN_MobileNet, CNN_GAP_new
import datetime
import gym
import gym_env
import numpy as np
from stable_baselines3 import TD3, PPO, SAC

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
import itertools
from wandb.integration.sb3 import WandbCallback
import wandb
from PyQt5 import QtCore
import argparse
import ast
from configparser import ConfigParser
import torch as th
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))





def with_permute(extractor_class):
    """
    Bu fonksiyon, SB3'ün 'features_extractor_class' girdisi olarak
    verilen bir sınıfı alır ve o sınıfı 'sarmalar'.
    Oluşturulan yeni sınıf, 'forward' metodu çağrıldığında
    otomatik olarak (H, W, C) -> (C, H, W) eksen değişimini
    (permute) yapar ve sonucu orijinal sınıfın 'forward'
    metoduna gönderir.
    
    Bu, VecTransposeImage'in float32 veriler için yaptığı işin aynısıdır.
    """
    class PermutedExtractor(extractor_class):
        def forward(self, observations: th.Tensor) -> th.Tensor:
            # Gözlem (N, H, W, C) formatında gelir
            # Biz onu (N, C, H, W) formatına çeviririz
            # --- EKLENEN KONTROL PRINT'LERİ ---
            if not hasattr(self, '_printed_shapes'):
                print(f"\n[WRAPPER KONTROLÜ] GİRİŞ shape: {observations.shape}")
                observations_permuted = observations.permute(0, 3, 1, 2)
                print(f"[WRAPPER KONTROLÜ] ÇIKIŞ shape (permute sonrası): {observations_permuted.shape}\n")
                self._printed_shapes = True
            else:
                observations_permuted = observations.permute(0, 3, 1, 2)
            # --- KONTROL SONU ---
            
            # (N, C, H, W) formatındaki tensörü,
            # orijinal sınıfın (örn: CNN_GAP_new) 'forward' metoduna yollarız
            return super().forward(observations_permuted)
            
    return PermutedExtractor

class SeedCycler(gym.Wrapper):
    def __init__(self, env, seeds):
        super().__init__(env)
        self._seeds = itertools.cycle(seeds)

    def reset(self):
        # gym==0.19: seed, reset'e parametre olarak verilmez;
        # env.seed(...) + env.reset() kullanılır
        seed = next(self._seeds)
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        if hasattr(self.env.action_space, "seed"):
            self.env.action_space.seed(seed)
        return self.env.reset()

def make_eval_env(cfg):
    # 0) EVAL log klasörü
    eval_log_dir = cfg.get('EVAL','eval_log_dir') if cfg.has_option('EVAL','eval_log_dir') else './logs_eval'
    os.makedirs(eval_log_dir, exist_ok=True)

    # 1) Held-out seed listesi
    if cfg.has_option('EVAL', 'eval_seeds'):
        seeds = [int(s) for s in cfg.get('EVAL', 'eval_seeds').split(',')]
    else:
        seeds = [101,202,303,404,505,606,707,808,909,1001]

    # 2) Tekil env fabrikası
    def _make_base():
        env = gym.make('airsim-env-v0')
        env.set_config(cfg)
        # SeedCycler ile her reset'te farklı seed kullan
        env = SeedCycler(env, seeds)
        
        # --- DÜZELTME BURADA ---
        # VecMonitor yerine, Monitor'ü buraya, tekil ortam seviyesine koyuyoruz.
        # Log dosyasını buraya veriyoruz.
        log_file = os.path.join(eval_log_dir, 'eval_monitor.csv')
        env = Monitor(env, filename=log_file)
        return env

    # 3) Vektörleştir (train ile aynı tip olsun: DummyVecEnv)
    venv = DummyVecEnv([_make_base])

    # 4) VecMonitor SATIRI SİLİNDİ veya YORUMA ALINDI
    # venv = VecMonitor(venv, filename=os.path.join(eval_log_dir, 'monitor_eval.csv'))

    print("eval obs space:", venv.observation_space)

    return venv



def get_parser():
    parser = argparse.ArgumentParser(
        description="Training thread without plot")
    parser.add_argument(
        '-c',
        '--config',
        help='config file name in configs folder, such as config_default',
        default='config_Trees_SimpleMultirotor_2D')
    parser.add_argument('-n',
                        '--note',
                        help='training objective',
                        default='depth_upper_split_5')

    return parser


class TrainingThread(QtCore.QThread):
    """
    QT thread for policy training
    """

    def __init__(self, config):
        super(TrainingThread, self).__init__()
        print("init training thread")

        # config
        self.cfg = ConfigParser()
        self.cfg.read(config)

        env_name = self.cfg.get('options', 'env_name')
        self.project_name = env_name

        # make gym environment
        #self.env = gym.make('airsim-env-v0')
        #self.env.set_config(self.cfg)

        self.env = DummyVecEnv([self.make_train_env])
        #self.env = VecTransposeImage(self.env)

        wandb_name = self.cfg.get(
            'options', 'policy_name') + '-' + self.cfg.get('options', 'algo')
        if self.cfg.get('options', 'dynamic_name') == 'SimpleFixedwing':
            if self.cfg.get('options', 'perception') == "lgmd":
                wandb_name += '-LGMD'
            else:
                wandb_name += '-depth'
            if self.cfg.getfloat('fixedwing', 'pitch_flap_hz') != 0:
                wandb_name += '-Flapping'

        # wandb
        if self.cfg.getboolean('options', 'use_wandb'):
            wandb.init(
                project=self.project_name,
                notes=self.cfg.get('wandb', 'notes'),
                name=self.cfg.get('wandb', 'name'),
                sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
                save_code=True,  # optional
            )

    def make_train_env(self):
        env = gym.make('airsim-env-v0')
        env.set_config(self.cfg)
        import inspect
        gercek_dosya_yolu = inspect.getfile(env.__class__)
        
        print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("GYM.MAKE TARAFINDAN KULLANILAN GERÇEK SINIFIN DOSYA YOLU:")
        print(gercek_dosya_yolu)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n")
        env = Monitor(env)
        return env
    
    def terminate(self):
        print('TrainingThread terminated')

    def run(self):
        print("run training thread")
        
        # ! -----------------------------------init folders-----------------------------------------
        now = datetime.datetime.now()
        # Koşu etiketi: YYYY_MM_DD_HH_MM_<dynamic>_<policy>_<algo>
        run_tag = "{}_{}_{}_{}".format(
            now.strftime('%Y_%m_%d_%H_%M'),
            self.cfg.get('options', 'dynamic_name'),
            self.cfg.get('options', 'policy_name'),
            self.cfg.get('options', 'algo'),
        )

        # Kökler (override istersen INI'ye [SAVE] final_root, [EVAL] best_root ekleyebilirsin)
        final_root = self.cfg.get('SAVE', 'final_root') if self.cfg.has_option('SAVE', 'final_root') else 'logs'
        best_root  = self.cfg.get('EVAL', 'best_root')  if self.cfg.has_option('EVAL', 'best_root')  else 'best_eval'

        # Final model ağacı: logs/<project>/<run_tag>/...
        file_path   = os.path.join(final_root, self.project_name, run_tag)
        log_path    = os.path.join(file_path, 'tb_logs')
        model_path  = os.path.join(file_path, 'models')
        config_path = os.path.join(file_path, 'config')
        data_path   = os.path.join(file_path, 'data')

        os.makedirs(log_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(config_path, exist_ok=True)
        os.makedirs(data_path, exist_ok=True)  # create data path to save q_map

        # Best (eval) ağacı: best_eval/<project>/<run_tag>/
        best_save_dir = os.path.join(best_root, self.project_name, run_tag)
        # Nadiren aynı dakikada iki koşu açılırsa sondan numaralandır
        suffix = 2
        while os.path.exists(best_save_dir):
            best_save_dir = os.path.join(best_root, self.project_name, f"{run_tag}_{suffix}")
            suffix += 1
        os.makedirs(best_save_dir, exist_ok=True)

        # Eval log’ları: best_eval/.../eval_logs
        eval_log_dir = os.path.join(best_save_dir, 'eval_logs')
        os.makedirs(eval_log_dir, exist_ok=True)

        # make_eval_env() bu klasörü kullansın diye cfg'ye yaz
        if not self.cfg.has_section('EVAL'):
            self.cfg.add_section('EVAL')
        self.cfg.set('EVAL', 'eval_log_dir', eval_log_dir)

        # config.ini güvenli biçimde yaz
        with open(os.path.join(config_path, 'config.ini'), 'w') as configfile:
            self.cfg.write(configfile)

        #! -----------------------------------policy selection-------------------------------------
        #feature_num_state = self.env.dynamic_model.state_feature_length
        feature_num_state = self.env.envs[0].dynamic_model.state_feature_length
        feature_num_cnn = self.cfg.getint('options', 'cnn_feature_num')
        policy_name = self.cfg.get('options', 'policy_name')

        # feature extraction network
        if self.cfg.get('options', 'activation_function') == 'tanh':
            activation_function = th.nn.Tanh
        else:
            activation_function = th.nn.ReLU
        
        if policy_name == 'mlp':
            policy_base = 'MlpPolicy'
            policy_kwargs = dict(activation_fn=activation_function)
        else:
            policy_base = 'CnnPolicy'
            if policy_name == 'CNN_FC':
                policy_used_base = CNN_FC
            elif policy_name == 'CNN_GAP':
                policy_used_base = CNN_GAP_new
            elif policy_name == 'CNN_GAP_BN':
                policy_used_base = CNN_GAP_BN
            elif policy_name == 'CNN_MobileNet':
                policy_used_base = CNN_MobileNet
            elif policy_name == 'No_CNN':
                policy_used_base = No_CNN
            else:
                raise Exception('policy select error: ', policy_name)


            policy_used = with_permute(policy_used_base)
            
            policy_kwargs = dict(
                features_extractor_class=policy_used,
                features_extractor_kwargs=dict(
                    features_dim=feature_num_state + feature_num_cnn,
                    state_feature_dim=feature_num_state),
                activation_fn=activation_function,
                    
                # Adam optimizer betalarını burada da ekleyebilirsin
               # optimizer_class=th.optim.Adam,
               # optimizer_kwargs=dict(
                #    betas=(0.9, 0.3) )
                      
                
                )

        # fully-connected work after feature extraction
        net_arch_list = ast.literal_eval(self.cfg.get("options", "net_arch"))
        policy_kwargs['net_arch'] = net_arch_list

        #! ---------------------------------algorithm selection-------------------------------------
        algo = self.cfg.get('options', 'algo')
        print('algo: ', algo)
        if algo == 'PPO':
            model = PPO(
                policy_base,
                self.env,
                # n_steps = 200,
                learning_rate=self.cfg.getfloat('DRL', 'learning_rate'),
                policy_kwargs=policy_kwargs,
                tensorboard_log=log_path,
                seed=0,
                verbose=2)
        elif algo == 'SAC':
            n_actions = self.env.action_space.shape[-1]
            noise_sigma = self.cfg.getfloat(
                'DRL', 'action_noise_sigma') * np.ones(n_actions)
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=noise_sigma)
            model = SAC(
                policy_base,
                self.env,
                replay_buffer_kwargs={"handle_timeout_termination": True},
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                buffer_size=self.cfg.getint('DRL', 'buffer_size'),
                gamma=self.cfg.getfloat('DRL', 'gamma'),
                learning_starts=self.cfg.getint('DRL', 'learning_starts'),
                learning_rate=self.cfg.getfloat('DRL', 'learning_rate'),
                batch_size=self.cfg.getint('DRL', 'batch_size'),
                train_freq=(self.cfg.getint('DRL', 'train_freq'), 'step'),
                gradient_steps=self.cfg.getint('DRL', 'gradient_steps'),
                tensorboard_log=log_path,
                #tau =0.002,
                #ent_coef_lr=1e-4,
                #target_entropy=-1,
                ent_coef="auto",
                seed=0,
                verbose=0)

            print("Actor betas:", model.actor.optimizer.defaults['betas'])
            print("Critic betas:", model.critic.optimizer.defaults['betas'])
            print("train obs space:", self.env.observation_space) 

        elif algo == 'TD3':
            # The noise objects for TD3
            n_actions = self.env.action_space.shape[-1]
            noise_sigma = self.cfg.getfloat(
                'DRL', 'action_noise_sigma') * np.ones(n_actions)
            action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                             sigma=noise_sigma)
            model = TD3(
                policy_base,
                self.env,
                action_noise=action_noise,
                learning_rate=self.cfg.getfloat('DRL', 'learning_rate'),
                gamma=self.cfg.getfloat('DRL', 'gamma'),
                policy_kwargs=policy_kwargs,
                learning_starts=self.cfg.getint('DRL', 'learning_starts'),
                batch_size=self.cfg.getint('DRL', 'batch_size'),
                train_freq=(self.cfg.getint('DRL', 'train_freq'), 'step'),
                gradient_steps=self.cfg.getint('DRL', 'gradient_steps'),
                buffer_size=self.cfg.getint('DRL', 'buffer_size'),
                tensorboard_log=log_path,
                seed=0,
                verbose=2)
        else:
            raise Exception('Invalid algo name : ', algo)

        

        if self.cfg.has_option('EVAL','n_eval_episodes'):
            n_eval_episodes = self.cfg.getint('EVAL','n_eval_episodes')
        else:
            n_eval_episodes = 10

        if self.cfg.has_option('EVAL','eval_freq'):
            eval_freq = self.cfg.getint('EVAL','eval_freq')
        else:
            eval_freq = 10000

        if self.cfg.has_option('EVAL','deterministic_eval'):
            deterministic = self.cfg.getboolean('EVAL','deterministic_eval')
        else:
            deterministic = True  # SAC için önerilir

        # Eval env (train ile aynı kurulum + seed devri + Monitor)
        eval_env = make_eval_env(self.cfg)

        # EvalCallback: her eval_freq adımda n_eval_episodes bölüm değerlendir, en iyiyi kaydet
        eval_cb = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        best_model_save_path=best_save_dir,  # <--- yeni konum
        log_path=eval_log_dir,               # <--- yeni log klasörü
        deterministic=deterministic,
        render=False,
    )

        # TODO create eval_callback
        # eval_freq = self.cfg.getint('TD3', 'eval_freq')
        # n_eval_episodes = self.cfg.getint('TD3', 'n_eval_episodes')
        # eval_callback = EvalCallback(self.env, best_model_save_path= file_path + '/eval',
        #                      log_path= file_path + '/eval', eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        #                      deterministic=True, render=False)

        #! -------------------------------------train-----------------------------------------
        

        print('start training model')
        total_timesteps = self.cfg.getint('options', 'total_timesteps')
        self.env.model = model
        self.env.data_path = data_path

        if self.cfg.getboolean('options', 'use_wandb'):
            # if algo == 'TD3' or algo == 'SAC':
            #     wandb.watch(model.actor, log_freq=100, log="all")  # log gradients
            # elif algo == 'PPO':
            #     wandb.watch(model.policy, log_freq=100, log="all")
            combo_cb = CallbackList([
                eval_cb,
                WandbCallback(
                    model_save_freq=10000,
                    gradient_save_freq=5000,
                    model_save_path=model_path,
                    verbose=2,
                )
            ])
            model.learn(
                total_timesteps,
                log_interval=1,
                callback=combo_cb
            )
        else:
            combo_cb = CallbackList([eval_cb])
            model.learn(total_timesteps, callback=combo_cb)

        #! ---------------------------model save----------------------------------------------------
        # Final model: logs/<project>/<run_tag>/models/final_model.zip
        model.save(os.path.join(model_path, 'final_model.zip'))

        print('training finished')
        print('model saved to: {}'.format(model_path))


def main():

        parser = get_parser()
        args = parser.parse_args()

        config_file = 'configs/' + args.config + '.ini'

        print(config_file)

        training_thread = TrainingThread(config_file)
        
        # ======== YENİ DOĞRULAMA KODU BURADA ========
        # 'training_thread.env' bir VecEnv olduğu için, 
        # 'check_env' direkt çalışmayabilir. Bu yüzden 'unwrapped'
        # edilmiş tekil env'i kontrol etmek daha güvenlidir.
        # Yeni bir env oluşturup onu kontrol edelim.
        
        print("\n--- Stable Baselines Env Checker Başlatılıyor ---")
        print("Gym ortamının (airsim-env-v0) tüm DRL kurallarına uyup uymadığı test ediliyor...")
        
        # 'make_train_env' içindeki gibi yeni bir env oluştur
        # (Monitor ile sarmalamadan, ham halini)
        try:
            check_env_instance = gym.make('airsim-env-v0')
            check_env_instance.set_config(training_thread.cfg)
            
            # Testi çalıştır
            check_env(check_env_instance)
            
            print("--- Env Checker BAŞARILI! ---")
            print("Observation space (float32, [0,1]) ve veri tipi EŞLEŞİYOR.")
            print("Ortam (Environment) DRL eğitimi için doğru yapılandırılmış.\n")
        
        except Exception as e:
            print("\n--- !!! ENV CHECKER BAŞARISIZ !!! ---")
            print("Ortamınız DRL standartlarını karşılamıyor. Hata:")
            print(e)
            print("Eğitime başlamadan önce bu hatayı düzeltmelisiniz.")
            return # Hata varsa eğitime başlama
        # =================================================

        training_thread.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
