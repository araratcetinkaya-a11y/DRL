# scripts/start_train_with_plot.py  — HEADLESS sürüm
import sys
import argparse

# Not: import yolu repoyu nasıl çalıştırdığınıza göre değişebilir.
# Dosyayı "scripts" klasöründen python ile çalıştırıyorsanız:
from utils.thread_train import TrainThread
# Eğer modül olarak çağırıyorsanız (python -m scripts.start_train_with_plot):
# from scripts.utils.thread_train import TrainThread

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/MR_NH.ini")
    args = parser.parse_args()

    t = TrainThread(args.config)
    t.start()

    # TrainThread 'threading.Thread' tabanlıysa join(), QThread ise wait() çalışır:
    try:
        t.join()
    except AttributeError:
        t.wait()

if __name__ == "__main__":
    main()
