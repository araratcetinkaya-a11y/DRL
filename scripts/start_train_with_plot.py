# --- make repo root importable no matter how we run this file ---
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# (debug - bir kez bakıp sonra silebilirsin)
# print("DEBUG start file:", __file__, "\nDEBUG sys.path[0]:", sys.path[0])

import argparse
from PyQt5 import QtWidgets
from configparser import ConfigParser

# İKİ MODDA DA ÇALIŞAN import (önce paketli, olmazsa fallback)
try:
    from scripts.utils.thread_train import TrainingThread
    from scripts.utils.ui_train import TrainingUi
except ImportError:
    from utils.thread_train import TrainingThread
    from utils.ui_train import TrainingUi

import sys
import argparse
import os 
from PyQt5 import QtWidgets

# from evaluate_td3 import evaluate
from scripts.utils.thread_train import TrainingThread
# from utils.thread_train_fixedwing import TrainingThread
from scripts.utils.ui_train import TrainingUi
from configparser import ConfigParser


def get_parser():
    parser = argparse.ArgumentParser(
        description="Training navigation model using TD3")
    parser.add_argument('-config', required=True,
                        help='config file name, such as config0925.ini', default='config_default.ini')
    parser.add_argument('-objective', required=True, help='training objective')

    return parser
    

def main():
    # select your config file here
    # config_file = 'configs/config_SimpleAvoid_SimpleMultirotor.ini'
    # config_file = 'configs/config_fixedwing.ini'
    config_file = 'configs/config_Maze_SimpleMultirotor_2D.ini'

    # 1. Create the qt thread
    app = QtWidgets.QApplication(sys.argv)
    #print("CONFIG PATH (raw):", config_file)
    #print("CONFIG PATH (abs):", os.path.abspath(config_file))
    #print("EXISTS?", os.path.exists(config_file))
    gui = TrainingUi(config_file)
    gui.show()

    # 2. Start training thread
    training_thread = TrainingThread(config_file)

    training_thread.env.action_signal.connect(gui.action_cb)
    training_thread.env.state_signal.connect(gui.state_cb)
    training_thread.env.attitude_signal.connect(gui.attitude_plot_cb)
    training_thread.env.reward_signal.connect(gui.reward_plot_cb)
    training_thread.env.pose_signal.connect(gui.traj_plot_cb)

    cfg = ConfigParser()
    cfg.read(config_file)

    training_thread.start()

    sys.exit(app.exec_())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
