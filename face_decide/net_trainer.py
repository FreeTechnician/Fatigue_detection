from face_decide.net import MainNet
from face_decide.trainer import *
import os
if __name__ == '__main__':
    net = MainNet()
    if not os.path.exists(r"param"):
        os.makedirs(r"param")
    trainer = Trainer(net,r"param\net.pth",r"D:\data\wflw_data_0", summery_path=r"tensor/net_logs")
    trainer.trainer(0.00001)