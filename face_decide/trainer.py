import torch
import torch.nn as nn
import torch.optim as optim
import os
from face_decide.simpling import FaceDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer():
    def __init__(self, net, net_save_path, data_path, summery_path):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.summery = SummaryWriter(summery_path)
        self.net = net.to(self.device)
        self.net_save_path = net_save_path
        self.data_path = data_path


        self.cls_loss = nn.BCELoss()
        self.offset_loss = nn.MSELoss()

        self.optimizer = optim.Adam(self.net.parameters())

        if os.path.exists(self.net_save_path):
            net.load_state_dict(torch.load(self.net_save_path))
            print("网络加载成功")

        else:
            print("无训练网络")

    def trainer(self, stop_volue):
        facedatas = FaceDataset(self.data_path)
        dataloader = DataLoader(facedatas, batch_size=256, shuffle=True)


        loss = 0
        cls_loss = 0
        offset_loss = 0
        # test_loss = 0
        # test_cls_loss = 0
        # test_offset_loss = 0

        epoches = 0
        while True:
            for i, (img_data_, cls_, offset_) in enumerate(dataloader):
                img_data_ = img_data_.to(self.device)
                cls_ = cls_.to(self.device)
                offset_ = offset_.to(self.device)

                output_cls_, output_offset_ = self.net(img_data_)
                output_cls = output_cls_.view(-1, 1)
                output_offset = output_offset_.view(-1, 200)

                category_mask = torch.lt(cls_, 2)
                category = torch.masked_select(cls_, category_mask)
                output_category = torch.masked_select(output_cls, category_mask)
                cls_loss = self.cls_loss(output_category, category)

                offset_mask = torch.gt(cls_, 0)

                offset = torch.masked_select(offset_, offset_mask)

                output_offset = torch.masked_select(output_offset, offset_mask)

                offset_loss = self.offset_loss(output_offset, offset)

                loss = cls_loss + offset_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                cls_loss = cls_loss.cpu().item()
                offset_loss = offset_loss.cpu().item()
                loss = loss.cpu().item()

                print("epoches:{3}  loss:{0}  cls_loss:{1}  offset_loss:{2}".format(loss, cls_loss, offset_loss,epoches))
            '''保存网络'''
            torch.save(self.net.state_dict(), self.net_save_path)
            print("save success epoches = {}".format(epoches))
            # for i, (test_img_data_, test_cls_, test_offset_) in enumerate(test_dataloader):
            #     test_img_data_ = test_img_data_.to(self.device)
            #     test_cls_ = test_cls_.to(self.device)
            #     test_offset_ = test_offset_.to(self.device)
            #     test_output_cls_, test_output_offset_ = self.net(test_img_data_)
            #     test_output_cls = test_output_cls_.view(-1, 1)
            #     test_output_offset = test_output_offset_.view(-1, 14)
            #     test_category_mask = torch.lt(test_cls_, 2)
            #     test_category = torch.masked_select(test_cls_, test_category_mask)
            #     test_output_category = torch.masked_select(test_output_cls, test_category_mask)
            #     test_cls_loss = self.cls_loss(test_output_category, test_category)
            #     test_offset_mask = torch.gt(test_cls_, 0)
            #     test_offset = torch.masked_select(test_offset_, test_offset_mask)
            #     test_output_offset = torch.masked_select(test_output_offset, test_offset_mask)
            #     test_offset_loss = self.offset_loss(test_output_offset, test_offset)
            #     test_loss = test_cls_loss + test_offset_loss
            # self.summery.add_scalar("cls_loss", {"train_loss": cls_loss}, epoches)
            # self.summery.add_scalar("offset_loss", {"train_loss": offset_loss}, epoches)
            # self.summery.add_scalar("loss", {"train_loss": loss}, epoches)
            epoches += 1

            if loss < stop_volue:
                break
