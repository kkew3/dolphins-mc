import pdb
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as trans

import vmdata
import more_trans


class FoveatPooling(nn.Module):
    def __init__(self, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, stride=stride, groups=3)
        _0, _1 = -1/16, 1/2
        w = torch.tensor([[_0,_0,_0],[_0,_1,_0],[_0,_0,_0]])
        w = w.unsqueeze(0).repeat(3, 1, 1).view(3, 1, 3, 3)
        self.conv.weight.requires_grad = False
        self.conv.weight.data.copy_(w)
        self.conv.bias.requires_grad = False
        self.conv.bias.data.zero_()
        self.relu = nn.ReLU()

    def forward(self, x):
        cs = self.conv(x)
        con, coff = self.relu(cs), self.relu(-cs)
        return torch.cat((con, coff), dim=1)

class Identity(object):
    def __init__(self, name):
        self.name = name
    def __call__(self, x):
        try:
            shape = x.shape
        except:
            shape = None
        pdb.set_trace()
        return x

root = vmdata.prepare_dataset_root(9, (8, 0, 0))
transform = trans.Compose([
        trans.ToTensor(),
        #trans.Lambda(lambda x: x.max(dim=0, keepdim=True)[0]),
        trans.Normalize(mean=(0.5,), std=(1.0,)),
])
if __name__ == '__main__':
    pool = FoveatPooling(stride=3)
    with vmdata.VideoDataset(root, transform=transform) as vdset:
        for i in range(len(vdset)):
            frame = vdset[i].unsqueeze(0)
            frame = F.interpolate(frame, size=(240, 351))
            frame = pool(frame)
            frame_up, frame_down = frame[:,:3], frame[:,3:]
            frame = frame_up + frame_down
            frame = frame.numpy()
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            frame = more_trans.chw2hwc(frame[0])
            cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(300) & 0xFF == ord('q'):
                break
