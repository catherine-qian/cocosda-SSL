import numpy as np
import torch
from torch.autograd import Variable
import hdf5storage
import argparse
from torch.utils.data import DataLoader
import sys
import time
import random
import torch.nn as nn
from torch.utils.data import Dataset

sys.path.append('modelclass')
sys.path.append('funcs')

# random seed
SEED = 7
torch.manual_seed(SEED)  # For reproducibility across different computers
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True


def angular_distance_compute(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)


def ACCcompute(MAE, th):
    acc = sum([error <= th for error in MAE]) / len(MAE)
    return acc


class MyDataloaderClass(Dataset):

    def __init__(self, X_data, label1, label2):
        self.x_data = X_data
        self.label1 = label1  # posterior
        self.label2 = label2  # posterior

        self.len = X_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.label1[index], self.label2[index]

    def __len__(self):
        return self.len


def minmax_scaletensor(x, faxis):
    xmin = x.min(faxis, keepdim=True)[0]
    xmax = x.max(faxis, keepdim=True)[0]
    xnorm = (x - xmin) / (xmax - xmin)
    return xnorm


def minmax_norm2d(data_in, faxis):
    # data_in: (270, 27, 306)

    dmin = data_in.min(axis=faxis, keepdims=True)
    dmax = data_in.max(axis=faxis, keepdims=True)
    data_out = ((data_in - dmin) / (dmax - dmin))
    data_out[data_in == 0] = 0.0
    return data_out


def ACCevent(evepred, evegt):
    label = np.argmax(evepred, axis=1)
    acc = sum(label == evegt) / label.shape[0]

    return acc


def MAEeval(Y_pred_t, Yte):
    # ------------ error evaluate   ----------
    erI1, erI2 = [], []

    DoA = []
    for i in range(Yte.shape[0]):  # time
        hyp = Y_pred_t[i]  # our estimate

        gt = Yte[i]
        pred = np.argmax(hyp)  # predict
        ang = angular_distance_compute(gt, pred)[0]
        erI1.append(ang)  # error
        DoA.append(pred)  # doa

    MAE1 = sum(erI1) / len(erI1)
    ACC1 = ACCcompute(erI1, 5)
    # print("Testing MAE:%.8f \t ACC: %.8f " % (MAE1, ACC1))
    return MAE1, ACC1


#######################  Model ################

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 618 dim: (306+ 312)=(6*51+8*39): gbcphat + mfcc
        # input feature: bs*time*618
        self.time = 27
        self.MLP3 = nn.Sequential(
            nn.Linear(618, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )

        self.DoALayer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 360, bias=True),
        )

        self.eveLayer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(1024, 10, bias=True),
            nn.Sigmoid()
        )

        self.pool = nn.MaxPool1d(kernel_size=self.time)

    def forward(self, x):
        bs, t, dim = x.shape
        input = x.reshape(-1, x.shape[-1])

        x1 = self.MLP3(input)  # ([bs, 27, 618])
        x1 = x1.reshape(bs, t, -1)
        x2 = self.pool(x1.transpose(1, 2)).transpose(1, 2).squeeze()  # max pooling

        DoApred = self.DoALayer(x2)  # localization
        evepred = self.eveLayer(x2)  # event classification

        return DoApred, evepred


################################################################
#######################     Main    ############################


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xinyuan experiments')
    parser.add_argument('-gpuidx', metavar='gpuidx', type=int, default=0, help='gpu number')
    parser.add_argument('-epoch', metavar='epoch', type=int, default=50)  # Options: gcc, melgcc
    parser.add_argument('-batch', metavar='batch', type=int, default=2 ** 5)
    parser.add_argument('-lr', metavar='lr', type=float, default=0.001)
    parser.add_argument('-wts0', metavar='wts', type=float, default=1)
    parser.add_argument('-model', metavar='model', type=str, default='None')

    parser.add_argument('-input', metavar='input', type=str, default="small")  #

    args = parser.parse_args()

BATCH_SIZE = args.batch
print("experiments - xinyuan", flush=True)

device = torch.device("cuda:{}".format(args.gpuidx) if torch.cuda.is_available() else 'cpu')
args.device = device
print(device, flush=True)

criterion = torch.nn.MSELoss(reduction='mean')  # loss
criterion2 = torch.nn.CrossEntropyLoss()  # classification loss
wts0 = args.wts0
wts1 = 1 - wts0
print(args, flush=True)
print('localization wts = ' + str(wts0) + '  event wts' + str(wts1))


def training(epoch):
    model.train()

    print("start training epoch " + str(epoch))

    for batch_idx, (data, DoAgt, evegt) in enumerate(train_loader, 0):

        inputs, DoAgt = Variable(data).type(torch.FloatTensor).to(device), Variable(DoAgt).type(torch.FloatTensor).to(
            device)  # DoA
        evegt = Variable(evegt).type(torch.FloatTensor).to(device).squeeze()  # sound class

        # start training -  
        DoApred, evepred = model.forward(inputs)  # return the predicted angle
        loss = criterion(DoApred.double(), DoAgt.double())  # MSE loss - DoA
        loss_event = criterion2(evepred.double(), evegt.long())  # CE loss - event label

        loss = wts0 * loss + wts1 * loss_event

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if (round(train_loader.__len__() / 5 / 100) * 100) > 0 and batch_idx % (
                round(train_loader.__len__() / 5 / 100) * 100) == 0:
            print("training - epoch%d-batch%d: loss=%.3f" % (epoch, batch_idx, loss.data.item()), flush=True)

    torch.cuda.empty_cache()


def testing(Xte, Yte, evegt):  # Xte: feature, Yte: binary flag
    model.eval()

    print('start testing' + '  ' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    Y_pred_t = []
    evepred = np.zeros((len(Xte), 10))

    for ist in range(0, len(Xte), BATCH_SIZE):
        ied = np.min([ist + BATCH_SIZE, len(Xte)])
        inputs = Variable(torch.from_numpy(Xte[ist:ied])).type(torch.FloatTensor).to(device)

        DoApred, eve = model.forward(inputs)
        evepred[ist:ied] = eve.cpu().detach().numpy()
        if device.type == 'cpu':
            Y_pred_t.extend(DoApred.detach().numpy())  # in CPU
        else:
            Y_pred_t.extend(DoApred.cpu().detach().numpy())  # in CPU

    # ------------ error evaluate   ----------

    MAE, ACC = MAEeval(Y_pred_t, Yte.astype('float32'))  # doa evaluation
    ACC2 = ACCevent(evepred, evegt.squeeze().astype('int64'))

    # event classification evaluation
    torch.cuda.empty_cache()
    return MAE, ACC, ACC2


# ############################# load the data and the model ##############################################################
modelname = args.model
lossname = 'MSE'
print(args, flush=True)

if args.input == "small":
    data = hdf5storage.loadmat('feat618dim.mat')
    print("use small debug set", flush=True)
else:
    data = hdf5storage.loadmat('featall.mat')
    print("use all set", flush=True)

# random
L = len(data['class'])
ridx = random.sample(range(0, L), L)  # random index

event = data['class'][ridx, :] - 1  # event label-start from 0
doa = data['doa'][ridx, :]  # doa label
doafeat360 = data['doafeat360'][ridx, :]  # doa posterior
feat = data['feat'][ridx, :]  # feature

# feature normalization
feat1, feat2 = feat[:, :, :306], feat[:, :, 306:]  # gccphat, mfcc
feat1, feat2 = minmax_norm2d(feat1, faxis=2), minmax_norm2d(feat2,
                                                            faxis=2)  # attention: gccphat feature is not in channel order
feat = np.concatenate((feat1, feat2), axis=2)

# split train/test set
ratio = round(0.7 * L)
eventtr, eventte = event[:ratio, :], event[ratio:, :]
doatr, doate = doa[:ratio, :], doa[ratio:, :]
doafeat360tr, doafeat360te = doafeat360[:ratio, :], doafeat360[ratio:, :]
feattr, featte = feat[:ratio, :], feat[ratio:, :]

train_loader_obj = MyDataloaderClass(feattr, doafeat360tr, eventtr)
train_loader = DataLoader(dataset=train_loader_obj, batch_size=args.batch, shuffle=True, num_workers=4)

model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)

print(model, flush=True)

######## Training + Testing #######

EP = args.epoch
MAE, ACC, ACC2 = np.zeros(EP), np.zeros(EP), np.zeros(EP)

for ep in range(EP):
    training(ep)

    MAE[ep], ACC[ep], ACC2[ep] = testing(featte, doate, eventte)  # loudspeaker
    print("Testing ep%2d:    MAE:%.2f \t ACC: %.2f | ACC: %.2f " % (ep, MAE[ep], ACC[ep] * 100, ACC2[ep] * 100),
          flush=True)

print("finish all!", flush=True)
