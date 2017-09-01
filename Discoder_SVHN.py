
from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets

import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm
import scipy.misc


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# to prevent opencv from initializing CUDA in workers
#torch.randn(8).cuda()
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape((nrows, ncols, height, width, intensity))
              .swapaxes(1,2)
              .reshape((height*nrows, width*ncols, intensity)))
    return result

def save_samples():
    img_bhwc = netG(noise).data.cpu().add_(1).mul_(.5)
    img_bhwc = img_bhwc.permute(0,2,3,1).numpy()
    array = img_bhwc.copy()
    result = gallery(array,10)*.5+.5
    scipy.misc.imsave('svhn.jpg', result)




count = 100
learning_rate = .0003
batch_size = 100
unlabeled_weight = 1
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)



#Loading labelled and unlabelled data
import svhn_data
def rescale(mat):
    return np.transpose(((-127.5 + mat)/127.5),(3,2,0,1))

trainx, trainy = svhn_data.load('./svhn','train')
testx, testy = svhn_data.load('./svhn','test')
trainx = rescale(trainx)
testx = rescale(testx)
trainx_unl = trainx.copy()
trainx_unl2 = trainx.copy()
nr_batches_train = int(trainx.shape[0]/batch_size)
nr_batches_test = int(np.ceil(float(testx.shape[0])/batch_size))
trainx = torch.from_numpy(trainx).float()
trainy = torch.from_numpy(trainy)
testx = torch.from_numpy(testx).float()
testy = torch.from_numpy(testy)
trainx_unl = torch.from_numpy(trainx_unl).float()
trainx_unl2 = torch.from_numpy(trainx_unl2).float()

#Prior distribution on the latent
prior = torch.zeros(10).cuda()
for i in range(0,10):
    prior[i] = (trainy==i).sum()/trainy.size(0)
prior = prior.unsqueeze(0)

#Definition of the generator
noise_dim = (batch_size, 100)

class _netG(nn.Module):
    def __init__(self):
        super(_netG, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 32 x 32
        )

    def forward(self, input):
        output = self.main(input)
        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear')!=-1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    
    
netG = _netG()
netG.cuda()
netG.apply(weights_init);





#Definition of the encoding network
#The architecture is exactly the same as that of Improved GAN
#A better architecture will result in better performance
ins = [3,64,64,64,128,128,128,128,128]
outs = [64,64,64,128,128,128,128,128,128]
filters = [3,3,3,3,3,3,3,1,1]
strides = [1,1,2,1,1,2,1,1,1]
pads = [1,1,1,1,1,1,0,0,0]
  

class _netD(nn.Module):
    def __init__(self):
        super(_netD, self).__init__()

        self.conv = [None]*9
        for i in range(0,9):
            self.conv[i] = nn.Conv2d(ins[i],outs[i],filters[i],strides[i],pads[i])
            self.conv[i].weight.data.normal_(0.0, 0.05)
            self.conv[i].bias.data.fill_(0)
            self.conv[i] = weight_norm(self.conv[i])
        self.conv = nn.ModuleList(self.conv)
        self.linear = nn.Linear(128,10)
        self.linear.weight.data.normal_(0.0, 0.05)
        self.linear.bias.data.fill_(0)
        self.linear = weight_norm(self.linear)
        
        self.drop1 = nn.Dropout(.2)              
        self.drop2 = nn.Dropout(.5)      
        self.drop3 = nn.Dropout(.5)

        self.global_pool = nn.AvgPool2d(6)
        
        
        
    def forward(self, x):
        x = self.drop1(x)
        for i in range(0,3):
            x = self.conv[i](x)
            x = F.leaky_relu(x, negative_slope=.2)

        x = self.drop2(x)
        for i in range(3,6):
            x = self.conv[i](x)
            x = F.leaky_relu(x, negative_slope=.2)

        x = self.drop3(x)
        for i in range(6,9):
            x = self.conv[i](x)
            x = F.leaky_relu(x, negative_slope=.2)
        
        features = self.global_pool(x).view(-1,128)
        x = self.linear(features)

        return x, features

netD = _netD()
netD.cuda()


#Setting up the optimizer and loss function
optimizerC = optim.Adam(netD.parameters(), lr=learning_rate, betas=(.5, .999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(.5, .999))
noise = Variable(torch.randn(batch_size,100,1,1).cuda())
loss_fn = nn.CrossEntropyLoss()
loss_fn.cuda()
#For storing the network output of the last 3000 samples
phi_all = torch.rand(3000,10).cuda()
phi_all /= phi_all.sum(1).unsqueeze(1).expand_as(phi_all)
classMat = torch.eye(10).cuda()



#Initializing the weights of the encoding network to ensure 0 mean and unit variance
def initialize(x_unl):
    global avg, saved_g
    netD.eval()
    x = Variable(x_unl.cuda())
    for i in range(0,9):
        x_new = netD.conv[i](x).data
        m = x_new.mean(3).mean(2).mean(0)
        inv_stdv = 1/(x_new**2).mean(3).mean(2).mean(0).sqrt().view(-1,1,1,1)
        netD.conv[i].weight_g.data.copy_(netD.conv[i].weight_g.data*inv_stdv)
        netD.conv[i].bias.data.copy_(-m*inv_stdv.squeeze())
        x = netD.conv[i](x)
        x = F.leaky_relu(x, negative_slope=.2)

    x = netD.global_pool(x).view(-1,128)        
    x_new = netD.linear(x).data
    m = x_new.mean(0).squeeze()
    inv_stdv = .1/(x_new**2).mean(0).squeeze().sqrt().view(-1,1)
    netD.linear.weight_g.data.copy_(netD.linear.weight_g.data*inv_stdv)
    netD.linear.bias.data.copy_(-m*inv_stdv.squeeze())
    
    avg = [None]*30
    i = 0
    for param in netD.parameters():
        avg[i] = param.data.clone()
        i += 1

    saved_g = [None]*9
    for i in range(0,9):
        saved_g[i] = netD.conv[i].weight_g.data.clone()








#The latent feature selection step
def select_latent(phi):
    if(epoch==0):
        px_z = phi/phi.sum(0).expand_as(phi)
    else:
        px_z = phi/phi_all.sum(0).expand_as(phi)
    px_z *= prior
    _, inds = px_z.max(1)
    z = classMat.index_select(0, inds.squeeze())
    return z
    


def train_classifier(x_lab, labels, x_unl):
    global phi_all
    netD.train() 
    optimizerC.zero_grad()

	#Training with labelled data
    labels = Variable(labels.cuda())
    x_lab = Variable(x_lab.cuda())
    output_before_softmax_lab = netD(x_lab)[0]
    loss_lab = loss_fn(output_before_softmax_lab, labels)
    
	#Training with unlabelled data
    x_unl = Variable(x_unl.cuda())
    output_before_softmax_unl = netD(x_unl)[0]
    output_after_softmax_unl = F.softmax(output_before_softmax_unl)
    phi_all = torch.cat((output_after_softmax_unl.data, phi_all),0)[0:3000]
    z = select_latent(output_after_softmax_unl.data)
    z = Variable(z)
    log_phi = torch.log(output_after_softmax_unl+1e-5)
    
    exponent = torch.mm(z, log_phi.t())
    exponent2 = exponent - torch.diag(exponent).view(batch_size,1).expand_as(exponent)
    temp = exponent2.exp()
    px_z_inv = temp.sum(1)
    loss_unl = px_z_inv.log().mean()

    
	#Training with fake data
    noise.data.normal_(0,1)
    gen_data = netG(noise)
    output_before_softmax_gen = netD(gen_data.detach())[0]
    output_after_softmax_gen = F.softmax(output_before_softmax_gen)
    loss_gen = (torch.log(output_after_softmax_gen+1e-5)).mean(1).mean()*-1
    

    loss = loss_lab + loss_unl + loss_gen
    loss.backward()
    
    optimizerC.step()
    train_err = (output_before_softmax_lab.data.max(1)[1]==labels.data).sum()/batch_size
    return train_err, loss_lab.data[0], loss_unl.data[0], loss_gen.data[0]
    



#Feature matching step for training the generator
def train_generator(x_unl):
    netD.train()

    optimizerG.zero_grad()
    x_unl = Variable(x_unl.cuda())
    noise.data.normal_(0,1)
    gen_data = netG(noise)
    output_unl = netD(x_unl)[1]
    output_gen = netD(gen_data)[1]
    m1 = output_unl.mean(0)
    m2 = output_gen.mean(0)
    loss_gen = (m1-m2).abs().mean()
    loss_gen.backward()
    optimizerG.step()
    
    return loss_gen.data[0]


print(testx.size())
#select labeled data
shuffle = torch.randperm(trainx.size(0))
trainx = trainx.index_select(0,shuffle)
trainy = trainy.index_select(0,shuffle)

txs = torch.zeros(count*10,3,32,32)
tys = torch.zeros(count*10)
for i in range(0,10):
    inds = trainy.eq(i).nonzero()[0:count]
    txs[i*count:(i+1)*count,:,:,:] = trainx.index_select(0, inds.squeeze())
    tys[i*count:(i+1)*count] = trainy.index_select(0, inds.squeeze())
    


#The epochs of training
train_err = torch.zeros(600)
loss_lab = torch.zeros(600)
loss_unl = torch.zeros(600)
loss_gen = torch.zeros(600)
lossG_gen = torch.zeros(600)
test_err = torch.zeros(600)

for epoch in range(0,600):
    
        
    trainx = torch.zeros(int(np.ceil(trainx_unl.size(0)/float(txs.size(0))))*txs.size(0),3,32,32)
    trainy = torch.zeros(int(np.ceil(trainx_unl.size(0)/float(txs.size(0))))*txs.size(0))

    for t in range(int(np.ceil(trainx_unl.size(0)/float(txs.size(0))))):
        inds = torch.randperm(txs.size(0))
        trainx[t*txs.size(0):(t+1)*txs.size(0)] = txs.index_select(0,inds)
        trainy[t*txs.size(0):(t+1)*txs.size(0)] = tys.index_select(0,inds)

    trainx_unl = trainx_unl[torch.randperm(trainx_unl.size(0))]
    trainx_unl2 = trainx_unl2[torch.randperm(trainx_unl2.size(0))]

    if epoch==0:
        print(trainx.shape)
        initialize(trainx[:500]) # data based initialization

    

    numBatches = 0
    N = trainx_unl.size(0) - trainx_unl.size(0)%batch_size
    for i in range(0, N, batch_size):
        numBatches +=1
        x_lab = trainx[i:i+batch_size]
        labels = trainy[i:i+batch_size].long()
        x_unl = trainx_unl[i:i+batch_size]
        te, ll, lu, lg = train_classifier(x_lab, labels, x_unl)
        train_err[epoch] += te
        loss_lab[epoch] += ll
        loss_unl[epoch] += lu
        loss_gen[epoch] += lg
        
        x_unl = trainx_unl2[i:i+batch_size]
        lgg = train_generator(x_unl)
        lossG_gen[epoch] += lgg

		#Averaging the parameters over multiple iterations. 
		#This average is used for computing test accuracy (same as in Improved GAN)
        j=0
        for param in netD.parameters():
            avg[j] = avg[j] + .0001*(param.data - avg[j])
            j += 1 
		#The norm of the weights in convolution layers is fixed throughout training
        for j in range(0,9):
            netD.conv[j].weight_g.data.copy_(saved_g[j])

    


    j=0
    backup = [None]*30
    for param in netD.parameters():
        backup[j] = param.data.clone()
        param.data.copy_(avg[j])
        j += 1 
	
    #Computation of test accuracy
    posterior = torch.zeros(testx.size(0), 10)
    netD.eval()
    training = False
    N_test = testx.size(0) - testx.size(0)%batch_size
    for i in range(0, N_test, batch_size):
        real_cpu = testx[i:i+batch_size].clone()
        input = Variable(real_cpu.cuda())
        output = netD(input)[0]
        posterior[i:i+batch_size] = output.data.cpu().clone() 
    if(testx.size(0)>N_test): 
		real_cpu = testx[N_test:].clone()
		input = Variable(real_cpu.cuda())
		output = netD(input)[0]
		posterior[N_test:] = output.data.cpu().clone() 
	

    j=0
    for param in netD.parameters():
        param.data.copy_(backup[j])
        j += 1 
    
    _, indices_fake = posterior.cpu().max(1)
    indices_fake = indices_fake.squeeze().byte()
    indices_real = testy.byte()
    test_err[epoch] = (indices_fake==indices_real).sum()/testx.size(0)
    print("epoch:%d, loss_lab:%.4f, loss_gen:%.4f, loss_unl:%.4f, lossG_gen:%.4f, train_err:%.4f, test_err:%.4f" % (epoch, loss_lab[epoch]/numBatches, loss_gen[epoch]/numBatches, 
          loss_unl[epoch]/numBatches, lossG_gen[epoch]/numBatches, train_err[epoch]/numBatches, test_err[epoch]))

    save_samples()


