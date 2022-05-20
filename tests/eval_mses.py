import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import soundfile as sf 
import os

from iirnet.data import IIRFilterDataset
from iirnet.system import IIRNet
from iirnet.mlp import MLPModel
from iirnet.lstm import LSTMModel
import iirnet.signal as signal


parser = ArgumentParser()

parser.add_argument('--shuffle', action="store_true")
parser.add_argument('--precompute', action="store_true")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--model_name', type=str, default='mlp', help='mlp or lstm')
parser.add_argument('--num_train_examples', type=int, default=1000000)
parser.add_argument('--num_val_examples', type=int, default=10000)

temp_args, _ = parser.parse_known_args()

# let the model add what it wants
if temp_args.model_name == 'mlp':
    parser = MLPModel.add_model_specific_args(parser)
elif temp_args.model_name == 'lstm':
    parser = LSTMModel.add_model_specific_args(parser)

parser = pl.Trainer.add_argparse_args(parser)       # add all the available trainer options to argparse
args = parser.parse_args()                          # parse them args                      


# init the trainer and model 
trainer = pl.Trainer.from_argparse_args(args)

# setup the dataloaders
train_dataset = IIRFilterDataset(num_points=args.num_points, 
                                 max_order=args.max_order, 
                                 num_examples=args.num_train_examples,
                                 precompute=args.precompute)

train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                               shuffle=args.shuffle,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers)

val_datasetA = IIRFilterDataset(method="char_poly",
                               num_points=args.num_points, 
                               max_order=args.max_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetB = IIRFilterDataset(method="pass",
                               num_points=args.num_points, 
                               max_order=args.max_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetC = IIRFilterDataset(method="parametric",
                               num_points=args.num_points, 
                               max_order=args.max_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

val_datasetD = IIRFilterDataset(method="uniform_parametric",
                               num_points=args.num_points, 
                               max_order=args.max_order, 
                               num_examples=args.num_val_examples,
                               precompute=args.precompute)

# val_dataset = torch.utils.data.ConcatDataset([val_datasetA, val_datasetB, val_datasetC])
# val_dataset = torch.utils.data.ConcatDataset([val_datasetA, val_datasetC])
# val_dataset = torch.utils.data.ConcatDataset([val_datasetC])
val_dataset = torch.utils.data.ConcatDataset([val_datasetD])

val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                             shuffle=args.shuffle,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers)

# build the model
if args.model_name == 'mlp':
    model1 = MLPModel(**vars(args))
    model2 = MLPModel(**vars(args))
elif args.model_name == 'lstm':
    model = LSTMModel(**vars(args))


# PATH1 = '/home/jt/qmul/random_polys/IIRNet-main/lightning_logs/version_18/checkpoints/epoch=46-step=36753.ckpt' #Uniform Sampling
# PATH2 = '/home/jt/qmul/random_polys/IIRNet-main/lightning_logs/version_19/checkpoints/epoch=49-step=39099.ckpt' #Char sampling, zero mean
PATH1 = '/home/jt/qmul/random_polys/IIRNet-main/lightning_logs/version_21/checkpoints/epoch=48-step=38317.ckpt' #Uniform Sampling, zero mean fixed variance
PATH2 = '/home/jt/qmul/random_polys/IIRNet-main/lightning_logs/version_20/checkpoints/epoch=39-step=31279.ckpt' #Char Sampling, zero mean fixed variance
checkpoint1 = torch.load(PATH1)
model1.load_state_dict(checkpoint1['state_dict'])
model1.eval()

checkpoint2 = torch.load(PATH2)
model2.load_state_dict(checkpoint2['state_dict'])
model2.eval()


##Dataset here: http://recherche.ircam.fr/equipes/salles/listen/ 
filename = 'IRC_1002_R_R0195_T090_P330.wav'
filenames = sorted([f for f in os.listdir("/home/jt/qmul/random_polys/IIRNet-main/IRC_1002/RAW/WAV/IRC_1002_R")])
# filenames = [filenames[0]]
ii = 0
uni_mse = 0
char_mse = 0
norm = float(len(filenames))
for filename in filenames:
  test, samplerate = sf.read('/home/jt/qmul/random_polys/IIRNet-main/IRC_1002/RAW/WAV/IRC_1002_R/'+filename)

  f, _, STFT_test = scipy.signal.stft(test[:,0],nperseg=1024)
  ind = np.unravel_index(np.argmax(np.abs(STFT_test), axis=None), STFT_test.shape)

  STFT = STFT_test[:,1] #maximum frame energy always winds up here

  mag = 20*np.log10(np.abs(STFT))
  mag = mag - np.mean(mag)
  mag = mag[1:] #1024 pt STFT yields 513 points, model was trained on 512, getting rid of DC component here


  x = torch.from_numpy(np.float32(mag))
  y_hat1 = model1(x)
  w_pred1, h_pred1 = signal.sosfreqz(y_hat1, worN=512, fs=48000)

  y_hat2 = model2(x)
  w_pred2, h_pred2 = signal.sosfreqz(y_hat2, worN=512, fs=48000)

  h_pred1 = h_pred1.detach().numpy()
  h_pred2 = h_pred2.detach().numpy()

  h1= 20 * np.log10(np.abs(h_pred1.squeeze()))
  h2= 20 * np.log10(np.abs(h_pred2.squeeze()))

  mse1 = np.mean((mag - h1)**2)
  mse2 = np.mean((mag - h2)**2)

  if ii%5==0:
    fig, ax = plt.subplots()
    ax.plot(f[1:],mag,color='gray')
    ax.plot(f[1:],h1,color='red')
    ax.plot(f[1:],h2,color='blue')
    ax.legend(['HRTF','Uniform Sampling, MSE={:10.4f}'.format(mse1),'Characteristic Sampling, MSE={:10.4f}'.format(mse2)])
    plt.tight_layout()
    plt.savefig('figs/my_stft_approx_{0:03d}.png'.format(ii))
    plt.close()
  uni_mse += mse1/norm
  char_mse += mse2/norm
  ii+=1

print(uni_mse)
print(char_mse)