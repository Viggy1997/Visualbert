# -*- coding: utf-8 -*-
"""openI_VQA.ipynb

"""

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/YIKUAN8/Transformers-VQA.git
# %cd Transformers-VQA/
!pip install -r requirements.txt


#line 1: UNITER; line 2:LXMERT, line 3: VisualBERT. Comment out selected lines if you don't want to use this model
#if the pre-trained VisualBERT cannot be downloaded succesfully, rerun one more time or refer to this link: https://drive.google.com/file/d/1kuPr187zWxSJbtCbVW87XzInXltM-i9Y/view?usp=sharing
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kuPr187zWxSJbtCbVW87XzInXltM-i9Y' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kuPr187zWxSJbtCbVW87XzInXltM-i9Y" -O models/pretrained/visualbert.th && rm -rf /tmp/cookies.txt

"""####**0.3 Download OpenI dataset.**

A detailed description of this dataset can be found at [here](https://openi.nlm.nih.gov/). In summary, there are 3684 CXR Image-Report pairs in this dataset. Each pair has an annotation of 15 throacic findings from MESH terms. We convert the raw data to a dataframe with better visibility. It can be accessed with the following command or this [link](https://drive.google.com/file/d/1i3wcfXJbH_4q3rS2rvLxtzbMiO-KuZCG/view?usp=sharing).
"""

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1i3wcfXJbH_4q3rS2rvLxtzbMiO-KuZCG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1i3wcfXJbH_4q3rS2rvLxtzbMiO-KuZCG" -O data/openIdf.csv && rm -rf /tmp/cookies.txt

"""***0.3.1 Have a glance of this dataframe, column 'TXT' is the radiology report; column 'split' and 'id' are self-explantory; All other columns are the 15 findings. Our task will be a 15-labels binary classification with visual and semantic input.***"""

import pandas as pd
openI = pd.read_csv('data/openIdf.csv',index_col=0)
openI.head()

"""####**0.4 Download the visaul features extracted by BUTD. 36 2048-dimension visual feature is extracted from each CXR Image. We use this [implementation](https://github.com/airsplay/py-bottom-up-attention). This step will take a while (~1min). To save downloading time, you can also make a copy of this [shareable link](https://drive.google.com/file/d/1BFw0jc0j-ffT2PhI4CZeP3IJFZg3GxlZ/view?usp=sharing) to your own google drive and mount you colab to your gdrive.**


*If you are interested in the original CXR images, which is unnecessary to out project , you can access them [here](https://drive.google.com/drive/folders/1s5A0CFB6-2N5ThbuorUK1t-bUEKmZnjz?usp=sharing).*
"""

!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BFw0jc0j-ffT2PhI4CZeP3IJFZg3GxlZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BFw0jc0j-ffT2PhI4CZeP3IJFZg3GxlZ" -O data/openI_v_features.pickle && rm -rf /tmp/cookies.txt

"""***0.4.1 Load visual features***"""

import pickle
openI_v_f = pickle.load( open( "/content/Transformers-VQA/data/openI_v_features.pickle", "rb" ) )

assert set(list(openI_v_f.keys())) == set(openI.id.values), "Visual Features are inconsistent with openI dataset"

feature_example, bbox_example, (img_w_example, img_h_example) = openI_v_f[openI.id.iloc[0]]

feature_example.shape, bbox_example.shape, (img_w_example, img_h_example)

"""####**Now We have download all data, models, and dependencies. We are good to go!!!**
**1. Change default arguments**

First, let's check it out!
"""

import sys
sys.argv=['']
del sys
from param import args

args.__dict__

"""***1.1*** Let's overwrite some arguments***"""

args.batch_size = 18
args.epochs = 2
args.model = 'visualbert' # use visualbert
args.load_pretrained = '/content/Transformers-VQA/models/pretrained/visualbert.th' #load pretrained visualbert model
args.max_seq_length = 128 #truncate or pad report lengths to 128 subwords



"""####**2. Create customized dataloader**"""

findings = list(openI.columns[1:-2])
findings

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
class OpenIDataset(Dataset):
  def __init__(self, df, vf, split, model = 'lxmert'):
    # train_test_split and prepare labels
    self.dataset = df[df['split'] == split]
    self.visual_features = vf
    self.id_list = self.dataset.id.tolist()
    self.report_list = self.dataset.TXT.tolist()
    self.findings_list = self.dataset.columns[1:-2]
    self.target_list = self.dataset[self.findings_list].to_numpy().astype(np.float32)
    self.model = model

  def __len__(self):
    return len(self.id_list)

  def __getitem__(self, item):
    cxr_id = self.id_list[item]
    target = self.target_list[item]
    boxes, feats, (img_w, img_h) = self.visual_features[cxr_id]
    report = self.report_list[item]
    if self.model == 'uniter':
      boxes = self._uniterBoxes(boxes)
    if self.model == 'lxmert':
      boxes[:, (0, 2)] /= img_w
      boxes[:, (1, 3)] /= img_h
    return cxr_id, feats, boxes, report, target

  def _uniterBoxes(self, boxes):#uniter requires a 7-dimensiom beside the regular 4-d bbox
    new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
    new_boxes = np.zeros((boxes.shape[0],7),dtype='float32')
    new_boxes[:,1] = boxes[:,0]
    new_boxes[:,0] = boxes[:,1]
    new_boxes[:,3] = boxes[:,2]
    new_boxes[:,2] = boxes[:,3]
    new_boxes[:,4] = new_boxes[:,3]-new_boxes[:,1] #w
    new_boxes[:,5] = new_boxes[:,2]-new_boxes[:,0] #h
    new_boxes[:,6]=new_boxes[:,4]*new_boxes[:,5] #area
    return new_boxes

training = OpenIDataset(df = openI, vf = openI_v_f,  split='train', model = args.model)
testing = OpenIDataset(df = openI, vf = openI_v_f,  split='test', model = args.model)

train_loader = DataLoader(training, batch_size=args.batch_size,shuffle=True, num_workers=0,drop_last=True, pin_memory=True)
test_loader = DataLoader(testing, batch_size=128,shuffle=False, num_workers=0,drop_last=False, pin_memory=True)

"""####**3. Model, Optimizer, Loss Function, and Evaluation Function**"""

from vqa_model import VQAModel
#init model
model = VQAModel(num_answers = len(findings), model = args.model)

#load pretrained weights
model.encoder.load(args.load_pretrained)

#send to GPU
model = model.cuda()

import torch
loss = torch.nn.BCEWithLogitsLoss()

from src.optimization import BertAdam
optim = BertAdam(list(model.parameters()),lr=args.lr,warmup=0.1,t_total=len(train_loader)*args.epochs)
# t_total denotes total training steps
# batch_per_epoch = len(train_loader)
# t_total = int(batch_per_epoch * args.epochs)

# Evaluation function, we will report the AUC and accuray of each finding
def eval(target, pred):
    acc_list = []
    for i, d in enumerate(findings[:-1]): #normal is excluded
        acc = np.mean(target[:,i] == (pred[:,i]>=0.5))
        print(i,d,acc)
        acc_list.append(acc)
    print('Averaged: '+str(np.average(acc_list)))

sgmd = torch.nn.Sigmoid()

"""####**4. HIT and RUN**"""

from tqdm.notebook import tqdm

iter_wrapper = (lambda x: tqdm(x, total=len(train_loader))) if args.tqdm else (lambda x: x)
best_valid = 0
for epoch in range(args.epochs):
  epoch_loss = 0
  for i, (cxr_id, feats, boxes, report, target) in iter_wrapper(enumerate(train_loader)):
    model.train()
    optim.zero_grad()
    feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
    logit = model(feats, boxes, report)
    running_loss = loss(logit, target)
    running_loss = running_loss * logit.size(1)
    epoch_loss += running_loss
    running_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optim.step()
  print("Epoch "+str(epoch)+": Training Loss: "+str(epoch_loss/len(train_loader)))
  print('Evaluation: ')
  model.eval()
  logit_list, target_list = [], []
  iter_wrapper = (lambda x: tqdm(x, total=len(test_loader)))
  for i, (cxr_id, feats, boxes, report, target) in iter_wrapper(enumerate(test_loader)):
    target_list.append(target)
    with torch.no_grad():
      feats, boxes = feats.cuda(), boxes.cuda()
      logit = model(feats, boxes, report)
      logit_list.append(sgmd(logit).cpu().numpy())

  eval(np.concatenate(target_list,axis = 0), np.concatenate(logit_list,axis = 0))
