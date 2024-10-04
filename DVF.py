from google.colab import drive
drive.mount('gdrive')

import torch
from torchvision import transforms
from torchvision.transforms import Compose
import math
from torch import optim
from tqdm import tqdm
import gc
import torch.nn as nn
import glob
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import ViTFeatureExtractor, ViTForImageClassification,ViTModel
import os

imageFolderpath = '/content/gdrive/MyDrive/birds'
images = glob.glob(imageFolderpath + '/*/*.jpg')


Grounding_dino = "IDEA-Research/grounding-dino-tiny"
Grounding_dino_processor = AutoProcessor.from_pretrained(Grounding_dino)
Grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(Grounding_dino).to('cuda')


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.Grayscale(num_output_channels=3),
    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
    transforms.ToTensor(),          
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



model_name = 'google/vit-base-patch16-224'
model = ViTModel.from_pretrained(model_name,attn_implementation="eager")
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

def splitToBatches(image_list,batch_size):
  batch_list = []

  for i in range(0,len(image_list),batch_size):
    batch_list.append(image_list[i:i+batch_size])

  return batch_list

"""
Batch images
"""
import random
B = 4
random.shuffle(images)
batch_images = splitToBatches(images,B)
print('batch_images',batch_images)


def classProxies(images):
  class_singles = {}
  class_proxies =  {}
  for i in images:
    # print(i)
    curr_class_name = i.split('/')[-2]
    img = Image.open(i)
    inputs = feature_extractor(images=img, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    embedding_tokens = outputs.pooler_output
    if curr_class_name not in class_singles.keys(): class_singles[curr_class_name] = [embedding_tokens]
    else : class_singles[curr_class_name].append(embedding_tokens)
  for k,v in class_singles.items() :
    v = torch.vstack(v)
    class_proxies[k] = v.mean(dim=0)

  return class_proxies


classProxy_dict = classProxies(images)

class Dvf_VIT(nn.Module):
    def __init__(self):
        super(Dvf_VIT, self).__init__()
        self.model_name = 'google/vit-base-patch16-224'
        self.model = ViTModel.from_pretrained(self.model_name,attn_implementation="eager")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)

        for param in self.model.parameters():
          param.requires_grad = True

    def forward(self,img_tensor):

      # inputs = self.feature_extractor(images=img_tensor, return_tensors="pt")
      # outputs = self.model(**inputs, output_attentions=True)
      outputs = self.model(img_tensor, output_attentions=True)
      attentions = outputs.attentions[-1]
      class_token = outputs.last_hidden_state[0, 0]
      embedding_tokens = outputs.last_hidden_state

      return outputs,attentions,class_token,embedding_tokens

dvf_model = Dvf_VIT()
dvf_model = dvf_model.to('cuda')

modelFolderpath = './models'
if not os.path.isdir(modelFolderpath):
  os.makedirs(modelFolderpath, exist_ok = True)


textClass = "Bird"

lr = 3 * math.exp(-2)
epoch = 10
all_network_params = list(dvf_model.parameters())
optimizer = optim.SGD(all_network_params, lr = lr)
exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epoch,eta_min=0)

for e in tqdm(range(epoch)):
  dvf_model.train()
  for b in batch_images :
    L_total = 0
    batch_El1 = []
    class_names = []
    L_con = 0 
    L_pnca = 0
    for i in b:
      curr_class_name = i.split('/')[-2]
      class_names.append(curr_class_name)
      img = Image.open(i)

      """Grounding dino to locate and magnify objects."""
      inputs = Grounding_dino_processor(images=img, text=textClass, return_tensors="pt")
      with torch.no_grad():
        outputs = Grounding_dino_model(**inputs.to('cuda'))
      results = Grounding_dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.25,
        target_sizes=[img.size[::-1]] )

      scores = results[0]['scores']
      boxes = results[0]['boxes']
      labels = results[0]['labels']

      for score, box, label in zip(scores, boxes, labels):
        if score > 0.25: ## According to paper, you may take score>0.5
          current_bbox = box.tolist()
          x1, y1, x2, y2 = current_bbox
          cropped_image = img.crop((x1, y1, x2, y2))
          original_size = cropped_image.size
          new_size = (int(original_size[0] * 1.1), int(original_size[1] * 1.1))
          resize_transform = transforms.Resize(new_size)
          image = resize_transform(cropped_image)  

      img = transform(image).to('cuda')
      outputs,attentions,class_token,embedding_tokens = dvf_model(img.unsqueeze(0))

      attention_shape = attentions[0].shape
      A_cap = torch.zeros((attention_shape))
      for a in attentions :
        A_cap = torch.add(A_cap.to('cuda'),a.to('cuda'))

      batch_size, sequence_length, hidden_size = embedding_tokens.shape
      l1 = nn.Parameter(torch.randn(hidden_size, 1))

      Z = torch.matmul(embedding_tokens.to('cuda'), l1.to('cuda'))
      Z = torch.sigmoid(Z)

      El1 = []
      mean_activations = embedding_tokens.mean(dim=-1)
      high_activation_indices = torch.topk(mean_activations, k=12).indices #(O -> Semantic score as per paper)
      for idx in high_activation_indices[0]:
        El1.append(mean_activations[0][idx].item())

      El1 = torch.tensor(El1)
      El1 = torch.cat((class_token.to('cuda'), El1.to('cuda')), dim = 0 ).reshape(1,-1)
      ### El1 forms the input sequence to the last layer of the Lth transformer layer (last)

      vit_pooler = nn.Sequential(
          nn.Linear(780, 780,bias=True),
          nn.Tanh(),
      ).to('cuda')
      Er = vit_pooler(El1)
      batch_El1.append(Er)

    """L_con and L_pnca Calculations"""
    out_cl = 0
    L_pnca = 0 
    for ei,cn_i in zip(batch_El1,class_names):
      """L_con"""
      in_cl = 0
      for ej,cn_j in zip(batch_El1,class_names):
        if cn_i == cn_j :
          in_cl += (1- torch.dot(ei.squeeze(0),ej.squeeze(0)))
        else :
          in_cl += max((torch.dot(ei.squeeze(0),ej.squeeze(0))),0)
      out_cl += in_cl

      """L_pnca"""
      num_pnca = torch.exp(-torch.dist(torch.norm(ei,p=2),torch.norm(classProxy_dict[cn_i],p=2), p = 2))
      denom_pnca = 0 
      for c,p in classProxy_dict.items(): 
        denom_pnca+= torch.exp(-torch.dist(torch.norm(ei,p=2),torch.norm(p,p=2), p = 2))
      L_pnca += -torch.log(num_pnca/denom_pnca)


    L_con = out_cl/(B*B)
    L_pnca = L_pnca/B

    L_total = torch.add(L_con,L_pnca)
    optimizer.zero_grad()
    L_total.backward(retain_graph=True)
    optimizer.step()
    torch.cuda.empty_cache()
    gc.collect()
  exp_lr_scheduler.step()
  if e % 5 == 0  :
      model_save_name = f'model_{e}.pt'
      model_save_path = os.path.join(modelFolderpath, f'model_{e}.pt')
      torch.save(model.state_dict() , model_save_path)