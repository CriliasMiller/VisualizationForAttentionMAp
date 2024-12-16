from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.xbert import BertConfig, BertModel, BertForTokenClassification

import torch
from torch import nn
from torchvision import transforms
from transformers import BertTokenizerFast
import json
from PIL import Image
import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt
import re
import yaml
from torch.utils.data import DataLoader, Dataset

def text_input_adjust(text_input, device, cap=False):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP]
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    return text_input

class VisualDataset(Dataset):
    # Writing your own dataset class
    def __init__(self,):
        self.name = 'VisualDataset'
        
class VisualModel(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 config_bert = ''
                 ):
        super().__init__()
    
        #your own config

    def build_mlp(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)
    
    def forward(self, image, text):
        image_embeds = self.visual_encoder(image) 

        text_output = self.text_encoder.bert(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state


        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds, 
                                            attention_mask = text.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts, 
                                            output_attentions = True,     
                                            return_dict = True,
                                            mode = 'fusion',
                                        )   
        bs = image.size(0)
        cls_tokens_local = self.cls_token_local.expand(bs, -1, -1)
        text_attention_mask_clone = text.attention_mask.clone()
        local_feat_padding_mask_text = text_attention_mask_clone==0 
        local_feat_it_cross_attn = self.it_cross_attn(query=self.norm_layer_it_cross_atten(image_embeds), 
                                              key=self.norm_layer_it_cross_atten(text_embeds), 
                                              value=self.norm_layer_it_cross_atten(text_embeds),
                                              key_padding_mask=local_feat_padding_mask_text)
        local_feat_aggr = self.aggregator(query=self.norm_layer_aggr(cls_tokens_local), 
                                              key=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]), 
                                              value=self.norm_layer_aggr(local_feat_it_cross_attn[:,1:,:]))[0]
        vl_output = self.itm_head(vl_embeddings)  

        return vl_output

def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap

def process_image_text(image_path, text, tokenizer, test_transform, device):

    image = Image.open(image_path).convert('RGB')
    image = test_transform(image)   

    text = pre_caption(text, 50)

    text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False) 

    return image, text_input

def pre_caption(caption,max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption

if __name__ == '__main__':
    #-----load dataset-----

    #-----load checkpoint-----
    
    
    for batch_idx, batch in enumerate(loader):
        output = model(image.to(device), text_input)
        loss = output[:, 1].sum()

        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            mask = text_input.attention_mask.view(text_input.attention_mask.size(0), 1, -1, 1, 1)

            grads = model.text_encoder.bert.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients()
            cams = model.text_encoder.bert.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()

            cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 16, 16) * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 16, 16) * mask

            gradcam = cams * grads
            gradcam = gradcam.mean(1).cpu().detach()

        for idx, image_path in enumerate(image_dir):
            fake_list = []
            for fake_word_pos, word_pos in enumerate(fake_text_pos_list[idx]):
                if word_pos == 0:
                    continue
                fake_list.append(fake_word_pos)

            if len(fake_list) == 0:
                continue
            num_image = len(fake_list) + 1
            fig, ax = plt.subplots(num_image, 1, figsize=(15, 5 * num_image))
            rgb_image = cv2.imread(image_path)[:, :, ::-1]
            rgb_image = np.float32(rgb_image) / 255

            if not torch.all(bbox[idx] == torch.tensor([0, 0, 0, 0])):
                xmin, ymin, xmax, ymax = bbox[idx].tolist()
                cv2.rectangle(rgb_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (1, 0, 0), 2)  # 红色框，线宽为 2


            ax[0].imshow(rgb_image)
            ax[0].set_yticks([])
            ax[0].set_xticks([])
            ax[0].set_xlabel("Image")

            for pic_num, x in enumerate(fake_list):
                word = tokenizer.decode(text_input.input_ids[idx][x + 1])
                gradcam_image = getAttMap(rgb_image, gradcam[idx][x])
                ax[pic_num + 1].imshow(gradcam_image)
                ax[pic_num + 1].set_yticks([])
                ax[pic_num + 1].set_xticks([])
                ax[pic_num + 1].set_xlabel(word)

            save_path = f"/visual/batch{batch_idx}_{idx}.png"
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close(fig)
