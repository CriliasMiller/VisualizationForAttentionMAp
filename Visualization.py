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
from dataset.dataset import DGM4_Dataset
import yaml
from torch.utils.data import DataLoader, Dataset

def text_input_adjust(text_input, device, cap=False):
    # input_ids adaptation
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

    return text_input

class VisualDataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30, is_train=True): 
        
        self.root_dir = ''       
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))

        self.transform = transform
        self.max_words = max_words
        self.image_res = 256
    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):
        ann = self.ann[index]
        img_dir = ann['image']    
        image_dir_all = f'{self.root_dir}/{img_dir}'

        try:
            image = Image.open(image_dir_all).convert('RGB')   
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")
        image = self.transform(image)
        caption = pre_caption(ann['text'], self.max_words)
        fake_text_pos = ann['fake_text_pos']
        fake_text_pos_list = torch.zeros(self.max_words)

        bbox = ann['fake_image_box']
    
        fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)
        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            fake_image_box = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float)

        for i in fake_text_pos:
            if i<self.max_words:
                fake_text_pos_list[i]=1
        return image_dir_all, image, caption, fake_text_pos_list, fake_image_box
    
class Multilmodelattention(nn.Module):
    def __init__(self,input_dim,head,*args,**kwargs):
        super().__init__()
        self.selfatten = nn.MultiheadAttention(input_dim,head,dropout=0.0, batch_first=True)
        self.crossatten = nn.MultiheadAttention(input_dim,head,dropout=0.0, batch_first=True)
        
    def forward(self, query,key,value,key_padding_mask):
        x, _ = self.selfatten(query, query, query)

        x, _ = self.crossatten(x, key, value, key_padding_mask=key_padding_mask)
        x = x + query
        return x
class VisualASAP(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 config_bert = ''
                 ):
        super().__init__()
    
        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 

        self.text_encoder = BertForTokenClassification.from_pretrained(text_encoder, config=bert_config, label_smoothing=0.0)   
        text_width = self.text_encoder.config.hidden_size
        self.itm_head = self.build_mlp(input_dim=text_width, output_dim=2)

        self.it_cross_attn = Multilmodelattention(text_width, 12)
        self.cls_token_local = nn.Parameter(torch.zeros(1, 1, text_width))
        self.norm_layer_aggr =nn.LayerNorm(text_width)
        self.aggregator = nn.MultiheadAttention(text_width, 12, dropout=0.0, batch_first=True)
        self.norm_layer_it_cross_atten =nn.LayerNorm(text_width)
        self.UtilsB = nn.Parameter(torch.tensor(0.5))

    def build_mlp(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim* 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, output_dim)
        )
    
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
        
        vl_embeddings = output_pos.last_hidden_state[:,0,:] + self.UtilsB*local_feat_aggr.squeeze(1)
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
    
    # # input_ids adaptation
    text_input.input_ids.unsqueeze(0)
    text_input.attention_mask.unsqueeze(0)
    input_ids_remove_SEP = [x[:-1] for x in text_input.input_ids]
    maxlen = max([len(x) for x in text_input.input_ids])-1
    input_ids_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in input_ids_remove_SEP] # only remove SEP as HAMMER is conducted with text with CLS
    text_input.input_ids = torch.LongTensor(input_ids_remove_SEP_pad).to(device) 

    # attention_mask adaptation
    attention_mask_remove_SEP = [x[:-1] for x in text_input.attention_mask]
    attention_mask_remove_SEP_pad = [x + [0] * (maxlen - len(x)) for x in attention_mask_remove_SEP]
    text_input.attention_mask = torch.LongTensor(attention_mask_remove_SEP_pad).to(device)

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
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    test_transform = transforms.Compose([
        transforms.Resize((256,256),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   
    ann_file = 'val.json'
    test_config = 'test.yaml'
    config = yaml.load(open(test_config, 'r'), Loader=yaml.Loader)

    train_dataset = VisualDataset(ann_file=config['val_file'], transform=test_transform, max_words=config['max_words'], is_train=False)              
    loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=1, pin_memory=True)

    #-----load checkpoint-----
    model_path = 'checkpoint.pth'
    bert_config_path = 'config_bert.json'
    device = 'cuda:1'

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = VisualASAP(text_encoder='bert-base-uncased', config_bert=bert_config_path)

    checkpoint = torch.load(model_path, map_location='cpu') 
    state_dict = checkpoint['model'] 

    pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)   
    state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

    msg = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    block_num = 8
    
    model.text_encoder.bert.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True
    
    for batch_idx, batch in enumerate(loader):
        image_dir, image, text, fake_text_pos_list, bbox = batch
        text_input = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True, return_attention_mask=True, return_token_type_ids=False)
        text_input = text_input_adjust(text_input, device)

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
