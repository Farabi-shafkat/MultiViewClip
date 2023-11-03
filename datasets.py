
import torch 
import numpy as np  
from torch.utils.data import Dataset
import os
from PIL import Image
import json


class dataset(Dataset):
    def __init__(self, mode = 'train',data_dir = 'Data',args=None):
        self.image_dir = os.path.join(data_dir,'Image','dataset_image')
        data_str = 'text_json_final' if args.dataset == 'MMSD2.0' else 'text_json_clean'
        if mode=='train':
            self.text_dir = os.path.join(data_dir,data_str,'train.json')
        elif mode=='val':
            self.text_dir = os.path.join(data_dir,data_str,'valid.json')
        else:
            self.text_dir = os.path.join(data_dir,data_str,'test.json')
        
        #loading the text data
        self.data = []
        with open(self.text_dir,'r',encoding='utf-8') as f:
            f1_json = json.load(f)
            for line in f1_json:
               self.data.append(
                   {
                       'image_id' : str(line['image_id']),
                       'text' : line['text'],
                       'label': line['label']
                   }
               )
        
    def __getitem__(self, index):   
        text = self.data[index]['text']
        image = Image.open(os.path.join(self.image_dir,self.data[index]['image_id']+'.jpg'))
        label =self.data[index]['label']
        return text,image,label
    
    def __len__(self):
        return len(self.data)
    

    def collate_func(batch_data):
        batch_size = len(batch_data)
 
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        label_list = []
        
        for instance in batch_data:
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
       #     id_list.append(instance[3])
        return text_list, image_list, label_list