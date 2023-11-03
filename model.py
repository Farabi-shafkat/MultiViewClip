from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertLayer,BertConfig
import copy
import torch

class TransformerEncoder(nn.Module):
    def __init__(self,num_layers,hid_size,n_head):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hid_size,nhead=n_head) for _ in range(num_layers)])
    def forward(self,src,att_mask):
        for layer in self.encoder:
            src = layer(src=src,src_mask=att_mask)
        return src

class MultimodalEncoder(nn.Module):
    def __init__(self, config, layer_number):
        super(MultimodalEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(layer_number)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_attentions = []
        for layer_module in self.layer:
            hidden_states, attention = layer_module(hidden_states, attention_mask, output_attentions=True)
            all_encoder_attentions.append(attention)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_encoder_attentions

class MV_CLIP(nn.Module):
    def __init__(self,device,args):
        super(MV_CLIP, self).__init__()
        self.image_size= 768
        self.text_size = 512
        self.label_number = 2
        self.args = args

        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_linear = nn.Sequential(
                                        nn.Linear(self.text_size,self.text_size),
                                        nn.Dropout(args.dropout),
                                        nn.GELU()
                                        )
        self.image_linear = nn.Sequential(
                                        nn.Linear(self.image_size,self.image_size),
                                        nn.Dropout(args.dropout),
                                        nn.GELU()
                                        )
       # self.image_linear = nn.Linear(self.image_size,self.image_size)

        self.config = BertConfig.from_pretrained("bert-base-uncased")
        self.config.hidden_size = 512
        self.config.num_attention_heads = 8
        
        self.transformerEncoder = MultimodalEncoder(self.config, layer_number=3)#TransformerEncoder(8,self.text_size,4)
        
        self.keyless_att = nn.Linear(512,1,bias= False)
        
        self.classifier_fuse = nn.Linear(self.text_size , self.label_number)
        self.classifier_text = nn.Linear(self.text_size, self.label_number)
        self.classifier_image = nn.Linear(self.image_size, self.label_number)

        self.device = device

    def forward(self,text,images):
        inputs = self.processor(text=text,images=images,return_tensors='pt',padding='max_length', truncation=True,max_length = 77).to(self.device)
        #print(inputs['attention_mask'].shape)
        outputs = self.clip_model(**inputs)
        text_feat = outputs['text_model_output']['last_hidden_state']
        vision_feat = outputs['vision_model_output']['last_hidden_state']
        text_cls = outputs['text_model_output']['pooler_output']
        vision_cls = outputs['vision_model_output']['pooler_output']
       # print(text_feat.size(),'text_feat')
       # print(vision_feat.size(),'vision_feat')
       # print(vision_cls.size(),'visioncls_feat')
       # print(text_cls.size(),'textcls_feat')
        y_image = nn.functional.softmax(self.classifier_image(vision_cls),dim=-1)
        y_text = nn.functional.softmax(self.classifier_text(text_cls),dim=-1)
        #print(y_image.size(),'y_image')
        #print(y_text.size(),'y_text')

        text_embed = self.text_linear(text_feat)
        vision_embed = self.image_linear(vision_feat)
        text_embed = self.clip_model.text_projection(text_embed)
        vision_embed = self.clip_model.visual_projection(vision_embed)

        #print(text_embed.size(),'text embed')
        #print(vision_embed.size(),'vision embed')
        
        fused_feat = torch.cat([vision_embed,text_embed],dim=1)
        #print(fused_feat.shape,'fused_feat')
        attention_mask = torch.cat((torch.ones(text_cls.shape[0], 50).to(text_cls.device), inputs['attention_mask']), dim=-1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        hid_layers,_ = self.transformerEncoder(fused_feat, extended_attention_mask, output_all_encoded_layers=False)
        decoded_feat = hid_layers[-1]
      #  print(decoded_feat.size(),'decoded_feat')
        decoded_image_cls = decoded_feat[:,0,:].squeeze(1)
        if not self.args.cls_only:
            #print("rntry")
            decoded_text_feat= decoded_feat[:,50:,:]#.squeeze(1)
        #print(torch.arange(decoded_text_feat.shape[0]))
        #print(inputs['input_ids'].to(torch.int).argmax(dim=-1))
        #print(inputs['input_ids'].shape)
        #print(inputs.keys())
            decoded_text_cls =  decoded_feat[torch.arange(decoded_text_feat.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)]
       # decoded_text_cls =  #decoded_text_feat[:,0,:].squeeze(1)#decoded_feat[torch.arange(decoded_text_feat.shape[0], device=inputs['input_ids'].device), inputs['input_ids'].to(torch.int).argmax(dim=-1)]
        #print(decoded_image_cls.size(),'decoded_imgfeat_cls')
        #print(decoded_text_cls.size(),'decoded_txtfeat_cls')
      #  print(decoded_image_cls.size())
        else:
            decoded_text_cls = decoded_feat[:,50,:].squeeze(1)
        pt = self.keyless_att(decoded_text_cls)
        pv = self.keyless_att(decoded_image_cls)
     #   print(pt.size(),'pt',pv.size())
      #  print(torch.stack((pt, pv), dim=1).size(),'stacked')
        att_score = nn.functional.softmax(torch.stack((pt, pv), dim=-1),dim=-1)
       # print(att_score.size(),'att')
        pt, pv = att_score.split([1,1], dim=-1)
       # print(pt.size(),'pt after')
       # print(decoded_text_cls.size(),'cls')
        fuse_feat = pt.squeeze(1)*decoded_text_cls+pv.squeeze(1)*decoded_image_cls
        #print(fuse_feat.shape,'fuse_feat')
        y_fuse = nn.functional.softmax(self.classifier_fuse(fuse_feat),dim=-1)
        

        y_o = (nn.functional.softmax(y_text,dim=-1) + nn.functional.softmax(y_image,dim=-1) + nn.functional.softmax(y_fuse,dim=-1) )
       
        return y_o,y_text,y_image,y_fuse