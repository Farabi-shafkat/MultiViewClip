from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import torch
from model import MV_CLIP
from datasets import dataset 
from torch import nn
import numpy as np
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    train_dataset = dataset(mode='train',data_dir=args.data_dir,args=args)
    valid_dataset = dataset(mode='val',data_dir=args.data_dir,args=args)

    train_dataloader = DataLoader(train_dataset,batch_size = 32, shuffle = True,collate_fn=dataset.collate_func)
    valid_dataloader = DataLoader(valid_dataset,batch_size = 32, shuffle = False,collate_fn=dataset.collate_func)

    epochs = args.epoch

    model = MV_CLIP(device,args).to(device=device)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = -1
    clip_params = list(map(id, model.clip_model.parameters()))
    base_params = filter(lambda p: id(p) not in clip_params, model.parameters())
    optimizer = torch.optim.AdamW([
        {"params": base_params},
                    {"params": model.clip_model.parameters(),"lr": args.lr_clip}
                    ],lr=args.lr,weight_decay=0.05)
    total_steps = int(len(train_dataloader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.2 * total_steps),
                                                    num_training_steps=total_steps)
    for epoch in range(epochs):
        label_lst = torch.tensor([])
        predict_lst = torch.tensor([])
        cnt=0
        for texts,images,labels in tqdm(train_dataloader):
            optimizer.zero_grad()
            outputs, y_t,y_i,y_fuse = model(texts,images)
            labels =  torch.tensor(labels).to(device)
            #print(outputs.shape)
            #print(labels.shape)

            loss = loss_fn(y_t,labels)+loss_fn(y_i,labels)+loss_fn(y_fuse,labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step() 
        # print(predict_lst.shape,outputs.shape)
            predict_lst=torch.concat([predict_lst,outputs.detach().cpu()],dim=0)
            label_lst=torch.concat([label_lst,labels.detach().cpu()])
            #print(predict_lst.shape,label_lst.shape)
            #break
            cnt+=1
           # break
      #  print(predict_lst.shape)
        predict_lst = np.argmax(predict_lst.numpy(),axis=-1)
        label_lst = label_lst.numpy()
       # predict_lst = predict_lst.numpy()
        #print(predict_lst.shape,label_lst.shape)
        train_acc =  (predict_lst == label_lst).sum() / label_lst.shape[0]
        train_f1 = f1_score(label_lst, predict_lst)
        with torch.no_grad():
            model.eval()
            label_lst = torch.tensor([])
            predict_lst = torch.tensor([])
            for texts,images,labels in tqdm(valid_dataloader):
                outputs,_,_,_ = model(texts,images)
                labels =  torch.tensor(labels).to(device)
                predict_lst=torch.concat([predict_lst,outputs.detach().cpu()],dim=0)
                label_lst=torch.concat([label_lst,labels.detach().cpu()])
            # print(predict_lst.shape,label_lst.shape)
            predict_lst = np.argmax(predict_lst.detach().cpu().numpy(),axis=1)
            label_lst = label_lst.detach().cpu().numpy()
           # predict_lst = predict_lst.numpy()
           
            val_acc =  (predict_lst == label_lst).sum() / label_lst.shape[0]
            val_f1 = f1_score(label_lst, predict_lst,labels=[0, 1],average='macro')
            model.train()
        print('epoch: ',epoch)
        print('training accuracy:',train_acc,'val_acc:',val_acc)
        print('training f1:',train_f1,'val_f1:',val_f1)
        if val_acc>=best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),'best_val_acc_model.pt')
            torch.save(optimizer.state_dict(),'best_val_acc_opt.pt')


