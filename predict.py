from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import numpy as np
import torch
from model import MV_CLIP
from datasets import dataset 

def predict(args):
    test_dataset = dataset(mode='test',data_dir=args.test_dataset,args=args)
    test_dataloader = DataLoader(test_dataset,batch_size = 32, shuffle = False,collate_fn=dataset.collate_func)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MV_CLIP(device,args).to(device=device)
    model.load_state_dict(torch.load('best_val_acc_model.pt'))
    with torch.no_grad():
        model.eval()
        label_lst = torch.tensor([])
        predict_lst = torch.tensor([])
        for texts,images,labels in tqdm(test_dataloader):
            outputs,_,_,_ = model(texts,images)
            labels =  torch.tensor(labels).to(device)
            predict_lst=torch.concat([predict_lst,outputs.detach().cpu()],dim=0)
            label_lst=torch.concat([label_lst,labels.detach().cpu()])
        # print(predict_lst.shape,label_lst.shape)
        predict_lst = np.argmax(predict_lst.detach().cpu().numpy(),axis=1)
        label_lst = label_lst.detach().cpu().numpy()
        # predict_lst = predict_lst.numpy()
        
        test_acc =  (predict_lst == label_lst).sum() / label_lst.shape[0]
        test_f1 = f1_score(label_lst, predict_lst)
        model.train()

        print('test accuracy:',test_acc)
        print('test f1:',test_f1)