import argparse
from train import train
from predict import predict

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Example script with command-line arguments')
    parser.add_argument('--data_dir', type=str, help='directory of the dataset')
    parser.add_argument('--dataset', type=str, help='MMSD/MMSD2.0/text_json_clean')
    parser.add_argument('--test_dataset', type=str, help='MMSD/MMSD2.0/text_json_clean')
    parser.add_argument('--lr',type=float)
    parser.add_argument('--lr_clip',type=float)
    parser.add_argument('--dropout',type=float)
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--cls_only',type=bool)
    args = parser.parse_args()
    #print(args.cls_only)
    train(args)
    predict(args)