# library
import argparse
import random
import numpy as np
import torch
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import pandas as pd ## code for repitition

parser = argparse.ArgumentParser(description= 'Tistory example')

# add argument
parser.add_argument('--no',                ## code for repitition
                    type= int,
                    help= 'index no')      # 몇 번째 반복 실험인지
parser.add_argument('--seed',
                    type= int,
                    default= 7,
                    help= 'seed number')    # 반복 실험을 위한 seed
parser.add_argument('--lr',
                    type= float,
                    default= 1e-3,
                    help= 'learning rate')  # learning rage
parser.add_argument('--epochs',
                    type= int,
                    default= 10,
                    help= 'epoch')          # epoch

# main
def main() :
    # argument
    args = parser.parse_args()

    # seed setting : 재현성을 위한 seed 고정
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only= True)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    # load data
    data = load_iris()
    X = data.data
    y = data.target

    # data split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size= 0.2,
                                                        shuffle= True,
                                                        random_state= args.seed)

    # make data loader
    trainset = TensorDataset(torch.tensor(X_train, dtype= torch.float32),
                             torch.tensor(y_train, dtype= torch.long))
    testset = TensorDataset(torch.tensor(X_test, dtype= torch.float32),
                            torch.tensor(y_test, dtype= torch.long))
    
    trainloader = DataLoader(trainset, batch_size= 32, shuffle= True)
    testloader = DataLoader(testset, batch_size= 32, shuffle= True)

    # model setting
    model = MLP(in_shape= data.data.shape[1],
                hidden_shape= 32,
                out_shape= len(data.target_names))
    optimizer = optim.Adam(model.parameters(), lr= args.lr)
    criterion = nn.CrossEntropyLoss()

    # train
    for epoch in range(args.epochs) :
        model.train()
        for batch in trainloader :
            # optimizer 초기화
            optimizer.zero_grad()
            # model 예측
            output = model(batch[0])
            # loss 계산
            loss = criterion(output, batch[1])
            # 역전파
            loss.backward()
            # 가중치 업데이트
            optimizer.step()
        # 진행 상황 확인
        print('[{}/{}] Loss : {:.4f}'.format(epoch+1, args.epochs, loss))

    # evaluation
    preds = []
    trues = []
    model.eval()

    with torch.no_grad() :
        for batch in testloader :
            output = model(batch[0])
            _, pred = torch.max(output.data, 1)
            preds.extend(pred.numpy())
            trues.extend(batch[1].numpy())
    acc = accuracy_score(trues, preds)
    print('\nSetting') ## code for repitition
    print('Seed : {}, LR : {}'.format(args.seed, args.lr)) ## code for repitition
    print('Acc : {:.4f}'.format(acc))

    ## code for repitition
    eval_result = pd.read_csv('./eval.csv')
    eval_result.loc[args.no] = [args.no,
                                args.lr,
                                args.seed,
                                acc]
    eval_result.to_csv('eval.csv', index= False)
    print('{} Done\n'.format(args.no))
    ## code for repitition

class MLP(nn.Module) :
    def __init__(self, in_shape, hidden_shape, out_shape) :
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_shape, hidden_shape)
        self.fc2 = nn.Linear(hidden_shape, out_shape)
    def forward(self, x) :
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == '__main__' :
    main()