import os
import sys
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split
from dataset.dataset import MatDataset
from model.CPNN import CPNN
from metrics.metrics import score

import yaml
from datetime import datetime

from utils.seed import setup_seed
from utils.logger import Logger

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    cudnn.enabled = True
    cudnn.benchmark = True
    
    parser = argparse.ArgumentParser(description="EmoRank")
    parser.add_argument('--exp_name', type=str, default='demo_exp', help='training experiment name')
    parser.add_argument('--config_path', type=str, default='./config/FI/FI_vgg16.yaml', help='training config yaml file path')
    parser.add_argument('--resume_path', type=str, default='', help='resume model path')
    parser.add_argument('--log_path', type=str, default='./logs/training', help='training log saving path')
    parser.add_argument('--seed', type=int, default='1234', help='random seed')
    args = parser.parse_args()
    
    print("============== start initialization ==============")
    
    setup_seed(int(args.seed))
    
    with open(args.config_path) as f:
        config = yaml.safe_load(f)
        
    log_path = os.path.join(args.log_path, args.exp_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    config["snapshots"] = os.path.join(log_path, 'snapshots')
    os.makedirs(config["snapshots"], exist_ok=True)
    config["imgshots"] = os.path.join(log_path, 'imgshots')
    os.makedirs(config["imgshots"], exist_ok=True)
    
    now = datetime.now()
    formatted_now = now.strftime("%y%m%d-%H%M%S")

    log_txt = os.path.join(log_path, f'log_{formatted_now}.txt')
    sys.stdout = Logger(log_txt, sys.stdout)
    
    print(config)
        
    print("============== data initialization ==============")
    
    dataset_params = config["dataset_params"]
    class_num = dataset_params["class_num"]

    dataset = MatDataset(
        dataset_name=dataset_params["dataset_name"],
        path=dataset_params["data_dir"]
    )
    
    print(len(dataset))
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # 使用random_split函数进行拆分
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(len(train_dataset))
    print(len(val_dataset))
    
    print("============== model initialization ==============")
    
    model_params = config['model_params']
    
    backbone = model_params["backbone"]
    if backbone == "ACPNN":
        mode = 'augment'
    elif backbone == "BCPNN":
        mode = 'binary'
    elif backbone == "CPNN":
        mode = 'none'
    else:
        NotImplementedError()
    
    v          = model_params["v"]
    num_dim    = model_params["num_dim"]
    n_hidden   = model_params["n_hidden"]
    n_latent   = model_params["n_latent"]
    
    model = CPNN(
        mode=mode, 
        v=v, 
        n_hidden=n_hidden, 
        n_latent=n_latent,
        n_feature=num_dim, 
        n_output=class_num,
    )
    
    from utils.parameter import count_parameters
    count_parameters(model)
    
    model = model.cuda()
    
    # if args.resume_path:
    #     print(f"resume from: {args.log_path}")
    #     checkpoint = torch.load(args.log_path)
    #     model.load_state_dict(checkpoint, strict=True)
    
    print("============== start training ==============")
    
    train_params = config["train_params"]
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_params["batch_size"], 
        num_workers=train_params["dataloader_workers"], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_params["batch_size"], 
        num_workers=train_params["dataloader_workers"], 
        shuffle=False
    )
    
    # for batch in train_loader:
    #     print("Training batch:", batch[0].shape, batch[1].shape)
    #     break

    # for batch in val_loader:
    #     print("Validation batch:", batch[0].shape, batch[1].shape)
    #     break

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
    epochs = train_params['max_epochs']
    every_epoch_check = train_params['save_ckpt_freq']
    
    for i_epoch in range(epochs):
        
        for i_iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            X, y = batch
            bs, _ = X.shape
            
            if mode == 'augment': # ACPNN
                one_hot = torch.nn.functional.one_hot(torch.argmax(y, axis=1), class_num)
                X = torch.repeat_interleave(X, v, 0)
                y = torch.repeat_interleave(y, v, 0)
                one_hot = torch.repeat_interleave(one_hot, v, 0)
                v_ = torch.reshape(torch.tile(torch.tensor([1 / (i + 1) for i in range(v)]), [bs]), (-1, 1))
                y += y * one_hot * v_


            if mode == 'augment':
                new_X = []
                new_y = []
                m = y.shape[1]
                for i in range(X.shape[0]):
                    threshold = torch.mean(y[i]).item()
                    mask = y[i] > threshold
                    filtered_values = y[i][mask]
                    indices = torch.nonzero(mask, as_tuple=False).view(-1)
                    filtered_values = filtered_values.numpy()
                    k = 0
                    v = len(indices)
                    identity_matrix = torch.eye(m)
                    zero_matrix = torch.zeros(m, m)
                    for j in range(v):
                        if k > 4:
                            continue
                        rou_j = filtered_values[j]
                        indices_j = torch.tensor([[indices[k], indices[k]]], dtype=torch.long)  # indices should be a 2D tensor
                        # Specify the values to update
                        updates = torch.tensor([rou_j / v], dtype=y.dtype)
                        # Use torch.scatter_ to update the tensor
                        zero_matrix[k,k]=rou_j / v
                        updated_tensor = zero_matrix
                        plus = updated_tensor + identity_matrix
                        vector_as_matrix = y[i].view(1, -1)
                        # Perform matrix multiplication
                        result_matrix = torch.mm(vector_as_matrix, plus)
                        new_X.append(X[i])
                        new_y.append(result_matrix)
                        k += 1
                new_X = torch.stack(new_X, dim=0)
                new_y = torch.stack(new_y, dim=0).squeeze(1)
                y = torch.cat((y, new_y), dim=0)
                X = torch.cat((X, new_X), dim=0)


            loss = model.loss(X.cuda(), y.cuda())
            loss.backward()
            optimizer.step()
            
            if i_iter == 0:
                print(f"[epoch: {i_epoch:04}] loss: {loss.item():.10f}", end=' ')
        
        ################ valid #############
        
        y_all = []
        y_pred_all = []
        
        for i_iter, batch in enumerate(val_loader):
            X, y = batch
            with torch.no_grad():  
                y_pred = model(X.cuda()).cpu()
            
            y_all.append(y)
            y_pred_all.append(y_pred)
        
        y_all = torch.cat(y_all)
        y_pred_all = torch.cat(y_pred_all)
        metrics = score(y_all.numpy(), y_pred_all.numpy())
        # print(f"[epoch: {i_epoch}]", end=' ')
        for metric in metrics:
            print(f"{metric:.4f}", end=' ')
        print()
    
