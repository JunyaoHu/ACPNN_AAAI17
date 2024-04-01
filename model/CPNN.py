import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from sklearn.preprocessing import OneHotEncoder
from metrics.metrics import score
from dataset.dataset import MatDataset

class CPNN(nn.Module):

    def __init__(self, mode='none', v=5, n_hidden=None, n_latent=None, n_feature=None, n_output=None):
        super(CPNN, self).__init__()
        if mode in ('none', 'binary', 'augment'):
            self.mode = mode
        else:
            NotImplementedError()
        self.v = v
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.n_feature = n_feature
        self.n_output = n_output
        
        if self.n_hidden is None:
            self.n_hidden = self.n_feature * 3 // 2
        
        input_shape = (self.n_feature + (1 if self.mode == 'none' else self.n_output),)
        
        self.model = nn.Sequential(
            nn.Linear(*input_shape, self.n_hidden),
            nn.Sigmoid(),
            nn.Linear(self.n_hidden, 1)
        )
        
        self.kl_div_loss = torch.nn.KLDivLoss()

    def forward(self, X):
        inputs = self._make_inputs(X)
        outputs = self.model(inputs)
        results = torch.reshape(outputs, (X.shape[0], self.n_output))
        b = torch.reshape(-torch.log(torch.sum(torch.exp(results), dim=1)), (-1, 1))
        return torch.exp(b + results)

    def _make_inputs(self, X):
        temp = torch.reshape(torch.tile(torch.tensor([i + 1 for i in range(self.n_output)]), [X.shape[0]]), (-1, 1)).cuda()
        if self.mode != 'none': # BCPNN and ACPNN
            temp = torch.Tensor(OneHotEncoder(sparse_output=False).fit_transform(temp.cpu())).cuda()
        return torch.cat([torch.repeat_interleave(X, self.n_output, 0), temp], dim=1)

    def loss(self, X, y):
        pred = self.predict(X)
        res = self.kl_div_loss(pred.log(), y)
        return res

    def predict(self, X):
        return self.forward(X)

# Example usage
if __name__ == "__main__":
    
    #######################  dataset #############################
    dataset_name = 'emotion6'
    dataset = MatDataset(dataset_name=dataset_name)
    
    print(len(dataset))
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # 使用random_split函数进行拆分
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = 32
    
    print(len(train_dataset))
    print(len(val_dataset))

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # # 使用示例
    # for batch in train_loader:
    #     print("Training batch:", batch[0].shape, batch[1].shape)
    #     break

    # for batch in val_loader:
    #     print("Validation batch:", batch[0].shape, batch[1].shape)
    #     break
    
    ########################### model ########################
    
    mode = 'none'
    v = 5
    num_dim = 168
    num_class = 8
    model = CPNN(
        mode=mode, 
        v=v, 
        n_hidden=100, 
        n_latent=10,
        n_feature=num_dim, 
        n_output=num_class,
    )
    
    from utils.parameter import count_parameters
    count_parameters(model)