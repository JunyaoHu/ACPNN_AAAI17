from torchvision import transforms
from PIL import Image
import torch
from torch.utils.data import Dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

path = "/home/ubuntu12/wwc/datasets/Emotion6"
FI_label_dict = {
    'Amusement': 0,
    'Contentment': 1,
    'Awe': 2,
    'Excitement': 3,
    'Fear': 4,
    'Sadness': 5,
    'Disgust': 6,
    'Anger': 7,
}

Unbiased_label_dict = {
    'anger': 0,
    'sadness': 1,
    'surprise': 2,
    'fear': 3,
    'love': 4,
    'joy': 5,
}

EmotionROI_label_dict = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'sadness': 4,
    'surprise': 5,
}


Emotion6_label_dict = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'sadness': 4,
    'surprise': 5,
    'neutral': 6,
}

LDL_label_dict = {
    'Amusement': 0,
    'Awe': 1,
    'Contentment': 2,
    'Excitement': 3,
    'Anger': 4,
    'Disgust': 5,
    'Fear': 6,
    'Sadness': 7,
}

class EmotionDataset(Dataset):
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, index):
        image = Image.open(self.path[index]).convert('RGB')
        label = self.label[index]
        if self.transform:
            image = self.transform(image)
        return image, label
    
class MultiClassClassificationEmotionDataset(EmotionDataset): 
    def __init__(self, path, type, label_dict, task_type, transform=None):
        labels = os.listdir(os.path.join(path, type))
        self.path = []
        self.label = []
        self.transform = transform
        self.num_class = len(label_dict)
        
        def _process_label(label, label_dict, task_type):
            """
            input
              0, {'xxx': 0,...}, xxx
            
            output
                if multi_class_classification
                    => 0
                if multi_label_classification
                    => [1,0,...]
                if label_ranking
                    => [1,2,...]
                if label_distribution
                    => [1,0,...]
            """
            
            if task_type == "multi_class_classification":
                res = label_dict[label]
            elif task_type == "multi_label_classification":
                res = [0 for _ in range(self.num_class)]
                res[label_dict[label]] = 1
            elif task_type == "label_ranking":
                res = [2 for _ in range(self.num_class)]
                res[label_dict[label]] = 1
            elif task_type == "label_distribution":
                res = [0 for _ in range(self.num_class)]
                res[label_dict[label]] = 1
            else:
                NotImplementedError()
            return torch.Tensor(res)
        
        for label in labels:
            label_path = os.path.join(path, type, label)
            imgs = os.listdir(label_path)
                    
            for img in imgs:
                img_path = os.path.join(label_path, img)
                self.path.append(img_path)                
                self.label.append(_process_label(label, label_dict, task_type))
                
class MultiLabelClassificationEmotionDataset(EmotionDataset):
    def __init__(self, path, type, label_dict, task_type, transform=None):
        labels = os.listdir(os.path.join(path, type))
        self.path = []
        self.label = []
        self.transform = transform
        
        for label in labels:
            label_path = os.path.join(path, type, label)
            imgs = os.listdir(label_path)
                    
            for img in imgs:
                img_path = os.path.join(label_path, img)
                self.path.append(img_path)                   
                self.label.append(label_dict[label])
    
class MultiLabelDistributionEmotionDataset(EmotionDataset):
    def __init__(self, path, type, transform=None):
        with open(os.path.join(path, f'{type}.txt'),'r') as f:
            self.data = f.readlines()
        self.data = self.data[1:]
        self.path = []
        self.label = []
        self.transform = transform
        
        for line in self.data:
            elems = line.split(' ')
            img_path = os.path.join(path, elems[0])
            
            label = elems[1:]
            label = [float(l) for l in label]
            # vote_num = sum(label)
            # label = [l / vote_num for l in label]
            
            self.path.append(img_path)
            self.label.append(label)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
])

dataset1 = MultiLabelDistributionEmotionDataset(path, "train", transform)
print(len(dataset1))

dataset2 = MultiLabelDistributionEmotionDataset(path, "test", transform)
print(len(dataset2))

total_len = len(dataset1)+len(dataset2)

# data  = dataset[0]
# img, label = data
# print(img.shape, img.min(), img.max(), label)

# img = einops.rearrange(img, "c h w -> h w c")
# media.show_image(img, title=str(label))

from torchvision import models
backbone = models.vgg16(pretrained=True)
# backbone = models.vgg19_bn(pretrained=True)
# backbone = models.vgg19(pretrained=True)
backbone.cuda()
backbone.eval()

from sklearn.decomposition import PCA
pca = PCA(n_components=280)

all_data = torch.zeros(total_len, 1000)
all_labels = torch.zeros(total_len, 7)

from tqdm import tqdm

for dataset in [dataset1, dataset2]:
    for i in tqdm(range(len(dataset1))):
        data, label = dataset1[i]
        # torch.Size([3, 224, 224])
        # print(data.shape)
        # print(label)
        
        with torch.no_grad():
            out = backbone(data.unsqueeze(0).cuda())
            # print(out.squeeze().detach().clone().shape)
            all_data[i] = out.squeeze().detach().clone()
        
        all_labels[i] = torch.tensor(label)
        
        # break
        
all_data = PCA(280).fit_transform(all_data)
print(all_data.shape)

# print(all_data[0])

import scipy.io

# 保存NumPy数组到MAT文件
scipy.io.savemat('/home/ubuntu12/wwc/ACPNN_AAAI17/data/Emotion6_vgg16_hjy.mat', 
    {
        'features': all_data,
        'labels': all_labels,
    }
)