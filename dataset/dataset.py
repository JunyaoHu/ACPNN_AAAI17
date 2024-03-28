import copy
import os
from PIL import Image
from torch.utils.data import Dataset

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
    
class SingleLabelClassificationEmotionDataset(EmotionDataset):
    def __init__(self, path, type, label_dict, transform=None):
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
                
class MultiLabelClassificationEmotionDataset(EmotionDataset):
    def __init__(self, path, type, label_dict, transform=None):
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
    def __init__(self, path, type, label_dict, transform=None):
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
            vote_num = sum(label)
            label = [l / vote_num for l in label]
            
            self.path.append(img_path)
            self.label.append(label)
    
if __name__ == "__main__":
    import mediapy as media
    import einops
    from torchvision import transforms
    
    # path = "/home/ubuntu/hjy/data/FI"
    # path = "/home/ubuntu/hjy/data/UnBiasedEmo"
    path = "/home/ubuntu/hjy/data/EmotionROI"
    # path = "/home/ubuntu/hjy/data/Emotion6"
    # path = "/home/ubuntu/hjy/data/Twitter_LDL"
    # path = "/home/ubuntu/hjy/data/Flickr_LDL"
    
    type = "train"
    # type = "test"
    
    # label_dict = FI_label_dict
    # label_dict = Unbiased_label_dict
    label_dict = EmotionROI_label_dict
    # label_dict = Emotion6_label_dict
    # label_dict = LDL_label_dict
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(
        # mean=[0.485, 0.456, 0.406], 
        # std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SingleLabelClassificationEmotionDataset(path, type, label_dict, transform)
    # dataset = MultiLabelDistributionEmotionDataset(path, type, label_dict, transform)
    
    print(len(dataset))
    
    data  = dataset[1137]
    img, label = data
    print(img.shape, img.min(), img.max(), label)
    
    img = einops.rearrange(img, "c h w -> h w c")
    media.show_image(img, title=str(label))