import torch as th
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision as tv
import random
import os
import numpy as np
import drawer

seed =42

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    
def get_dataloader(batch_size=32, num_workers=8, seed=42):
    seed_torch(seed)
    transform = tv.transforms.Compose([
        tv.transforms.Resize((32, 32)),
        tv.transforms.ToTensor()
    ])
    
    mnist_train = tv.datasets.MNIST(root='./database', download=True, transform=transform)
    mnist_test = tv.datasets.MNIST(root='./database', train=False, download=True, transform=transform)
    
    generator=th.Generator().manual_seed(seed)
    train_dataset,val_dataset = random_split(mnist_train,[50000,10000],generator=generator)
    
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,persistent_workers=True)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_dataloader()
    for i, (x, y) in enumerate(train_loader):
        print("============================")
        print("x的形状和y的形状：")
        print(x.shape, y.shape)
        print("============================")
        print("x数据具体的数值：")
        print(x[0])
        drawer.draw_tensor_image(x[0])
        print("============================")
        print("y标签具体的数值：")
        print(y)
        print("============================")
        
        if i > 2:
            break
        
# 在这段代码中，train_loader是一个DataLoader对象，它负责管理数据的加载。DataLoader对象的batch_size参数设置为32，
# 这意味着每次从train_loader中获取数据时，它会返回32个样本。当你在主程序中遍历train_loader时，每次迭代都会返回一
# 个批次的数据，这个批次包含32个样本。这就是为什么x.shape的第一个维度是32的原因。x是一个包含32个图像的批次，每个图
# 像都是一个32x32的张量（由于tv.transforms.Resize((32, 32))）。y是对应的32个标签。
# 总结一下，train_loader的维度是32，是因为在创建DataLoader时，我们设置了batch_size=32，这意味着每次从train_loader
# 中获取数据时，它会返回32个样本。