import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import dataloader

def draw_tensor_image(img_tensor,img_name):
    # 将张量转换为PIL图像
    to_pil = transforms.ToPILImage()
    img = to_pil(img_tensor)
    img.save("../resources/"+img_name+".png")
    print("{img_name} has saved".format(img_name=img_name))
    
if __name__ == "__main__":
    
    train_loader, val_loader = dataloader.get_dataloader()
    for i, (x, y) in enumerate(train_loader):
        draw_tensor_image(x[0],str(i))    
        if i>3:
            break