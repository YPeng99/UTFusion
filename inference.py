import os.path

import torch
from torch.utils.data import DataLoader

from data_loader import InferenceDataset
from models.UTFusion import UTFusion
from utils import  denormalizer,rgb2ycbcr,ycbcr2rgb,fuse_seq_cb_cr,fuse_cb_cr
from torchvision import transforms





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = './results'
    model_name = 'UTFusion'
    # Multi-exposure settings 1 Other settings 0
    fuse_scheme = torch.tensor([1]).to(device)
    model = UTFusion()
    model.load_state_dict(torch.load("logs/UTFusion.ckpt")['state_dict'])
    model.to(device)
    model.eval()

    dataset = InferenceDataset()

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    normalize = transforms.Normalize(mean=[0.5,],std=[0.5,])
    denorm = denormalizer([0.5,],[0.5,])
    to_pil = transforms.ToPILImage()


    with torch.inference_mode():
        for img_list, img_name in data_loader:
            Y = []
            Cb = []
            Cr = []
            for img in img_list:
                img = img.to(device)
                _y,_cb,_cr = rgb2ycbcr(img)
                _y = normalize(_y)
                Y.append(_y)
                Cb.append(_cb)
                Cr.append(_cr)
            Y = model(Y, fuse_scheme)
            Y = denorm(Y)
            Cb,Cr = fuse_seq_cb_cr(Cb,Cr)
            fused = ycbcr2rgb(Y,Cb,Cr)
            fused = torch.clamp(fused, 0, 1)
            fused = to_pil(fused[0])
            fused.save(os.path.join(save_path,img_name[0]))
            print(os.path.join(save_path,img_name[0]))


