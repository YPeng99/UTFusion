import os.path

import torch
from torch.utils.data import DataLoader

from data_loader import CTMRIDataset, PETMRIDataset, LytroDataset, MEFBDataset, MSRSDataset
from models.P2Fusion import P2Fusion
from utils import  denormalizer,rgb2ycbcr,ycbcr2rgb,fuse_cb_cr
from torchvision import transforms






if __name__ == '__main__':
    save_path = './results'
    model_name = 'P2Fusion'
    model = P2Fusion()
    model.load_state_dict(torch.load("./logs/SwitchFusion.ckpt")['state_dict'])
    model.cuda()
    model.eval()

    ct_mri_dataset = CTMRIDataset()
    pet_mri_dataset = PETMRIDataset()
    lytro_dataset = LytroDataset()
    mefb_dataset = MEFBDataset()
    msrsd_dataset = MSRSDataset()

    ct_mri_dataloader = DataLoader(ct_mri_dataset, batch_size=1, shuffle=False)
    pet_mri_dataloader = DataLoader(pet_mri_dataset, batch_size=1, shuffle=False)
    lytro_dataloader = DataLoader(lytro_dataset, batch_size=1, shuffle=False)
    mefb_dataloader = DataLoader(mefb_dataset, batch_size=1, shuffle=False)
    msrsd_dataloader = DataLoader(msrsd_dataset, batch_size=1, shuffle=False)

    normalize = transforms.Normalize(mean=[0.5,],std=[0.5,])
    denorm = denormalizer([0.5,],[0.5,])
    to_pil = transforms.ToPILImage()

    dataset_list = ['ct-mri', 'pet-mri', 'lytro', 'mefb', 'msrsd']
    dataloader_list = [ct_mri_dataloader, pet_mri_dataloader, lytro_dataloader, mefb_dataloader, msrsd_dataloader]

    with torch.inference_mode():
        for i, (dataloader, n) in enumerate(zip(dataloader_list, dataset_list)):
            for j, (S1, S2, fuse_scheme, file_name) in enumerate(dataloader):
                S1 = S1.cuda()
                S2 = S2.cuda()
                fuse_scheme = fuse_scheme.cuda()
                if 'ct' in n:
                    Y_1 = normalize(S1)
                    Y_2 = normalize(S2)
                    path = os.path.join(save_path,model_name+'_'+'CT-MRI',file_name[0])
                elif 'pet' in n:
                    Y_1, Cb, Cr = rgb2ycbcr(S1)
                    Y_1 = normalize(Y_1)
                    Y_2 = normalize(S2)
                    path = os.path.join(save_path, model_name+'_'+'PET-MRI', file_name[0])
                elif 'msrs' in n:
                    Y_1 = normalize(S1)
                    Y_2, Cb, Cr = rgb2ycbcr(S2)
                    Y_2 = normalize(Y_2)
                    path = os.path.join(save_path, model_name+'_'+'MSRS', file_name[0])
                elif 'lytro' in n:
                    Y_1, Cb_1, Cr_1 = rgb2ycbcr(S1)
                    Y_2, Cb_2, Cr_2 = rgb2ycbcr(S2)
                    Y_1 = normalize(Y_1)
                    Y_2 = normalize(Y_2)
                    Cb, Cr = fuse_cb_cr(Cb_1, Cr_1, Cb_2, Cr_2,tao=0.5)
                    path = os.path.join(save_path, model_name+'_'+'Lytro', file_name[0])
                elif 'mefb' in n:
                    Y_1, Cb_1, Cr_1 = rgb2ycbcr(S1)
                    Y_2, Cb_2, Cr_2 = rgb2ycbcr(S2)
                    Y_1 = normalize(Y_1)
                    Y_2 = normalize(Y_2)
                    Cb, Cr = fuse_cb_cr(Cb_1, Cr_1, Cb_2, Cr_2,tao=0.5)
                    path = os.path.join(save_path, model_name+'_'+'MEFB', file_name[0])

                if 'ct' in n:
                    fused = model((Y_1, Y_2), fuse_scheme)
                    fused = denorm(fused)
                else:
                    Y = model((Y_1, Y_2), fuse_scheme)
                    Y = denorm(Y)
                    fused = ycbcr2rgb(Y, Cb, Cr)
                fused = torch.clamp(fused, 0, 1)
                fused = to_pil(fused[0])
                fused.save(path)
                print(path)
