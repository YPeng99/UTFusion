import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms





class InferenceDataset(Dataset):
    def __init__(self, root="./datasets"):
        super().__init__()
        self.root = root
        self.file_path = os.listdir(self.root)
        self.file_list = os.listdir(os.path.join(self.root,self.file_path[0]))

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_name = self.file_list[index]
        img_list = []
        for p in self.file_path:
            img_path = os.path.join(self.root,p,img_name)
            img = Image.open(img_path)
            img = self.to_tensor(img)
            img_list.append(img)
        return img_list,img_name



if __name__ == '__main__':
    dataset = InferenceDataset()
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(data_loader.__len__())
    for img_list,img_name in data_loader:
        print(len(img_list))
        print(img_name)
        print(type(img_list))


