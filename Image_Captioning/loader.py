import os
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.stoi = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
        self.itos = {0:'<PAD>', 1: '<SOS>', 2:'<EOS>', 3:'<UNK>'}
    
    def __len__(self):
        return len(self.itos)
    
    def build_vocabulary(self):
        hindi_alphabets = [chr(alpha) for alpha in range(2304, 2432)]
        hindi_alphabet_size = len(hindi_alphabets)
        for index, alpha in enumerate(hindi_alphabets):
            if alpha not in self.stoi:
                self.stoi[alpha] = index+1
                self.itos[index+1] = alpha
        # return self.stoi

    def numericalized(self, word):
        gt_rep = torch.zeros([len(word), 1], dtype=torch.long)
        for letter_index, letter in enumerate(word):
            pos = self.stoi[letter]
            gt_rep[letter_index][0] = pos
        #gt_rep[letter_index+1][0] = self.stoi[pad_char]
        return gt_rep


class FlickerDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_excel(captions_file)
        self.caption_file = captions_file
        self.transform = transform

        self.imgs = self.df["filename"]
        self.caption = self.df["word"]

        #Initialize vocab and build vocab

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary()
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        caption = self.caption[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        
        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalized(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)
    

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets
    


def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True
):
    dataset = FlickerDataset(root_folder, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx)
    )
    return loader, dataset


def main():
    data_transform = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]
    )
    dataloader = get_loader("Data/Images/", annotation_file="Data/path_to_output_excel_file.xlsx", transform=data_transform)
    for idx, (imgs, captions) in enumerate(dataloader):
        print(imgs.shape)
        print(captions.shape)
        print(captions)
        break


if __name__ == "__main__":
    main()

