import copy
import pickle

import torch, esm, random, os, json
import numpy as np
from Bio import SeqIO

import pandas as pd
import ast


import numpy as np

def parse_numpy_array_string(s):
    # Remove newlines, then split the array rows
    s = s.replace('\n', ' ').replace('[', '').replace(']', '')
    # Convert string of floats into a 1D array
    float_array = np.fromstring(s, sep=' ')
    # Infer shape (e.g. 100 x 4)
    num_rows = int(len(float_array) / 4)
    return float_array.reshape((num_rows, 4))

def parse_numpy_array_string_protein(s):
    # Remove newlines, then split the array rows
    s = s.replace('\n', ' ').replace('[', '').replace(']', '')
    # Convert string of floats into a 1D array
    float_array = np.fromstring(s, sep=' ')
    # Infer shape (e.g. 100 x 4)
    num_rows = int(len(float_array) / 538)
    return float_array.reshape((num_rows, 538))

class EnhancerDataset(torch.utils.data.Dataset):
    def __init__(self, args, split='train', data="original"):
        self.data = data
        if data == "original":
            all_data = pickle.load(open(f'data/the_code/General/data/Deep{"MEL2" if args.mel_enhancer else "FlyBrain"}_data.pkl', 'rb'))
            self.seqs = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'{split}_data'])), dim=-1)
            self.clss = torch.argmax(torch.from_numpy(copy.deepcopy(all_data[f'y_{split}'])), dim=-1)
            self.num_cls = all_data[f'y_{split}'].shape[-1]
        elif data == "aptamer_utexas":
            all_data = pd.read_csv("data/aptamer_texas/aptamer_seq_utexas.csv")
            if split == 'train':
                start = 0
                end = 1200
            else:
                start = 1200
                end = len(all_data)

            using_data =  all_data.iloc[start:end].copy()

            # Safely evaluate the strings into lists, then convert to numpy arrays
            using_data['onehot_padded'] = using_data['onehot_padded'].apply(parse_numpy_array_string)   
            using_data['onehot_padded'] = using_data['onehot_padded'].apply(np.array)

            using_data['protein_onehot'] = using_data['protein_onehot'].apply(parse_numpy_array_string_protein) 
            using_data['protein_onehot'] = using_data['protein_onehot'].apply(np.array) 

            self.seqs = torch.argmax(torch.tensor(np.stack(copy.deepcopy(using_data['onehot_padded'].values)), dtype=torch.float32), dim=-1)
            self.clss = torch.argmax(torch.tensor(np.stack(copy.deepcopy(using_data['protein_onehot'].values)), dtype=torch.float32), dim=-1).squeeze(-1)
            self.num_cls = 538

        elif data == "aptamer_trans":

            all_data = pickle.load(open("data/aptamer_aptatrans/aptatrans_data.pkl", 'rb'))
            
            if split == 'train':
                pass
            elif split == 'valid':
                pass
            elif split == 'test':
                pass
            else:
                raise ValueError('')


        else: 
            raise ValueError("File: dataset.py ; Line 33 ; Gave an invalid argument for data object")

        self.alphabet_size = 4

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if self.data == "original":
            return self.seqs[idx], self.clss[idx]
        elif self.data == "aptamer":
            return self.seqs[idx]
        else:
            raise ValueError("File: dataset.py ; Line 74 ; Gave an invalid argument for data object")


class TwoClassOverfitDataset(torch.utils.data.IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim
        self.num_cls = 2

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'overfit_dataset.pt'))
            self.data_class1 = distribution_dict['data_class1']
            self.data_class2 = distribution_dict['data_class2']
        else:
            self.data_class1 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
            self.data_class2 = torch.stack([torch.from_numpy(np.random.choice(np.arange(self.alphabet_size), size=args.toy_seq_len, replace=True)) for _ in range(args.toy_num_seq)])
            distribution_dict = {'data_class1': self.data_class1, 'data_class2': self.data_class2}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'overfit_dataset.pt'))

    def __len__(self):
        return 10000000000

    def __iter__(self):
        while True:
            if np.random.rand() < 0.5:
                yield self.data_class1[np.random.choice(np.arange(len(self.data_class1)))], torch.tensor([0])
            else:
                yield self.data_class2[np.random.choice(np.arange(len(self.data_class2)))], torch.tensor([1])

class ToyDataset(torch.utils.data.IterableDataset):
    def __init__(self, args):
        super().__init__()
        self.num_cls = args.toy_num_cls
        self.seq_len = args.toy_seq_len
        self.alphabet_size = args.toy_simplex_dim

        if args.cls_ckpt is not None:
            distribution_dict = torch.load(os.path.join(os.path.dirname(args.cls_ckpt), 'toy_distribution_dict.pt'))
            self.probs = distribution_dict['probs']
            self.class_probs = distribution_dict['class_probs']
        else:
            self.probs = torch.softmax(torch.rand((self.num_cls, self.seq_len, self.alphabet_size)), dim=2)
            self.class_probs = torch.ones(self.num_cls)
            if self.num_cls > 1:
                self.class_probs = self.class_probs * 1 / 2 / (self.num_cls - 1)
                self.class_probs[0] = 1 / 2
            assert self.class_probs.sum() == 1

            distribution_dict = {'probs': self.probs, 'class_probs': self.class_probs}
        torch.save(distribution_dict, os.path.join(os.environ["MODEL_DIR"], 'toy_distribution_dict.pt' ))

    def __len__(self):
        return 10000000000
    def __iter__(self):
        while True:
            cls = np.random.choice(a=self.num_cls,size=1,p=self.class_probs)
            seq = []
            for i in range(self.seq_len):
                seq.append(torch.multinomial(replacement=True,num_samples=1,input=self.probs[cls,i,:]))
            yield torch.tensor(seq), cls

