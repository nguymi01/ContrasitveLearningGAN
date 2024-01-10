# '''
# Data Loader for CelebA Data, to be shared by all team members.
# '''


import os
import PIL
from PIL import Image
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def show_image_ongrid(images):
    n = images.shape[0] // 4
    plt.figure(figsize=(20, 20))
    plt.imshow(np.vstack([
        np.hstack([images[4 * i+j] for j in range(4)])
        for i in range(n)
    ]))
    plt.axis('off')
    plt.show()

def load_image(target_fname, img_resolution=256, transform=None):
    # Load target image.
    target_pil = PIL.Image.open(target_fname)
    if transform is None:
        target_pil = target_pil.convert('RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((img_resolution, img_resolution), PIL.Image.LANCZOS)
        target_pil = np.array(target_pil, dtype=np.uint8)
    else:
        target_pil = transform(target_pil)
    return target_pil

def load_image_batch(_image_fnames, img_resolution=256, transform=None,
                     dataset_path='', batch_size=16, batch_num=0, randomize=False, seed=0):
    if randomize:
        np.random.seed(seed)
        _target_fnames = np.random.permutation(_image_fnames)[batch_size*batch_num:batch_size*(batch_num+1)]
    elif len(_image_fnames) > batch_size:
        _target_fnames = _image_fnames[batch_size*batch_num:batch_size*(batch_num+1)]
    else:
        _target_fnames = _image_fnames
        
    target_batch = np.array([
        load_image(os.path.join(dataset_path, _fnames), img_resolution=img_resolution, transform=transform)
        for _fnames in _target_fnames
    ])
    return target_batch

def synthesize_images(G, _embeddings, device=torch.device('cuda')):
    _embeddings = torch.tensor(_embeddings)
    if _embeddings.dim() == 2:
        _embeddings = _embeddings.unsqueeze(1).repeat([1, G.mapping.num_ws, 1])
    synth_images = G.synthesis(_embeddings.to(device), noise_mode='const')
    synth_images = (synth_images + 1) * (255/2)
    return synth_images

class DataLoader():
    def __init__(
        self,
        dataset_path       = '/projectnb/cs585bp/585-Project-nguymi01/data/',
        image_path         = '/projectnb/cs585bp/585-Project-nguymi01/data/img_align_celeba/img_align_celeba/',
        attr_data_filename = 'list_attr_celeba.csv',
        part_data_filename = 'list_eval_partition.csv',
        attribute_keys     = None,
        ignore_partition   = False,
        pretrained_embeddings_filename = None,
        image_transform = None
    ):
        self.dataset_path    = dataset_path
        self.image_path      = image_path
        self.image_transform = image_transform
        self.state = {
            'partition': -1,
            'queue': None
        }
        
        if image_path is None:
            _all_fnames = [
                os.path.relpath(os.path.join(root, fname), start=dataset_path) 
                for root, _dirs, files in os.walk(dataset_path) for fname in files
            ]
            self.image_fnames = sorted(fname for fname in _all_fnames if os.path.splitext(fname)[-1].lower() == '.jpg')
            self.image_path = self.dataset_path
        else:
            self.image_fnames = sorted(fname for fname in os.listdir(image_path) if os.path.splitext(fname)[-1].lower() == '.jpg')
            
        self.attr_data = pd.read_csv(os.path.join(dataset_path, attr_data_filename))
        self.partitioned = False
        if part_data_filename is not None:
            part_data = pd.read_csv(os.path.join(dataset_path, part_data_filename))
            self.attr_data = self.attr_data.merge(part_data, on='image_id')
            self.partitioned = True
            self.state['partition'] = 0
            
        if pretrained_embeddings_filename is not None:
            with open(pretrained_embeddings_filename, 'rb') as f:
                self.pretrained_embedings = np.load(f)
        
        self.set_keys(keys=attribute_keys, ignore_partition=ignore_partition)
            
    def get_partition(self, partition):
        assert partition in [0, 1, 2] and self.partitioned  # train, valid, test
        return self.attr_data[self.attr_data.partition == partition]
    
    def set_keys(self, keys=None, ignore_partition=False):
        if keys is None:
            if self.partitioned:
                self.keys = list(self.attr_data.columns)[1:-1]
            else:
                self.keys = list(self.attr_data.columns)[1:]
        else:
            self.keys = keys
                
        self.key_inds = np.array([list(self.attr_data.columns)[1:].index(k) for k in self.keys])
        self.attr_grouped_by_keys = self.attr_data.groupby(by=self.keys)
        if self.partitioned and ~ignore_partition:
            ## Making sure that at least a pair exists in training set
            self.sample_labels = [
                [_label, _group] for _label, _group in self.attr_grouped_by_keys
                if _group[_group.partition == 0].shape[0] > 2
            ]
        else:
            self.sample_labels = [
                [_label, _group]
                for _label, _group in self.attr_grouped_by_keys
                if _group.shape[0] > 2
            ]

    def get_positive_pairs(self, batch_size=16, partition=0, ignore_partition=False):
        batch_labels = []
        batch_attr = []
        for i in np.random.choice(len(self.sample_labels), batch_size, replace=False):
            batch_labels.append(self.sample_labels[i][0])
            if self.partitioned and ~ignore_partition:
                batch_attr.append(self.sample_labels[i][1][self.sample_labels[i][1].partition == partition].sample(2))
            else:
                batch_attr.append(self.sample_labels[i][1].sample(2))
                
        batch_labels = np.array(batch_labels) == 1
        
        batch_attr = pd.concat(batch_attr)
        batch_attr = [
            batch_attr.iloc[np.arange(0, 2*batch_size, 2)],
            batch_attr.iloc[np.arange(1, 2*batch_size, 2)]
        ]

        batch_images = [
            load_image_batch(batch_attr[i].image_id.values,
                             dataset_path=self.image_path,
                             transform=self.image_transform,
                             batch_size=batch_size)
            for i in range(2)
        ]
        
        return batch_images, batch_labels, batch_attr
    
    def get_batch(self, batch_size, refresh=False, permutate=True, with_pretrained_embeddings=False):
        if refresh or self.state['queue'] is None:
            print('...refreshing queue...')
            if with_pretrained_embeddings:
                self.state['queue'] = np.arange(self.pretrained_embedings.shape[0])
            elif self.state['partition'] == -1:
                self.state['queue'] = np.array(list(self.attr_data.index))
            else:
                self.state['queue'] = np.array(list(self.get_partition(0).index))
            if permutate:
                self.state['queue'] = np.random.permutation(self.state['queue'])
            
        elif self.state['queue'] is not None and len(self.state['queue']) < batch_size:
            print('...end of queue...')
            self.state['current'] = None
            self.state['queue'] = None
            return None, None, None
        
        
        self.state['current'], self.state['queue'] = self.state['queue'][:batch_size], self.state['queue'][batch_size:]
        batch_inds = self.state['current']
        batch_attr = self.attr_data.loc[batch_inds]
        batch_images = load_image_batch(batch_attr.image_id.values,
                                        dataset_path=self.image_path,
                                        transform=self.image_transform,
                                        batch_size=batch_size)
        batch_labels = batch_attr[self.keys].values == 1
        
        if with_pretrained_embeddings:
            batch_embs = self.pretrained_embedings[batch_inds]
            return batch_images, batch_labels, batch_embs
        return batch_images, batch_labels, batch_attr
