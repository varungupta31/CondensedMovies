""" MoviePlots dataset module.

This code is loosely based from the collaborative-experts dataloaders:
https://github.com/albanie/collaborative-experts/tree/master/data_loader

"""
import os
from os.path import join as osj
import ast
import ipdb
import itertools

import pandas as pd
import numpy as np
import torch
import nltk
import pdb
from torch.utils.data import Dataset
from utils.util import memcache, memory_summary


class MovieClips(Dataset):
    
    def __init__(self, data_dir, metadata_dir, label, experts_used, experts, max_tokens, split='train'):
        # @R Most of these arguments are picked up from the json config.
        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.experts_used = [expert for expert in experts_used if experts_used[expert]]
        self.label = label
        if self.label not in experts_used:
            raise ValueError('Label expert must be used.')
        self.experts = experts
        self.expert_dims = self._expert_dims()

        # @R TODO: Understand the functionality for the 'max_tokens'
        self.max_tokens = max_tokens
        self.split = split
        #This parameter decides if the self.data[clip] will be overridden or not. Overriding is required when comparing with the cmdformer implementation.
        self.running_as_baseline = False
        self._load_metadata()
        self._load_data()

    def _load_metadata(self):
        """
        @R
        - Creates a data dict.
        - Filters the values based on the dataset split (train/test).
        - Cleans the data (NaN and duplicate removal).
        - Sets or Updates the data variable, essentially a dict of metadata CSVs.
        """
        # import pdb; pdb.set_trace()

        # @R Create a big dictionary indexed by the following key items.
        data = {
            'movies': pd.read_csv(osj(self.metadata_dir, 'movies.csv')).set_index('imdbid'),
            'casts': pd.read_csv(osj(self.metadata_dir, 'casts.csv')).set_index('imdbid'),
            'clips': pd.read_csv(osj(self.metadata_dir, 'clips.csv')).set_index('videoid').groupby('imdbid', group_keys=False).apply(lambda x: x.sort_values('clip_idx')),
            'descs': pd.read_csv(osj(self.metadata_dir, 'descriptions.csv')).set_index('videoid'),

        }
        # filter by split {'train', 'val', 'test'}
        split_data = pd.read_csv(osj(self.metadata_dir, 'split.csv')).set_index('imdbid')
        if self.split == 'train_val':
            ids = split_data[split_data['split'].isin(['train', 'val'])].index
        else:
            # @R ids store the imbdb id of the movies belonging to the defined split.
            ids = split_data[split_data['split'] == self.split].index    

        for key in data:
            if 'imdbid' in data[key]:
                filter = data[key]['imdbid'].isin(ids)
            else:
                filter = data[key].index.isin(ids)
            # @R TODO: Need to understand what passing a boolean list to a dataframe does. Expected is that it returns a subset of the filtered/True columns.
            data[key] = data[key][filter]


        # Remove inappropriate data
        #empty_clips = pd.read_csv(osj(self.metadata_dir, 'empty_vids.csv')).set_index('videoid')
        #data['clips'] = data['clips'][~data['clips'].index.isin(empty_clips.index)]
        # duplicated descriptions are probably errors by the channel

        data['descs'].dropna(subset=['description'], inplace=True)
        data['descs'].drop_duplicates(subset=['description'], keep=False, inplace=True)

        # remove clips without descriptions (since this is supervised)...
        if self.label == 'description':
            data['clips'] = data['clips'][data['clips'].index.isin(data['descs'].index)]
        elif self.label == 'plot':
            # @R TODO: data['plots'] does not exist!
            data['clips'] = data['clips'][data['clips']['imdbid'].isin(data['plots'].index)]
        else:
            raise NotImplementedError('Change data removal technique to remove clips without...')

        self.data = data

    def _load_data(self):
        # @R Updating the self.data variable. Creating a dictionary with keys as the expert and values as the loaded numpy objects.
        # Then creating a 'set' of clips to identify all the clips that are used by / covered under atleast one expert. 
        
        self.expert_data = {}
        for expert in self.experts_used:
            if expert != 'context':
                data_pth = osj(self.data_dir, 'features', self.experts[expert])
                # @R 'expert_data' -> A dictionary keyed by the corresponding expert, and the value is the loaded .npy file object.
                # @R Load the npy files and apply L2-Norm over the features (currently NOT/False applying)  
                
                print(f"{expert}: {data_pth}")
                self.expert_data[expert] = memcache(data_pth)
                memory_summary()
        
        # @R So far, all the expert's numpy files (.npy) files have been read and L2-Normed (False) applied, if specified. 
        # For exp5.json -> dict_keys(['description', 'face', 'rgb', 'scene', 'video'])

        # @R Appending the clips names contained each experts npy. TODO: why?
        
        clips_with_data = []
        for expert in self.expert_data:
            if expert != 'description' and expert != 'label':
                # @R the self.expert_data[expert].keys() contains the televant clips IDs.
                clips_with_data += self.expert_data[expert].keys()

        # debugging (input random tensors)
        random = False
        if random:
            for expert in self.expert_data:
                for videoid in self.expert_data[expert]:
                    self.expert_data[expert][videoid] = np.random.randn(*self.expert_data[expert][videoid].shape)

        # debugging (input zero tensors)
        zeros = False
        if zeros:
            for expert in self.expert_data:
                for videoid in self.expert_data[expert]:
                    self.expert_data[expert][videoid] = np.zeros(self.expert_data[expert][videoid].shape)

        # @R Getting rid of the duplicates --> End up with a 34584 length list.
        clips_with_data = set(clips_with_data)

        #sanity check
        #pdb.set_trace()
        #if not self.data['clips'].index.isin(clips_with_data).all():
        #    print(self.data['clips'][~self.data['clips'].index.isin(clips_with_data)].index)
        #    raise NotImplementedError

        # @R Before len(self.data['clips']) = 24098 || After --> 24035
        # @R Most clips covered under some feature or the other...
        self.data['clips'] = self.data['clips'][self.data['clips'].index.isin(clips_with_data)]
        
        # """
        # Modifying the self.data[clips] depending if the .csv files are specified or not, based on the split! [Making the data for cmdformer and baseline consistent]
        # """

        # if self.running_as_baseline:
        #     if self.split == "train":
        #         train_split = pd.read_csv("/ssd_scratch/cvit/varun/cmdformer_data/cmd_format/metadata/baseline_splits/train.csv", index_col='videoid')
        #         train_desc =  pd.read_csv("/ssd_scratch/cvit/varun/cmdformer_data/cmd_format/metadata/baseline_splits/desc_train.csv", index_col='videoid')
        #         self.data["clips"] = train_split
        #         self.data["descs"] = train_desc
        #     elif self.split == "val":
        #         val_split = pd.read_csv("/ssd_scratch/cvit/varun/cmdformer_data/cmd_format/metadata/baseline_splits/val.csv", index_col='videoid')
        #         val_desc =  pd.read_csv("/ssd_scratch/cvit/varun/cmdformer_data/cmd_format/metadata/baseline_splits/desc_val.csv", index_col='videoid')
        #         self.data["clips"] = val_split
        #         self.data["descs"] = val_desc
        #     elif self.split == "test":
        #         test_split = pd.read_csv("/ssd_scratch/cvit/varun/cmdformer_data/cmd_format/metadata/baseline_splits/test.csv", index_col='videoid')
        #         test_desc =  pd.read_csv("/ssd_scratch/cvit/varun/cmdformer_data/cmd_format/metadata/baseline_splits/desc_test.csv", index_col='videoid')
        #         self.data["clips"] = test_split
        #         self.data["descs"] = test_desc

        print(f'{self.split} size: {len(self.data["clips"])} clips')

    def __len__(self):
        return len(self.data['clips'])
        # return 1280


    def __getitem__(self, item):
        # @R The self.data is a dictionary, with the 'experts' as the keys.
        # @R self.data.keys --> dict_keys(['movies', 'casts', 'clips', 'descs'])
        # @R self.data['movies'] is just the metadata (dataframe).
        # @R self.data['movies'].columns --> Index(['title', 'year'], dtype='object')

        #How many clips BEFORE and AFTER to be considered
        p_context_window = 1
        f_context_window = 1
        

        imdbid = self.data['clips'].iloc[item].imdbid
        clip_idx = self.data['clips'].iloc[item].clip_idx
        
        
        #one clip id is stored here.
        #videoid = self.data['clips'].iloc[item].name

        #videoids will contain all the videos ids depending on the specified context window.
        
        videoids = []

        #Finding all the videoids

        # for i in range(1, context_window+1):
        #     try:
        #         videoids.append(self.data['clips'].loc[(self.data['clips']['clip_idx']==clip_idx-i) & (self.data['clips']['imdbid']==imdbid)].index.item())
        #     except:
        #         #this will happen when the previous clip or the next clip doesn't exist.
        #         #in that case, a garbage id is pushed and the missing conditional is triggered.
        #         videoids.append('theofficeishyped')
            
        #     try:
        #         videoids.append(self.data['clips'].loc[(self.data['clips']['clip_idx']==clip_idx+i) & (self.data['clips']['imdbid']==imdbid)].index.item())
        #     except:
        #         videoids.append('theofficeishyped')

        for i in range(1, p_context_window+1):
            try:
                videoids.append(self.data['clips'].loc[(self.data['clips']['clip_idx']==clip_idx-i) & (self.data['clips']['imdbid']==imdbid)].index.item())
            except:
                #this will happen when the previous clip or the next clip doesn't exist.
                #in that case, a garbage id is pushed and the missing conditional is triggered.
                videoids.append('theofficeishyped')
        
        for j in range(1, f_context_window+1):
            try:
                videoids.append(self.data['clips'].loc[(self.data['clips']['clip_idx']==clip_idx+j) & (self.data['clips']['imdbid']==imdbid)].index.item())
            except:
                videoids.append('theofficeishyped')

        videoids.append(self.data['clips'].iloc[item].name)
        
      
        data_list = []

        for videoid in videoids:
            data = {}    
            for expert in self.experts_used:            
                packet = self._get_expert_ftr(expert, videoid)
                if expert == self.label:
                    #data['label'] = packet
                    pass
                else:
                    data[expert] = packet
            data_list.append(data)
        
        data = sum_nested_dictionaries(data_list)

        #regardless of the context window, the description mimicked for the query in our case, will always be 1.
        data['label'] = self._get_expert_ftr('description', self.data['clips'].iloc[item].name)
        
        id = {'imdbid': self.data['clips'].loc[videoid]['imdbid'], 'videoid': videoid}
        return data, id
    
    

    def _get_expert_ftr(self, expert, videoid, context=False):

        packet = {}

        if expert == 'plot':
            videoid = self.data['clips'].loc[videoid]['imdbid']  # TODO: maybe this breaks for clips with no imdbid?

        if videoid not in self.expert_data[expert]:
            missing = True
            some_entry = list(self.expert_data[expert].keys())[0]
            ftr = np.zeros_like(self.expert_data[expert][some_entry])
        else:
            missing = False

            #@R Type --> Numpy array
            ftr = self.expert_data[expert][videoid]
            #if context:
            #    ftr = np.zeros(ftr.shape)
                #ftr = np.random.randn(*ftr.shape)


        ftr = torch.from_numpy(ftr)
        ftr = ftr.float()
        if len(ftr.shape) == 1:
            pass
        elif len(ftr.shape) == 2:
            ftr, n_tokens = self._pad_to_max_tokens(ftr, expert)
            packet['n_tokens'] = torch.Tensor([n_tokens])
            # @R packet --> {'n_tokens': tensor([20.])}
        else:
            raise ValueError
        
        packet['ftr'] = ftr.unsqueeze(dim=0)
        packet['missing'] = torch.Tensor([missing])
        return packet

    def _pad_to_max_tokens(self, array, expert):
        """
        @R
        If the input features from the given expert are within (or equal) to the max_tokens for the given expert
        as specified in the configs, this won't change anything. 
        If however, the length of the token in the loaded sample exceed the max length, it simply crops the top max_tokens features.

        """
        # @R TODO: Understand what exactly the max tokens imply.
        n_tokens, dim = array.shape
        if n_tokens >= self.max_tokens[expert]:
            res = array[:self.max_tokens[expert]]
            n_tokens = self.max_tokens[expert]
        else:
            res = torch.zeros((self.max_tokens[expert], dim))
            res[:n_tokens] = array
        return res, n_tokens

    def _characters_txt(self, texts, clean_cast):
        raise NotImplementedError

    def _clean_cast(self, cast):
        for actor in cast:
            char = cast[actor]
            char = char.replace('(voice)', '')
            char = char.strip()
            char = [c.strip() for c in char.split('/')]  # deals with one-many actor
            cast[actor] = char  # char here is a list
        return cast

    def _expert_dims(self):
        expert_dims = {
            'BERT': 1024,
            'I3D': 1024,
            'DenseNet-161': 2208,
            'SE-ResNet-154': 2048,
            'S3DG': 1024,
            'SE-ResNet-50': 256,
            '': None
        }
        ftrs_dim = {}
        for key in self.experts:
            arch = self.experts[key].split('/')[0]
            if arch not in expert_dims and arch != "":
                raise ValueError('Expert not found in dims dict, please update')
            ftrs_dim[key] = expert_dims[arch]

        return ftrs_dim

def sum_nested_dictionaries(nested_dictionaries):
    result = {}
    for dictionary in nested_dictionaries:
        for key, value in dictionary.items():
            if isinstance(value, dict):
                if key not in result:
                    result[key] = sum_nested_dictionaries([value])
                else:
                    sub_dict = sum_nested_dictionaries([value])
                    result[key] = cat_dicts(result[key], sub_dict)
            else:
                if key not in result:
                    result[key] = value
                else:
                    result[key] = torch.cat((result[key], value))
    return result


def cat_dicts(dict1, dict2):
    cat_dict = dict1.copy()
    for key, value in dict2.items():
        if key in cat_dict:
            if isinstance(value, dict) and isinstance(cat_dict[key], dict):
                cat_dict[key] = cat_dicts(cat_dict[key], value)
            else:
                cat_dict[key] = torch.cat((cat_dict[key], value))
        else:
            cat_dict[key] = value
    return cat_dict
