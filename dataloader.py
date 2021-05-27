#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, nrelation, count, true_head, true_tail, entity_dict):
        self.len = len(triples['head'])
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.count = count
        self.true_head = true_head
        self.true_tail = true_tail
        self.entity_dict = entity_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation = self.triples['head'][idx], self.triples['relation'][idx]
        head_type = self.triples['head_type'][idx]

        # get all the existing tails for this head and relation combination to create our target vector
        true_tail_idx = np.where((self.triples['head'] == head) & (self.triples['relation'] == relation))[0]
        true_tail_idx = [i for i in true_tail_idx if self.triples['head_type'][i] == head_type]

        true_tails = self.triples['tail'][true_tail_idx]
        true_tail_types = [self.triples['tail_type'][i] for i in true_tail_idx]
        true_tails_entity_idx = [x + self.entity_dict[y][0] for x, y in zip(true_tails, true_tail_types)]

        tails = np.zeros(self.nentity)
        tails[true_tails_entity_idx] = 1

        head = torch.LongTensor([head + self.entity_dict[head_type][0]])
        relation = torch.LongTensor([relation])
        tails = torch.FloatTensor(tails)

        return head, relation, tails
    #
    # @staticmethod
    # def collate_fn(data):
    #     positive_sample = torch.stack([_[0] for _ in data], dim=0)
    #     negative_sample = torch.stack([_[1] for _ in data], dim=0)
    #     subsample_weight = torch.cat([_[2] for _ in data], dim=0)
    #     mode = data[0][3]
    #     return positive_sample, negative_sample, subsample_weight, mode


class TestDataset(Dataset):
    def __init__(self, triples, args, mode, random_sampling, entity_dict):
        self.len = len(triples['head'])
        self.triples = triples
        self.nentity = args.nentity
        self.nrelation = args.nrelation
        self.mode = mode
        self.random_sampling = random_sampling
        if random_sampling:
            self.neg_size = args.neg_size_eval_train
        self.entity_dict = entity_dict

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
        head_type, tail_type = self.triples['head_type'][idx], self.triples['tail_type'][idx]
        positive_sample = torch.LongTensor(
            (head + self.entity_dict[head_type][0], relation, tail + self.entity_dict[tail_type][0]))

        if self.mode == 'head-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([head + self.entity_dict[head_type][0]]),
                                             torch.from_numpy(
                                                 self.triples['head_neg'][idx] + self.entity_dict[head_type][0])])
            else:
                negative_sample = torch.cat([torch.LongTensor([head + self.entity_dict[head_type][0]]),
                                             torch.randint(self.entity_dict[head_type][0],
                                                           self.entity_dict[head_type][1], size=(self.neg_size,))])
        elif self.mode == 'tail-batch':
            if not self.random_sampling:
                negative_sample = torch.cat([torch.LongTensor([tail + self.entity_dict[tail_type][0]]),
                                             torch.from_numpy(
                                                 self.triples['tail_neg'][idx] + self.entity_dict[tail_type][0])])
            else:
                negative_sample = torch.cat([torch.LongTensor([tail + self.entity_dict[tail_type][0]]),
                                             torch.randint(self.entity_dict[tail_type][0],
                                                           self.entity_dict[tail_type][1], size=(self.neg_size,))])

        return positive_sample, negative_sample, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        mode = data[0][2]

        return positive_sample, negative_sample, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data


class DataIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0

    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data