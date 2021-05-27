#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataloader import TestDataset
from collections import defaultdict

from ogb.linkproppred import Evaluator

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, evaluator,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]), 
            requires_grad=False
        )
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim
        
        if model_name == 'TransM':
            # self.entity_dim = self.entity_dim * self.entity_dim
            self.relation_dim = self.entity_dim * self.entity_dim

        if model_name == 'ConvFM':
            self.entity_dim = self.entity_dim * self.entity_dim
            self.relation_dim = self.entity_dim

        if model_name == '5Star':
            self.relation_dim = self.entity_dim * 4

        if model_name == 'RelConv':
            self.entity_dim = self.entity_dim ** 2
            self.nfilters = self.entity_dim // 9
            self.relation_dim = 9 * self.nfilters

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.xavier_uniform_(self.entity_embedding)
        # nn.init.uniform_(
        #     tensor=self.entity_embedding, 
        #     a=-self.embedding_range.item(), 
        #     b=self.embedding_range.item()
        # )
        
        # if model_name != 'TransM':
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.xavier_uniform_(self.relation_embedding)
        # nn.init.uniform_(
        #     tensor=self.relation_embedding, 
        #     a=-self.embedding_range.item(), 
        #     b=self.embedding_range.item()
        # )
        # else:
        #     self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.entity_dim, self.entity_dim))
        #     nn.init.uniform_(
        #         tensor=self.relation_embedding, 
        #         a=-self.embedding_range.item(), 
        #         b=self.embedding_range.item()
        #     )
          

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'ConvE', 
        'ConvKB', 'NormConvKB', 'SymNormConvKB', 'TransM', '5Star', 'RelConv', 'QuatE', 'FullConvKB',
        'ConvFM', 'ConEx']:
            raise ValueError('model %s not supported' % model_name)
            
        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        if model_name == 'ConvE':
            assert self.entity_dim == self.relation_dim

            self.emb_dim1 = 10
            self.emb_dim2 = self.entity_dim // self.emb_dim1
            self.inp_drop = nn.Dropout(0.0)
            self.fm_drop = nn.Dropout2d(0.0)
            self.hid_drop = nn.Dropout(0.0)
            self.conv1 = nn.Conv2d(1, 16, (3, 3))
            self.bn0 = nn.BatchNorm2d(1)
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm1d(self.entity_dim)
            self.fc = torch.nn.Linear(16 * (self.emb_dim1 - 1) * (self.emb_dim2 - 2) * 2, self.entity_dim)
        
        if model_name == 'ConvKB':
            self.nfmap = 3
            self.conv1_bn = nn.BatchNorm2d(1)
            self.conv_layer = nn.Conv2d(1, self.nfmap, (1, 3))  # kernel size x 3
            self.conv2_bn = nn.BatchNorm2d(self.nfmap)
            self.dropout = nn.Dropout(0.0)
            self.non_linearity = nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
            self.fc_layer = nn.Linear(self.nfmap * self.entity_dim, 1, bias=False)

        if model_name == 'NormConvKB':
            self.nfmap = 12
            self.conv1_bn = nn.BatchNorm2d(1)
            self.conv_layer = nn.Conv2d(1, self.nfmap, (1, 3))  # kernel size x 3
            self.conv2_bn = nn.BatchNorm2d(self.nfmap)
            self.conv2_lrn = nn.LocalResponseNorm(2)
            self.dropout = nn.Dropout(0.0)
            self.non_linearity = nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
            self.fc_layer = nn.Linear(self.nfmap * self.entity_dim, 1, bias=False)

        if model_name == 'ConvFM':
            self.nfmap = 1
            self.conv_layer = nn.Conv2d(3, self.nfmap, (3, 3), padding=1)  # kernel size x 3
            self.dropout = nn.Dropout(0.0)
            self.non_linearity = nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
            self.fc_layer = nn.Linear(self.nfmap * self.entity_dim, 1, bias=False)

        if model_name == 'FullConvKB':
            self.nfmaps = [8, 16, 8]
            self.conv_layer_1 = nn.Conv2d(1, self.nfmaps[0], (3, 3), padding=1)
            self.conv_layer_2 = nn.Conv2d(self.nfmaps[0], self.nfmaps[1], (3, 3), padding=1)
            self.conv_layer_3 = nn.Conv2d(self.nfmaps[1], self.nfmaps[2], (2, 2))
            self.conv_layer_4 = nn.Conv2d(self.nfmaps[2], 1, (1, 1))

            self.conv_layers = [
              self.conv_layer_1,
              self.conv_layer_2,
              self.conv_layer_3,
              self.conv_layer_4
            ]

            self.dropout = nn.Dropout(0.0)
            self.non_linearity = nn.ReLU()

        if model_name == 'ConEx':
            self.nfmap = 1
            self.conv_layer = nn.Conv2d(1, self.nfmap, (3, 3), padding=1)
            self.dropout = nn.Dropout(0.0)
            self.non_linearity = nn.ReLU()
            self.fc_layer = nn.Linear(self.nfmap * self.entity_dim * 2, self.entity_dim)

        if model_name == 'SymNormConvKB':
            self.nfmap = 8  
            self.conv1_bn = nn.BatchNorm2d(1)
            self.conv_layer1 = nn.Conv2d(1, self.nfmap, (1, 2))
            self.conv_layer2 = nn.Conv2d(self.nfmap, self.nfmap, (1, 2))
            self.conv2_bn = nn.BatchNorm2d(self.nfmap)
            self.dropout = nn.Dropout(0.0)
            self.non_linearity = nn.ReLU() # you should also tune with torch.tanh() or torch.nn.Tanh()
            self.fc_layer = nn.Linear(self.nfmap * self.entity_dim, 1, bias=False)

        self.evaluator = evaluator
        
    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)
            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)
            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'ConvE': self.ConvE,
            'ConvKB': self.ConvKB,
            'NormConvKB': self.NormConvKB,
            'SymNormConvKB': self.SymNormConvKB,
            'TransM': self.TransM,
            '5Star': self.FiveStar,
            'RelConv': self.RelConv,
            'QuatE': self.QuatE,
            'FullConvKB': self.FullConvKB,
            'ConvFM': self.ConvFM,
            'ConEx': self.ConEx
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        # print(head.shape, relation.shape, tail.shape, mode)
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def ConvE(self, head, relation, tail, mode):
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()

        # if required, repeat embeddings acc to mode
        num_neg = 1
        if mode == 'head-batch':
            num_neg = head_shape[1]
            tail = tail.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)
        elif mode == 'tail-batch':
            num_neg = tail_shape[1]
            head = head.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)

        # reshape and stack the triplet embeddings
        head = head.view(-1, 1, self.emb_dim1, self.emb_dim2)
        # tail = tail.view(-1, 1, self.emb_dim1, self.emb_dim2)
        relation = relation.view(-1, 1, self.emb_dim1, self.emb_dim2)

        x = torch.cat([head, relation], 2)
        # x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.fm_drop(x)
        # x = x.mean(1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = x.view(head_shape[0], num_neg, -1)
        x = (x * tail).sum(2)
        score = torch.sigmoid(x)
        # score = x - tail
        # score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        # x = torch.mean(x, 1)
        # x = x.view(head_shape[0], num_neg, -1)
        # score = torch.mean(x, 2)
        # x = self.fc(x)
        # score = x.view(-1, num_neg)
        return score

    def ConvKB(self, head, relation, tail, mode):
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()

        # if required, repeat embeddings acc to mode
        num_neg = 1
        if mode == 'head-batch':
            num_neg = head_shape[1]
            tail = tail.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)
        elif mode == 'tail-batch':
            num_neg = tail_shape[1]
            head = head.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)

        # reshape and stack the triplet embeddings
        head = head.view(-1, 1, self.entity_dim, 1)
        tail = tail.view(-1, 1, self.entity_dim, 1)
        relation = relation.view(-1, 1, self.entity_dim, 1)

        x = torch.cat([head, relation, tail], 3)

        # conv_input = self.conv1_bn(x)
        out_conv = self.conv_layer(x)
        # out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, self.nfmap * self.entity_dim)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc)

        # x = self.bn0(x)
        # x = self.conv1(x)
        # # x = F.relu(x)
        # x = x.view(-1, self.nfmap * self.entity_dim)
        # # score = x.sum(1)
        # score = torch.mm(x, self.w)
        score = score.view(head_shape[0], num_neg)
        # score = self.gamma.item() - torch.norm(score, p=1, dim=2)

        # regularization
        l2_reg = torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)

        self.l2_reg = l2_reg

        return score

    def NormConvKB(self, head, relation, tail, mode):
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        relation = F.normalize(relation, 2, -1)

        # if required, repeat embeddings acc to mode
        num_neg = 1
        if mode == 'head-batch':
            num_neg = head_shape[1]
            tail = tail.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)
        elif mode == 'tail-batch':
            num_neg = tail_shape[1]
            head = head.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)

        # reshape and stack the triplet embeddings
        head = head.view(-1, 1, self.entity_dim, 1)
        tail = tail.view(-1, 1, self.entity_dim, 1)
        relation = relation.view(-1, 1, self.entity_dim, 1)

        x = torch.cat([head, relation, tail], 3)

        # conv_input = self.conv1_bn(x)
        out_conv = self.conv_layer(x)
        # out_conv = self.conv2_lrn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, self.nfmap * self.entity_dim)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc)

        score = score.view(head_shape[0], num_neg)

        # regularization
        l2_reg = torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)

        self.l2_reg = l2_reg

        return score

    def ConEx(self, head, relation, tail, mode):
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()

        # if required, repeat embeddings acc to mode
        num_neg = 1
        if mode == 'head-batch':
            num_neg = head_shape[1]
            tail = tail.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)
        elif mode == 'tail-batch':
            num_neg = tail_shape[1]
            head = head.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        re_head = F.normalize(re_head, 2, -1)
        im_head = F.normalize(im_head, 2, -1)
        re_relation = F.normalize(im_head, 2, -1)
        im_relation = F.normalize(im_relation, 2, -1)
        re_tail = F.normalize(re_tail, 2, -1)
        im_tail = F.normalize(im_tail, 2, -1)

        # reshape and stack the head and relation embeddings
        re_head = re_head.view(-1, 1, self.entity_dim // 2, 1)
        im_head = im_head.view(-1, 1, self.entity_dim // 2, 1)
        re_tail = re_tail.view(-1, self.entity_dim // 2)
        im_tail = im_tail.view(-1, self.entity_dim // 2)
        re_relation = re_relation.view(-1, 1, self.entity_dim // 2, 1)
        im_relation = im_relation.view(-1, 1, self.entity_dim // 2, 1)

        x = torch.cat([re_head, im_head, re_relation, im_relation], 3)

        out_conv = self.conv_layer(x)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, self.nfmap * self.entity_dim * 2)
        input_fc = self.dropout(out_conv)
        head_relation = self.fc_layer(input_fc)
        re_head_relation, im_head_relation = torch.chunk(head_relation, 2, dim=1)
        # re_head_relation = out_conv[:, :, :, 0].view(-1, self.entity_dim // 2)
        # im_head_relation = out_conv[:, :, :, 1].view(-1, self.entity_dim // 2)

        re_head = re_head.view(-1, self.entity_dim // 2)
        im_head = im_head.view(-1, self.entity_dim // 2)
        re_relation = re_relation.view(-1, self.entity_dim // 2)
        im_relation = im_relation.view(-1, self.entity_dim // 2)

        real_real_real = (re_head_relation * re_head * re_relation * re_tail).sum(dim=1)
        real_imag_imag = (re_head_relation * re_head * im_relation * im_tail).sum(dim=1)
        imag_real_imag = (im_head_relation * im_head * re_relation * im_tail).sum(dim=1)
        imag_imag_real = (im_head_relation * im_head * im_relation * re_tail).sum(dim=1)
        score = real_real_real + real_imag_imag + imag_real_imag - imag_imag_real

        score = score.view(head_shape[0], num_neg)

        # regularization
        l2_reg = torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)

        self.l2_reg = l2_reg

        return score

    def ConvFM(self, head, relation, tail, mode):
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()
        matrix_dim = int(np.sqrt(self.entity_dim))

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        relation = F.normalize(relation, 2, -1)

        # if required, repeat embeddings acc to mode
        num_neg = 1
        if mode == 'head-batch':
            num_neg = head_shape[1]
            tail = tail.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)
        elif mode == 'tail-batch':
            num_neg = tail_shape[1]
            head = head.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)

        # reshape and stack the triplet embeddings
        head = head.view(-1, 1, matrix_dim, matrix_dim)
        tail = tail.view(-1, 1, matrix_dim, matrix_dim)
        relation = relation.view(-1, 1, matrix_dim, matrix_dim)

        x = torch.cat([head, relation, tail], 1)

        out_conv = self.conv_layer(x)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, self.nfmap * self.entity_dim)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc)

        score = score.view(head_shape[0], num_neg)

        # regularization
        l2_reg = torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)
        for W in self.conv_layer.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)

        self.l2_reg = l2_reg

        return score

    def SymNormConvKB(self, head, relation, tail, mode):
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()

        # if required, repeat embeddings acc to mode
        num_neg = 1
        if mode == 'head-batch':
            num_neg = head_shape[1]
            tail = tail.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)
        elif mode == 'tail-batch':
            num_neg = tail_shape[1]
            head = head.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)

        # reshape and stack the triplet embeddings
        head = head.view(-1, 1, self.entity_dim, 1)
        tail = tail.view(-1, 1, self.entity_dim, 1)
        relation = relation.view(-1, 1, self.entity_dim, 1)

        x = torch.cat([head, relation, tail], 3)

        # conv_input = self.conv1_bn(x)
        out_conv = self.conv_layer1(x)
        # out_conv = self.conv2_bn(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = self.conv_layer2(out_conv)
        out_conv = self.non_linearity(out_conv)

        out_conv = out_conv.view(-1, self.nfmap * self.entity_dim)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc)

        score = score.view(head_shape[0], num_neg)

        # regularization
        l2_reg = torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)
        for W in self.conv_layer1.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.conv_layer2.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.fc_layer.parameters():
            l2_reg = l2_reg + W.norm(2)

        self.l2_reg = l2_reg

        return score

    def TransM(self, head, relation, tail, mode):
        matrix_dim = self.entity_dim
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()
        
        # head = head.view(-1, head_shape[1], matrix_dim, matrix_dim)
        # tail = tail.view(-1, tail_shape[1], matrix_dim, matrix_dim)
        relation = relation.view(-1, matrix_dim, matrix_dim)

        # print(head.mean(), head.std(), relation.mean(), relation.std(), tail.mean(), tail.std())

        transform = torch.matmul(head, relation).view(-1, head_shape[1], matrix_dim)
        score = transform - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def FiveStar(self, head, relation, tail, mode):
        relation = relation.view(relation.shape[0], -1, 2, 2)

        # convert head to homogeneous coords
        ones = torch.ones_like(head)
        head = torch.stack([head, ones], dim=2)

        transform = torch.matmul(relation, head).view(-1, head_shape[1], matrix_dim)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score


    def RelConv(self, head, relation, tail, mode):
        matrix_dim = int(np.sqrt(self.entity_dim))
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()

        relation = relation.view(-1, self.nfilters, 1, 3, 3)
        head = head.view(-1, head_shape[1], 1, matrix_dim, matrix_dim)
        tail = tail.view(-1, tail_shape[1], matrix_dim, matrix_dim)

        # convolve
        score = torch.zeros(head_shape[0], max(head_shape[1], tail_shape[1]))
        for s in range(head_shape[0]):
            conv_out = F.conv2d(head[s], relation[s], padding=1)
            conv_out = conv_out.mean(1)
            score[s] = (conv_out - tail[s]).sum(-1).sum(-1)

        return score


    def QuatE(self, head, relation, tail, mode):
        # head = F.normalize(head, 2, -1)
        # tail = F.normalize(tail, 2, -1)
        # relation = F.normalize(relation, 2, -1)

        # if required, repeat embeddings acc to mode
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()
        num_neg = 1
        if mode == 'head-batch':
            num_neg = head_shape[1]
            tail = tail.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)
        elif mode == 'tail-batch':
            num_neg = tail_shape[1]
            head = head.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)

        h_0, h_1, h_2, h_3 = torch.chunk(head, 4, dim=2)
        re_0, re_1, re_2, re_3 = torch.chunk(relation, 4, dim=2)
        t_0, t_1, t_2, t_3 = torch.chunk(tail, 4, dim=2)

        # normalize relations
        den = torch.sqrt(re_0 ** 2 + re_1 ** 2 + re_2 ** 2 + re_3 ** 2)
        re_0, re_1, re_2, re_3 = re_0 / den, re_1 / den, re_2 / den, re_3 / den

        # Hamiltonian product
        A = h_0 * re_0 - h_1 * re_1 - h_2 * re_2 - h_3 * re_3
        B = h_0 * re_1 + re_0 * h_1 + h_2 * re_3 - re_2 * h_3
        C = h_0 * re_2 + re_0 * h_2 + h_3 * re_1 - re_3 * h_1
        D = h_0 * re_3 + re_0 * h_3 + h_1 * re_2 - re_1 * h_2

        score = A * t_0 + B * t_1 + C * t_2 + D * t_3

        score = -score.sum(dim = 2)
        return score

    def FullConvKB(self, head, relation, tail, mode):
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        relation = F.normalize(relation, 2, -1)

        # if required, repeat embeddings acc to mode
        num_neg = 1
        if mode == 'head-batch':
            num_neg = head_shape[1]
            tail = tail.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)
        elif mode == 'tail-batch':
            num_neg = tail_shape[1]
            head = head.repeat(1, num_neg, 1)
            relation = relation.repeat(1, num_neg, 1)

        # reshape and stack the triplet embeddings
        head = head.view(-1, 1, self.entity_dim, 1)
        tail = tail.view(-1, 1, self.entity_dim, 1)
        relation = relation.view(-1, 1, self.entity_dim, 1)

        out_conv = torch.cat([head, relation, tail], 3)

        for conv_layer in self.conv_layers:
            out_conv = conv_layer(out_conv)
            out_conv = self.non_linearity(out_conv)
            out_conv = F.max_pool2d(out_conv, (3, 1))

        score = torch.mean(out_conv.view(out_conv.shape[0], -1), 1)

        score = score.view(head_shape[0], num_neg)

        # regularization
        l2_reg = torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)
        for conv_layer in self.conv_layers:
            for W in conv_layer.parameters():
                l2_reg = l2_reg + W.norm(2)
        self.l2_reg = l2_reg

        return score


    @staticmethod
    def train_step(model, optimizer, train_iterator, args, accumulate=False):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args.model in ['ConvKB', 'NormConvKB', 'SymNormConvKB']:
            neg_regularization = model.l2_reg
        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)
        if args.model in ['ConvKB', 'NormConvKB', 'SymNormConvKB']:
            pos_regularization = model.l2_reg
        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2
        if args.model in ['ConvKB', 'NormConvKB', 'SymNormConvKB']:
            regularization = (pos_regularization + neg_regularization)/2
        
        if args.regularization != 0.0:
            if args.model in ['ConvKB', 'NormConvKB', 'SymNormConvKB']:
                #Use L2 regularization for ConvKB
                regularization = args.regularization * regularization
            else:
                #Use L3 regularization for ComplEx and DistMult
                regularization = args.regularization * (
                    model.entity_embedding.norm(p = 3)**3 + 
                    model.relation_embedding.norm(p = 3).norm(p = 3)**3
                )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
        
        loss.backward()

        if not accumulate:
            optimizer.step()
            optimizer.zero_grad()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, args, entity_dict, random_sampling=False):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()

        #Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples, 
                args, 
                'head-batch',
                random_sampling,
                entity_dict
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples, 
                args, 
                'tail-batch',
                random_sampling,
                entity_dict
            ), 
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num//2), 
            collate_fn=TestDataset.collate_fn
        )
        
        test_dataset_list = [test_dataloader_head, test_dataloader_tail]
        
        test_logs = defaultdict(list)

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    batch_size = positive_sample.size(0)
                    score = model((positive_sample, negative_sample), mode)

                    batch_results = model.evaluator.eval({'y_pred_pos': score[:, 0], 
                                                'y_pred_neg': score[:, 1:]})
                    for metric in batch_results:
                        test_logs[metric].append(batch_results[metric])

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()

        return metrics
