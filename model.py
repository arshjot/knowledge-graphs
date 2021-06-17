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

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

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

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'ConvE',
                              'ConvKB', 'NormConvKB', 'SymNormConvKB', 'TransM', 'HolE', 'RelConv', 'QuatE',
                              'FullConvKB',
                              'ConvFM', 'ConEx', 'NTN', 'ConvQuatE', 'ComplExQuatE', 'OctonionE']:
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
            self.non_linearity = nn.ReLU()  # you should also tune with torch.tanh() or torch.nn.Tanh()
            self.fc_layer = nn.Linear(self.nfmap * self.entity_dim, 1, bias=False)

        if model_name == 'NormConvKB':
            self.nfmap = 12
            # self.conv1_bn = nn.BatchNorm2d(1)
            self.conv_layer = nn.Conv2d(1, self.nfmap, (1, 3))  # kernel size x 3
            # self.conv2_bn = nn.BatchNorm2d(self.nfmap)
            # self.conv2_lrn = nn.LocalResponseNorm(2)
            self.dropout = nn.Dropout(0.0)
            self.non_linearity = nn.ReLU()  # you should also tune with torch.tanh() or torch.nn.Tanh()
            self.fc_layer = nn.Linear(self.nfmap * self.entity_dim, 1, bias=False)

        if model_name == 'ConvFM':
            self.nfmap1 = 12
            self.nfmap2 = 3
            self.conv_layer_1 = nn.Conv2d(3, self.nfmap1, (3, 3), padding=1)  # kernel size x 3
            self.conv_layer_2 = nn.Conv2d(self.nfmap1, self.nfmap2, (3, 3), padding=1)  # kernel size x 3
            self.dropout = nn.Dropout(0.0)
            self.non_linearity = nn.ReLU()  # you should also tune with torch.tanh() or torch.nn.Tanh()
            self.fc_layer = nn.Linear(self.nfmap2 * self.entity_dim, 1, bias=False)

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
            self.non_linearity = nn.ReLU()  # you should also tune with torch.tanh() or torch.nn.Tanh()
            self.fc_layer = nn.Linear(self.nfmap * self.entity_dim, 1, bias=False)

        if model_name == 'NTN':
            num_slices = 4
            self.w = nn.Parameter(data=torch.empty(
                nrelation,
                num_slices,
                self.entity_dim,
                self.entity_dim), requires_grad=True)
            self.vh = nn.Parameter(data=torch.empty(
                nrelation,
                num_slices,
                self.entity_dim), requires_grad=True)
            self.vt = nn.Parameter(data=torch.empty(
                nrelation,
                num_slices,
                self.entity_dim), requires_grad=True)
            self.b = nn.Parameter(data=torch.empty(
                nrelation,
                num_slices), requires_grad=True)
            self.u = nn.Parameter(data=torch.empty(
                nrelation,
                num_slices), requires_grad=True)
            self.non_linearity = nn.Tanh()

        if model_name == 'ConvQuatE':
            self.conv_layer_h = nn.Conv2d(1, 4, (3, 3), padding=1)
            self.conv_layer_r = nn.Conv2d(1, 4, (3, 3), padding=1)
            self.conv_layer_t = nn.Conv2d(1, 4, (3, 3), padding=1)

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

            self.head_idx = sample[:, 0]
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            self.relation_idx = sample[:, 1]
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            self.tail_idx = sample[:, 2]
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            self.head_idx = head_part.view(-1)
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            self.relation_idx = tail_part[:, 1]
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            self.tail_idx = tail_part[:, 2]
            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            self.head_idx = head_part[:, 0]
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            self.relation_idx = head_part[:, 1]
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            self.tail_idx = tail_part.view(-1)
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
            'HolE': self.HolE,
            'RelConv': self.RelConv,
            'QuatE': self.QuatE,
            'FullConvKB': self.FullConvKB,
            'ConvFM': self.ConvFM,
            'ConEx': self.ConEx,
            'NTN': self.NTN,
            'ConvQuatE': self.ConvQuatE,
            'ComplExQuatE': self.ComplExQuatE,
            'OctonionE': self.OctonionE
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

        score = score.sum(dim=2)
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

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

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

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
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

        out_conv = self.conv_layer_1(x)
        out_conv = self.non_linearity(out_conv)
        out_conv = self.conv_layer_2(out_conv)
        out_conv = self.non_linearity(out_conv)
        out_conv = out_conv.view(-1, self.nfmap2 * self.entity_dim)
        input_fc = self.dropout(out_conv)
        score = self.fc_layer(input_fc)

        score = score.view(head_shape[0], num_neg)

        # regularization
        l2_reg = torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)
        for W in self.conv_layer_1.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.conv_layer_2.parameters():
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

        relation = relation.view(-1, matrix_dim, matrix_dim)

        transform = torch.matmul(head, relation).view(-1, head_shape[1], matrix_dim)
        score = torch.matmul(transform, tail.view(-1, matrix_dim, tail_shape[1])).view(head_shape[0], -1)

        return score

    def HolE(self, head, relation, tail, mode):
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()

        # circular correlation
        head_fft = torch.fft.rfft(head)
        tail_fft = torch.fft.rfft(tail)

        head_fft = torch.conj(head_fft)

        # Hadamard product in frequency domain
        p_fft = head_fft * tail_fft

        # inverse real FFT, shape: (batch_size, num_entities, d)
        composite = torch.fft.irfft(p_fft, dim=-1, n=head_shape[-1])

        # inner product with relation embedding
        score = torch.sum(relation * composite, dim=-1, keepdim=False)

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

        score = -score.sum(dim=2)
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

    def NTN(self, head, relation, tail, mode):
        head_shape, tail_shape, relation_shape = head.size(), tail.size(), relation.size()

        # if required, repeat embeddings acc to mode
        num_neg = 1
        if mode == 'head-batch':
            num_neg = head_shape[1]
            tail = tail.repeat(1, num_neg, 1)
        elif mode == 'tail-batch':
            num_neg = tail_shape[1]
            head = head.repeat(1, num_neg, 1)

        #: Prepare h: (b, e, d) -> (b, e, 1, 1, d)
        h_for_w = head.unsqueeze(dim=-2).unsqueeze(dim=-2)

        #: Prepare t: (b, e, d) -> (b, e, 1, d, 1)
        t_for_w = tail.unsqueeze(dim=-2).unsqueeze(dim=-1)

        #: Prepare w: (R, k, d, d) -> (b, k, d, d) -> (b, 1, k, d, d)
        w_r = self.w.index_select(dim=0, index=self.relation_idx).unsqueeze(dim=1)

        # h.T @ W @ t, shape: (b, e, k, 1, 1)
        hwt = (h_for_w @ w_r @ t_for_w)

        #: reduce (b, e, k, 1, 1) -> (b, e, k)
        hwt = hwt.squeeze(dim=-1).squeeze(dim=-1)

        #: Prepare vh: (R, k, d) -> (b, k, d) -> (b, 1, k, d)
        vh_r = self.vh.index_select(dim=0, index=self.relation_idx).unsqueeze(dim=1)

        #: Prepare h: (b, e, d) -> (b, e, d, 1)
        h_for_v = head.unsqueeze(dim=-1)

        # V_h @ h, shape: (b, e, k, 1)
        vhh = vh_r @ h_for_v

        #: reduce (b, e, k, 1) -> (b, e, k)
        vhh = vhh.squeeze(dim=-1)

        #: Prepare vt: (R, k, d) -> (b, k, d) -> (b, 1, k, d)
        vt_r = self.vt.index_select(dim=0, index=self.relation_idx).unsqueeze(dim=1)

        #: Prepare t: (b, e, d) -> (b, e, d, 1)
        t_for_v = tail.unsqueeze(dim=-1)

        # V_t @ t, shape: (b, e, k, 1)
        vtt = vt_r @ t_for_v

        #: reduce (b, e, k, 1) -> (b, e, k)
        vtt = vtt.squeeze(dim=-1)

        #: Prepare b: (R, k) -> (b, k) -> (b, 1, k)
        b = self.b.index_select(dim=0, index=self.relation_idx).unsqueeze(dim=1)

        # a = f(h.T @ W @ t + Vh @ h + Vt @ t + b), shape: (b, e, k)
        pre_act = hwt + vhh + vtt + b
        act = self.non_linearity(pre_act)

        # prepare u: (R, k) -> (b, k) -> (b, 1, k, 1)
        u = self.u.index_select(dim=0, index=self.relation_idx).unsqueeze(dim=1).unsqueeze(dim=-1)

        # prepare act: (b, e, k) -> (b, e, 1, k)
        act = act.unsqueeze(dim=-2)

        # compute score, shape: (b, e, 1, 1)
        score = act @ u

        # reduce
        score = score.squeeze(dim=-1).squeeze(dim=-1)

        # regularization
        l2_reg = torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)
        # for W in self.w.parameters():
        #     l2_reg = l2_reg + W.norm(2)
        # for W in self.vh.parameters():
        #     l2_reg = l2_reg + W.norm(2)
        # for W in self.vt.parameters():
        #     l2_reg = l2_reg + W.norm(2)
        # for W in self.u.parameters():
        #     l2_reg = l2_reg + W.norm(2)

        self.l2_reg = l2_reg

        return score

    def ConvQuatE(self, head, relation, tail, mode):
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

        out_conv_h = self.conv_layer_h(head).view(head_shape[0], num_neg, 4, matrix_dim * matrix_dim)
        out_conv_r = self.conv_layer_r(relation).view(head_shape[0], num_neg, 4, matrix_dim * matrix_dim)
        out_conv_t = self.conv_layer_t(tail).view(head_shape[0], num_neg, 4, matrix_dim * matrix_dim)

        h_0, h_1, h_2, h_3 = out_conv_h[:, :, 0], out_conv_h[:, :, 1], out_conv_h[:, :, 2], out_conv_h[:, :, 3]
        re_0, re_1, re_2, re_3 = out_conv_r[:, :, 0], out_conv_r[:, :, 1], out_conv_r[:, :, 2], out_conv_r[:, :, 3]
        t_0, t_1, t_2, t_3 = out_conv_t[:, :, 0], out_conv_t[:, :, 1], out_conv_t[:, :, 2], out_conv_t[:, :, 3]

        # normalize relations
        den = torch.sqrt(re_0 ** 2 + re_1 ** 2 + re_2 ** 2 + re_3 ** 2)
        re_0, re_1, re_2, re_3 = re_0 / den, re_1 / den, re_2 / den, re_3 / den

        # Hamiltonian product
        A = h_0 * re_0 - h_1 * re_1 - h_2 * re_2 - h_3 * re_3
        B = h_0 * re_1 + re_0 * h_1 + h_2 * re_3 - re_2 * h_3
        C = h_0 * re_2 + re_0 * h_2 + h_3 * re_1 - re_3 * h_1
        D = h_0 * re_3 + re_0 * h_3 + h_1 * re_2 - re_1 * h_2

        score = A * t_0 + B * t_1 + C * t_2 + D * t_3

        score = -score.sum(dim=2)

        # regularization
        l2_reg = torch.mean(head ** 2) + torch.mean(tail ** 2) + torch.mean(relation ** 2)
        for W in self.conv_layer_h.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.conv_layer_t.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.conv_layer_r.parameters():
            l2_reg = l2_reg + W.norm(2)

        self.l2_reg = l2_reg

        return score

    def ComplExQuatE(self, head, relation, tail, mode):
        complex_score = self.ComplEx(head, relation, tail, mode)
        quate_score = self.QuatE(head, relation, tail, mode)

        return torch.mean(torch.stack([complex_score, quate_score]), dim=0)

    def OctonionE(self, head, relation, tail, mode):
        """https://github.com/Sujit-O/pykg2vec/blob/492807b627574f95b0db9e7cb9f090c3c45a030a/pykg2vec/models/pointwise.py#L772"""

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

        e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h = torch.chunk(head, 8, dim=2)
        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = torch.chunk(relation, 8, dim=2)
        e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t = torch.chunk(tail, 8, dim=2)

        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self._onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)

        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = self._omult(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                                                             r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)

        score_r = (o_1 * e_1_t + o_2 * e_2_t + o_3 * e_3_t + o_4 * e_4_t
                   + o_5 * e_5_t + o_6 * e_6_t + o_7 * e_7_t + o_8 * e_8_t)

        return -torch.sum(score_r, -1)

    def _qmult(self, s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        a = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        b = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        c = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        d = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        return a, b, c, d

    def _qstar(self, a, b, c, d):
        return a, -b, -c, -d

    def _omult(self, a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c_1, c_2, c_3, c_4, d_1, d_2, d_3, d_4):

        d_1_star, d_2_star, d_3_star, d_4_star = self._qstar(d_1, d_2, d_3, d_4)
        c_1_star, c_2_star, c_3_star, c_4_star = self._qstar(c_1, c_2, c_3, c_4)

        o_1, o_2, o_3, o_4 = self._qmult(a_1, a_2, a_3, a_4, c_1, c_2, c_3, c_4)
        o_1s, o_2s, o_3s, o_4s = self._qmult(d_1_star, d_2_star, d_3_star, d_4_star, b_1, b_2, b_3, b_4)

        o_5, o_6, o_7, o_8 = self._qmult(d_1, d_2, d_3, d_4, a_1, a_2, a_3, a_4)
        o_5s, o_6s, o_7s, o_8s = self._qmult(b_1, b_2, b_3, b_4, c_1_star, c_2_star, c_3_star, c_4_star)

        return o_1 - o_1s, o_2 - o_2s, o_3 - o_3s, o_4 - o_4s, \
               o_5 + o_5s, o_6 + o_6s, o_7 + o_7s, o_8 + o_8s

    def _onorm(self, r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8):
        denominator = torch.sqrt(r_1 ** 2 + r_2 ** 2 + r_3 ** 2 + r_4 ** 2
                                 + r_5 ** 2 + r_6 ** 2 + r_7 ** 2 + r_8 ** 2)
        r_1 = r_1 / denominator
        r_2 = r_2 / denominator
        r_3 = r_3 / denominator
        r_4 = r_4 / denominator
        r_5 = r_5 / denominator
        r_6 = r_6 / denominator
        r_7 = r_7 / denominator
        r_8 = r_8 / denominator

        return r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, accumulate=False):
        '''
        A single train step. Apply back-propagation and return the loss
        '''

        model.train()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        l2_reg_models = ['ConvKB', 'NormConvKB', 'SymNormConvKB', 'NTN', 'ConvQuatE']
        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args.model in l2_reg_models:
            neg_regularization = model.l2_reg
        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        if args.model in l2_reg_models:
            pos_regularization = model.l2_reg
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2
        if args.model in l2_reg_models:
            regularization = (pos_regularization + neg_regularization) / 2

        if args.regularization != 0.0:
            if args.model in l2_reg_models:
                # Use L2 regularization for ConvKB
                regularization = args.regularization * regularization
            else:
                # Use L3 regularization for ComplEx and DistMult
                regularization = args.regularization * (
                        model.entity_embedding.norm(p=3) ** 3 +
                        model.relation_embedding.norm(p=3).norm(p=3) ** 3
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

        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                args,
                'head-batch',
                random_sampling,
                entity_dict
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
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
            num_workers=max(1, args.cpu_num // 2),
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
