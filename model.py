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

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.xavier_uniform_(self.entity_embedding)
        # nn.init.uniform_(
        #     tensor=self.entity_embedding,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )
        self.entity_b = nn.Parameter(torch.zeros(nentity))

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.xavier_uniform_(self.relation_embedding)
        # nn.init.uniform_(
        #     tensor=self.relation_embedding,
        #     a=-self.embedding_range.item(),
        #     b=self.embedding_range.item()
        # )

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'ConvE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        if model_name == 'ConvE':
            assert self.entity_dim == self.relation_dim

            self.emb_dim1 = 5
            self.emb_dim2 = self.entity_dim // self.emb_dim1
            self.inp_drop = nn.Dropout(0.2)
            self.fm_drop = nn.Dropout2d(0.2)
            self.hid_drop = nn.Dropout(0.3)
            self.conv1 = nn.Conv2d(1, 16, (3, 3))
            self.bn0 = nn.BatchNorm2d(1)
            self.bn1 = nn.BatchNorm2d(16)
            self.bn2 = nn.BatchNorm1d(self.entity_dim)
            self.fc = torch.nn.Linear(16 * (self.emb_dim1 - 1) * (self.emb_dim2 - 2) * 2, self.entity_dim)

        self.evaluator = evaluator
        self.loss = nn.BCELoss()

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
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            b = torch.index_select(
                self.entity_b,
                dim=0,
                index=sample[:, 2]
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

            b = torch.index_select(
                self.entity_b,
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

            b = torch.index_select(
                self.entity_b,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        elif mode == 'all':
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[0][:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[1][:, 0]
            ).unsqueeze(1)

            tail, b = None, self.entity_b
        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'ConvE': self.ConvE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, b, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def TransE(self, head, relation, tail, mode):

        pred = head + relation
        if mode == 'all':
            pred = pred.squeeze(1)
            score = torch.mm(pred, self.entity_embedding.transpose(1, 0))
        else:
            score = torch.bmm(pred, tail.permute(0, 2, 1))
            score = score.squeeze()

        score = torch.sigmoid(score)

        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, b, mode):
        if mode == 'all':
            re_head, im_head = torch.chunk(head, 2, dim=2)
            re_relation, im_relation = torch.chunk(relation, 2, dim=2)
            re_tail, im_tail = torch.chunk(self.entity_embedding, 2, dim=1)

            re_score = (re_head * re_relation - im_head * im_relation).view(-1, re_head.shape[-1])
            im_score = (re_head * im_relation + im_head * re_relation).view(-1, re_head.shape[-1])

            score = torch.mm(re_score, re_tail.transpose(1, 0)) + torch.mm(im_score, im_tail.transpose(1, 0))

        else:
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

        return torch.sigmoid(score)

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

    def ConvE(self, head, relation, tail, b, mode):
        head_shape, relation_shape = head.size(), relation.size()

        head = F.normalize(head, 2, -1)
        relation = F.normalize(relation, 2, -1)

        # if required, repeat embeddings acc to mode
        num_neg = 1
        if mode == 'head-batch':
            num_neg = head_shape[1]
            relation = relation.repeat(1, num_neg, 1)

        # reshape and stack the triplet embeddings
        head = head.view(-1, 1, self.emb_dim1, self.emb_dim2)
        relation = relation.view(-1, 1, self.emb_dim1, self.emb_dim2)

        x = torch.cat([head, relation], 2)
        # x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.fm_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hid_drop(x)
        # x = self.bn2(x)
        x = F.relu(x)
        x = x.view(head_shape[0], num_neg, -1)

        if mode == 'all':
            x = x.squeeze(1)
            score = torch.mm(x, self.entity_embedding.transpose(1, 0))
        else:
            score = torch.bmm(x, tail.permute(0, 2, 1))
            score = score.squeeze()
            if len(b.size()) == 3:
                b = b.squeeze(dim=2)

        score += b.expand_as(score)

        score = torch.sigmoid(score)

        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, accumulate=False):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        head, relation, tails = next(train_iterator)

        if args.cuda:
            head = head.cuda()
            relation = relation.cuda()
            tails = tails.cuda()

        scores = model((head, relation), mode='all')

        loss = model.loss(scores, tails)

        if args.regularization != 0.0:
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
