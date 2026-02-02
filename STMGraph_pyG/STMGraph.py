import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# from .gat_conv import GATConv
from .gatv2_conv import GATv2Conv as GATConv
import random

class encoding_mask_noise(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(encoding_mask_noise, self).__init__()   
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.enc_mask_token = nn.Parameter(torch.zeros(size=(1, in_dim)))
        self.reset_parameters_for_token()
        
    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token.data, gain=1.414)#
        
    def forward(self, x, mask_rate=0.5, replace_rate=0.05):
        # num_nodes = g.num_nodes()
        num_nodes = x.size()[0]
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]
        
        if replace_rate > 0.0:
            num_noise_nodes = int(replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: -num_noise_nodes]]#int(mask_token_rate * num_mask_nodes)
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            # out_x[token_nodes] = torch.zeros_like(out_x[token_nodes])
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
            # out_x[noise_nodes] = torch.add(x[noise_to_be_chosen], out_x[noise_nodes]) 
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        # use_g = g.clone()
        return out_x, mask_nodes, keep_nodes

class random_remask(torch.nn.Module):
    def __init__(self, hidden_dims):
        super(random_remask, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        self.dec_mask_token = nn.Parameter(torch.zeros(size=(1, out_dim)))
        self.reset_parameters_for_token()
        
    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.dec_mask_token.data, gain=1.414)
        
    def forward(self,rep,remask_rate=0.5):
        num_nodes = rep.size()[0]
        # num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes: ]

        out_rep = rep.clone()
        out_rep[remask_nodes] = 0.0
        out_rep[remask_nodes] += self.dec_mask_token
        return out_rep, remask_nodes, rekeep_nodes


class STMGraph(torch.nn.Module):
    def __init__(self, hidden_dims, delayed_ema_epoch, momentum):
        super(STMGraph, self).__init__()
        [in_dim, num_hidden, out_dim] = hidden_dims
        # self.enc_mask_token = nn.Parameter(torch.zeros(size=(1, in_dim)))
        # self.dec_mask_token = nn.Parameter(torch.zeros(size=(1, out_dim)))
        # self.reset_parameters_for_token()
        self.encoding_mask_noise = encoding_mask_noise(hidden_dims)
        self.random_remask = random_remask(hidden_dims)
        self._delayed_ema_epoch = delayed_ema_epoch
        self._momentum = momentum
        self.conv1 = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2 = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv3 = GATConv(out_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv4 = GATConv(num_hidden, in_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv1_ema = GATConv(in_dim, num_hidden, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.conv2_ema = GATConv(num_hidden, out_dim, heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        for p in self.conv1_ema.parameters():
            p.requires_grad = False
            p.detach_()
        for p in self.conv2_ema.parameters():
            p.requires_grad = False
            p.detach_()
    # def reset_parameters_for_token(self):
    #     nn.init.xavier_normal_(self.enc_mask_token.data, gain=1.414)#
    #     nn.init.xavier_normal_(self.dec_mask_token.data, gain=1.414)
    def ema_update(self):
        def update(student, teacher):
            with torch.no_grad():
            # m = momentum_schedule[it]  # momentum parameter
                m = self._momentum
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
        update(self.conv1, self.conv1_ema)
        update(self.conv2, self.conv2_ema)

    def forward(self, epoch, features, edge_index, mask_rate=0.5,replace_rate=0.05):
        x, mask_nodes, keep_nodes = self.encoding_mask_noise(features, mask_rate=mask_rate, replace_rate=replace_rate)
        h1 = F.elu(self.conv1(x, edge_index))
        h2 = self.conv2(h1, edge_index, attention=False)
        x_ema = features
        with torch.no_grad():
            h1_ema = F.elu(self.conv1_ema(x_ema, edge_index))
            h2_ema = self.conv2_ema(h1_ema, edge_index, attention=False)

        # self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        # self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        # self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        # self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        self.conv3.lin_l.data = self.conv2.lin_l.transpose(0, 1)
        self.conv3.lin_r.data = self.conv2.lin_r.transpose(0, 1)
        self.conv4.lin_l.data = self.conv1.lin_l.transpose(0, 1)
        self.conv4.lin_r.data = self.conv1.lin_r.transpose(0, 1)

        h2_1,_,_=self.random_remask(h2, remask_rate=mask_rate)
        h2_2,_,_=self.random_remask(h2, remask_rate=mask_rate)
        h3_1 = F.elu(self.conv3(h2_1, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4_1 = self.conv4(h3_1, edge_index, attention=False)
        
        h3_2 = F.elu(self.conv3(h2_2, edge_index, attention=True,
                              tied_attention=self.conv1.attentions))
        h4_2 = self.conv4(h3_2, edge_index, attention=False)

        if epoch >= self._delayed_ema_epoch:
            self.ema_update()
        return h2, h2_ema, h4_1, h4_2, mask_nodes, keep_nodes, self.conv1.attentions  # F.log_softmax(x, dim=-1)

    # def encoding_mask_noise(self, x, mask_rate=0.5, replace_rate=0.05):
    #     # num_nodes = g.num_nodes()
    #     num_nodes = x.size()[0]
    #     perm = torch.randperm(num_nodes, device=x.device)
    #     num_mask_nodes = int(mask_rate * num_nodes)
    #     mask_token_rate = 1-replace_rate
    #     # random masking
    #     num_mask_nodes = int(mask_rate * num_nodes)
    #     mask_nodes = perm[: num_mask_nodes]
    #     keep_nodes = perm[num_mask_nodes: ]
        
    #     if replace_rate > 0.0:
    #         num_noise_nodes = int(replace_rate * num_mask_nodes)
    #         perm_mask = torch.randperm(num_mask_nodes, device=x.device)
    #         token_nodes = mask_nodes[perm_mask[: int(mask_token_rate * num_mask_nodes)]]
    #         noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
    #         noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

    #         out_x = x.clone()
    #         # out_x[token_nodes] = torch.zeros_like(out_x[token_nodes])
    #         out_x[token_nodes] = 0.0
    #         out_x[noise_nodes] = x[noise_to_be_chosen]
    #         # out_x[noise_nodes] = torch.add(x[noise_to_be_chosen], out_x[noise_nodes]) 
    #     else:
    #         out_x = x.clone()
    #         token_nodes = mask_nodes
    #         # out_x[token_nodes] = torch.zeros_like(out_x[token_nodes]) 
    #         out_x[mask_nodes] = 0.0

    #     out_x[token_nodes] += self.enc_mask_token
    #     # enc_mask_token_expanded = self.enc_mask_token.expand(int(mask_token_rate * num_mask_nodes), -1)
    #     # out_x[token_nodes] = torch.add(enc_mask_token_expanded, out_x[token_nodes])
    #     # use_g = g.clone()
    #     return out_x, mask_nodes, keep_nodes
    
    # def random_remask(self,rep,remask_rate=0.5):
    #     num_nodes = rep.size()[0]
    #     # num_nodes = g.num_nodes()
    #     perm = torch.randperm(num_nodes, device=rep.device)
    #     num_remask_nodes = int(remask_rate * num_nodes)
    #     remask_nodes = perm[: num_remask_nodes]
    #     rekeep_nodes = perm[num_remask_nodes: ]

    #     out_rep = rep.clone()
    #     out_rep[remask_nodes] = 0.0
    #     out_rep[remask_nodes] += self.dec_mask_token
    #     # out_rep[remask_nodes] = torch.zeros_like(rep[remask_nodes])  # 使用torch.zeros_like来创建相同形状的零张量
    #     # dec_mask_token_expanded = self.dec_mask_token.expand(num_remask_nodes, -1)
    #     # out_rep[remask_nodes] = torch.add(dec_mask_token_expanded, out_rep[remask_nodes])
    #     return out_rep, remask_nodes, rekeep_nodes
