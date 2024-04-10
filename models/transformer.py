import math


import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

import torch


from torch import Tensor, device, dtype
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import numpy as np


from models.utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

from torch.nn.init import uniform_

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x



def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    return float(-np.log((1 - prior_prob) / prior_prob))  # return bias_init

def linear_init_(module):
    bound = 1 / math.sqrt(module.weight.shape[0])
    uniform_(module.weight, -bound, bound)
    if hasattr(module, 'bias') and module.bias is not None:
        uniform_(module.bias, -bound, bound)

class RTDETRDecoder(nn.Module):

    def __init__(
            self,
            nc=80,
            ch=(512, 1024, 2048),
            hidden_dim=256,
            num_queries=300,
            strides=(8, 16, 32),  # TODO
            nl=3,
            num_decoder_points=4,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.,
            act=nn.ReLU(),
            eval_idx=-1,
            # training args
            num_denoising=100,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learnt_init_query=False):
        super().__init__()
        assert len(ch) <= nl
        assert len(strides) == len(ch)
        for _ in range(nl - len(strides)):
            strides.append(strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = strides
        self.nl = nl
        self.nc = nc
        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers

        # backbone feature projection
        self._build_input_proj_layer(ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, act, nl,
                                                          num_decoder_points)
        self.decoder = DeformableTransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        # denoising part
        self.denoising_class_embed = nn.Embedding(nc, hidden_dim)
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.enc_score_head = nn.Linear(hidden_dim, nc)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hidden_dim, nc) for _ in range(num_decoder_layers)])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(num_decoder_layers)])

        self._reset_parameters()

    def forward(self, feats, gt_meta=None):
        # input projection and embedding
        memory, spatial_shapes, _ = self._get_encoder_input(feats)

        # prepare denoising training
        if self.training:
            raise NotImplementedError
            # denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
            #     get_contrastive_denoising_training_group(gt_meta,
            #                                 self.num_classes,
            #                                 self.num_queries,
            #                                 self.denoising_class_embed.weight,
            #                                 self.num_denoising,
            #                                 self.label_noise_ratio,
            #                                 self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask = None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits = self.decoder(target,
                                              init_ref_points_unact,
                                              memory,
                                              spatial_shapes,
                                              self.dec_bbox_head,
                                              self.dec_score_head,
                                              self.query_pos_head,
                                              attn_mask=attn_mask)
        if not self.training:
            out_logits = out_logits.sigmoid_()
        return out_bboxes, out_logits  # enc_topk_bboxes, enc_topk_logits, dn_meta

    def _reset_parameters(self):
        # class and bbox head init
        bias_cls = bias_init_with_prob(0.01)
        linear_init_(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            linear_init_(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.)
            constant_(reg_.layers[-1].bias, 0.)

        linear_init_(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)

    def _build_input_proj_layer(self, ch):
        self.input_proj = nn.ModuleList()
        for in_channels in ch:
            self.input_proj.append(
                nn.Sequential(nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, bias=False),
                              nn.BatchNorm2d(self.hidden_dim)))
        in_channels = ch[-1]
        for _ in range(self.nl - len(ch)):
            self.input_proj.append(
                nn.Sequential(nn.Conv2D(in_channels, self.hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
                              nn.BatchNorm2d(self.hidden_dim)))
            in_channels = self.hidden_dim

    def _generate_anchors(self, spatial_shapes, grid_size=0.05, dtype=torch.float32, device='cpu', eps=1e-2):
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(torch.arange(end=h, dtype=torch.float32),
                                            torch.arange(end=w, dtype=torch.float32),
                                            indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)

            valid_WH = torch.tensor([h, w]).to(torch.float32)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape([-1, h * w, 4]))

        anchors = torch.concat(anchors, 1)
        valid_mask = ((anchors > eps) * (anchors < 1 - eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors.to(device=device, dtype=dtype), valid_mask.to(device=device)

    def _get_encoder_input(self, feats):
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.nl > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.nl):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0]
        for feat in proj_feats:
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop()
        return feat_flatten, spatial_shapes, level_start_index

    def _get_decoder_input(self, memory, spatial_shapes, denoising_class=None, denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        anchors, valid_mask = self._generate_anchors(spatial_shapes, dtype=memory.dtype, device=memory.device)
        memory = torch.where(valid_mask, memory, 0)
        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)  # (bs, h*w, nc)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors  # (bs, h*w, 4)

        # (bs, topk)
        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)
        # extract region proposal boxes
        # (bs, topk_ind)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        topk_ind = topk_ind.view(-1)

        # Unsigmoided
        reference_points_unact = enc_outputs_coord_unact[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        enc_topk_bboxes = torch.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)
        if self.training:
            reference_points_unact = reference_points_unact.detach()
        enc_topk_logits = enc_outputs_class[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        else:
            target = output_memory[batch_ind, topk_ind].view(bs, self.num_queries, -1)
            if self.training:
                target = target.detach()
        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact, enc_topk_bboxes, enc_topk_logits


class DeformableTransformerDecoder(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(self,
                tgt,
                reference_points,
                src,
                src_spatial_shapes,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                src_padding_mask=None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points = None
        ref_points_detach = torch.sigmoid(reference_points)
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)
            output = layer(output, ref_points_input, src, src_spatial_shapes, src_padding_mask, attn_mask,
                           query_pos_embed)

            inter_ref_bbox = torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(torch.sigmoid(bbox_head[i](output) + inverse_sigmoid(ref_points)))
            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            ref_points_detach = inter_ref_bbox.detach() if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits)

class MSDeformAttn(nn.Module):
    """
    Original Multi-Scale Deformable Attention Module.
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model must be divisible by n_heads, but got {d_model} and {n_heads}')
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        assert _d_per_head * n_heads == d_model, '`d_model` must be divisible by `n_heads`'

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(
            1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
        """
        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        _, len_v = value.shape[:2]
        assert sum(s[0] * s[1] for s in value_spatial_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        n = reference_points.shape[-1]
        if n == 2:
            offset_normalizer = torch.as_tensor(value_spatial_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = reference_points[:, :, None, :, None, :] + add

        elif n == 4:
            add = sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            sampling_locations = reference_points[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f'Last dim of reference_points must be 2 or 4, but got {n}.')
        output = multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

class DeformableTransformerDecoderLayer(nn.Module):
    """
    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0., act=nn.ReLU(), n_levels=4, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                reference_points,
                src,
                src_spatial_shapes,
                src_padding_mask=None,
                attn_mask=None,
                query_pos=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        if attn_mask is not None:
            attn_mask = torch.where(attn_mask.astype('bool'), torch.zeros(attn_mask.shape, tgt.dtype),
                                    torch.full(attn_mask.shape, float('-inf'), tgt.dtype))
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, src_spatial_shapes,
                               src_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt



