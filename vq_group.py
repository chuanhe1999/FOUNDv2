#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import kmeans, sinkhorn_algorithm


class GroupVectorQuantizer(nn.Module):
    def __init__(
        self,
        share_n_e: int,
        specific_n_e_list: list,
        e_dim=256,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.01,
        sk_iters=100,
        use_linear=0,
    ):
        r"""
        Args:
            share_n_e: number of codes shared by all modalities.
            specific_n_e_list: list of number of codes of each modality.
        """
        super().__init__()
        self.share_n_e = share_n_e
        self.specific_n_e_list = specific_n_e_list
        self.specific_n_e_sum = sum(specific_n_e_list)

        self.total_n_e = share_n_e + self.specific_n_e_sum
        self.specific_prefix_sum = torch.cumsum(
            torch.tensor([self.share_n_e] + specific_n_e_list), dim=0
        ).tolist()

        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters
        self.use_linear = use_linear

        self.embedding = nn.Embedding(self.total_n_e, self.e_dim)

        if not kmeans_init:
            self.initted = True
            # self.embedding.weight.data.uniform_(
            #     -1.0 / self.total_n_e, 1.0 / self.total_n_e
            # )
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

        if use_linear == 1:
            self.codebook_projection = torch.nn.Linear(self.e_dim, self.e_dim)
            torch.nn.init.normal_(
                self.codebook_projection.weight, std=self.e_dim**-0.5
            )
        self.assert_flag = False

    def get_codebook(self):
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        # get quantized latent vectors
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, split_index: list, use_sk=False):
        r"""
        Args:
            x: [N, e_dim]
            split_index: [num_modality] list of split index for each modality.
        """
        if not self.assert_flag:
            assert len(split_index) == len(self.specific_n_e_list) + 1
            assert x.shape[0] == split_index[-1]
            self.assert_flag = True

        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        if self.use_linear == 1:
            embeddings_weight = self.codebook_projection(self.embedding.weight)
        else:
            embeddings_weight = self.embedding.weight

        # Calculate the L2 Norm between latent and Embedded weights
        d = (
            torch.sum(latent**2, dim=1, keepdim=True)
            + torch.sum(embeddings_weight**2, dim=1, keepdim=True).t()
            - 2 * torch.matmul(latent, embeddings_weight.t())
        )

        # group split
        mask = torch.zeros(self.total_n_e, device=d.device, dtype=torch.bool)
        mask[: self.share_n_e] = True
        for i, (start, end) in enumerate(
            zip(split_index[:-1], split_index[1:])
        ):
            mask[
                self.specific_prefix_sum[i] : self.specific_prefix_sum[i + 1]
            ] = True
            # increase distance for non-specific codes
            # d[start:end, ~mask] = d.max().item() * 10
            d[start:end, ~mask] = 10000000
            mask[
                self.specific_prefix_sum[i] : self.specific_prefix_sum[i + 1]
            ] = False

        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)
            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        if self.use_linear == 1:
            x_q = F.embedding(indices, embeddings_weight).view(x.shape)
        else:
            x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        # we split loss for each modality
        q_losses = []
        for start, end in zip(split_index[:-1], split_index[1:]):
            _x = x[start:end]
            _x_q = x_q[start:end]
            commitment_loss = F.mse_loss(_x_q.detach(), _x)
            codebook_loss = F.mse_loss(_x_q, _x.detach())
            loss = codebook_loss + self.beta * commitment_loss
            q_losses.append(loss)

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, indices, d, q_losses


if __name__ == "__main__":
    # from pytorch_memlab import MemReporter

    vq = GroupVectorQuantizer(
        512,
        [256, 256],
    ).cuda(0)
    # reporter = MemReporter(vq)
    while True:
        x = torch.randn(10000, 256).cuda(0)
        split_index = [0, 5000, 10000]
        x_q, loss, indices, d = vq(x, split_index)
        # reporter.report(verbose=True)
        print(1)
