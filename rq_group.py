#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import atorch
import torch
import torch.nn as nn

from .vq_group import GroupVectorQuantizer


class GroupResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        share_n_e_each_layer: list,
        specific_n_e_list_each_layer: list,
        sk_epsilons: list,
        e_dim=256,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
        use_linear=0,
        increase_first_commitment_loss=False,
    ):
        r"""
        Args:
            share_n_e_each_layer: list of int, the number of codebook vectors to share in each layer
                Samples: [512, 512], 2 layers, and each layer has 512 share codes.
            specific_n_e_list_each_layer: list of list of int, the number of codebook vectors to use in each layer
                Samples: [[256, 256, 256, 256, 256], [256, 256, 256, 256, 256]], 2 layers, 5 modalities,
                         and each modality has 256 specific codes.
        """
        super().__init__()
        self.share_n_e_each_layer = share_n_e_each_layer
        self.specific_n_e_list_each_layer = specific_n_e_list_each_layer

        self.e_dim = e_dim
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.increase_first_commitment_loss = increase_first_commitment_loss

        self.vq_layers = nn.ModuleList(
            [
                GroupVectorQuantizer(
                    share_n_e=share_n_e,
                    specific_n_e_list=specific_n_e_list,
                    e_dim=e_dim,
                    beta=0.5
                    if index == 0 and self.increase_first_commitment_loss
                    else 0.25,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                    use_linear=use_linear,
                )
                for index, (
                    share_n_e,
                    specific_n_e_list,
                    sk_epsilon,
                ) in enumerate(
                    zip(
                        self.share_n_e_each_layer,
                        self.specific_n_e_list_each_layer,
                        sk_epsilons,
                    )
                )
            ]
        )

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()  # type: ignore
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, split_index: list, use_sk=False):
        r"""
        Args:
            x: [N, e_dim]
            split_index: [num_modality] list of split index for each modality.
        """
        q_losses = []
        all_indices = []
        all_distances = []

        x_q = torch.zeros(self.e_dim, device=x.device)
        residual = x
        for quantizer in self.vq_layers:
            current_x_q, indices, distance, q_loss = quantizer(
                residual, split_index, use_sk=use_sk
            )
            residual = residual - current_x_q
            x_q = x_q + current_x_q
            q_loss = torch.stack(q_loss)

            q_losses.append(q_loss)
            all_indices.append(indices)
            all_distances.append(distance)

        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)

        split = [x1 - x2 for x1, x2 in zip(split_index[1:], split_index[:-1])]
        x_q = list(torch.split(x_q, split, dim=0))
        residual = list(torch.split(residual, split, dim=0))

        return x_q, residual, all_indices, all_distances, q_losses

    @torch.no_grad()
    def respawn_dead_codebook(self, x, mask):
        r"""
        Respawns dead codebook vectors,
            currenly, only in the first layer.
        """
        rank: int = atorch.rank()  # type: ignore
        emb: torch.Tensor = self.vq_layers[0].embedding.weight  # type: ignore | torch.nn.Embedding
        if rank == 0:
            # replace masked self.vq_layers[0].embedding with new_x randomed selected from x
            dead_ids = mask.nonzero(as_tuple=False).squeeze(-1)
            if dead_ids.numel() == 0:
                torch.distributed.broadcast(emb, src=0)
                return

            idx = torch.randperm(x.shape[0], device=x.device)[
                : dead_ids.numel()
            ]
            new_entries = x[idx]
            emb[dead_ids] = new_entries

        # broadcast emb to all ranks
        torch.distributed.broadcast(emb, src=0)

    @torch.no_grad()
    def record_codebook_usage(self, x, split_index):
        r"""
        Records codebook usage,
            currenly, only in the first layer.
        """
        quantizer = self.vq_layers[0]
        current_x_q, indices, distance, q_loss = quantizer(
            x, split_index, use_sk=False
        )
        return indices


if __name__ == "__main__":
    rq = GroupResidualVectorQuantizer(
        share_n_e_each_layer=[512, 512],
        specific_n_e_list_each_layer=[[256, 256], [256, 256]],
        sk_epsilons=[0, 0],
    )
    rq = rq.cuda(0)
    while True:
        x = torch.randn(10000, 256).cuda(0)
        split_index = [0, 5000, 10000]
        x_q, residual, mean_losses, all_indices, all_distances = rq(
            x, split_index
        )
        ...
