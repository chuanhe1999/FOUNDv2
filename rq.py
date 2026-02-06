#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import atorch
import torch
import torch.nn as nn

from .vq import VectorQuantizer


class ResidualVectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e_list,
        e_dim,
        sk_epsilons,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
        use_linear=0,
        increase_first_commitment_loss=False,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.increase_first_commitment_loss = increase_first_commitment_loss

        self.vq_layers = nn.ModuleList(
            [
                VectorQuantizer(
                    n_e,
                    e_dim,
                    kmeans_init=self.kmeans_init,
                    kmeans_iters=self.kmeans_iters,
                    sk_epsilon=sk_epsilon,
                    sk_iters=sk_iters,
                    use_linear=use_linear,
                    beta=0.5
                    if index == 0 and self.increase_first_commitment_loss
                    else 0.25,
                )
                for index, (n_e, sk_epsilon) in enumerate(
                    zip(n_e_list, sk_epsilons)
                )
            ]
        )

    def get_codebook(self):
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()  # type: ignore
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        all_losses = []
        all_indices = []
        all_distances = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices, distance = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)
            all_distances.append(distance)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)
        all_distances = torch.stack(all_distances, dim=1)

        return x_q, residual, mean_losses, all_indices, all_distances

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
        # if torch.distributed.is_initialized():
        # print("broadcast emb tabular")
        torch.distributed.broadcast(emb, src=0)
        # print("broadcast emb tabular successful")

    @torch.no_grad()
    def record_codebook_usage(self, x):
        r"""
        Records codebook usage,
            currenly, only in the first layer.
        """
        quantizer = self.vq_layers[0]
        x_res, loss, indices, distance = quantizer(x, use_sk=False)
        return indices
