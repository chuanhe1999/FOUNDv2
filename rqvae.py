#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import atorch
import torch
from torch import nn

from .layers import MLPLayers
from .rq import ResidualVectorQuantizer
from .rq_group import GroupResidualVectorQuantizer


class RQVAE(nn.Module):
    r"""
    RQVAE for split shared codebook and modality specific codebook.
    """

    def __init__(
        self,
        in_dim=1024,  # embedding dim of each modality, except for the tabular data
        num_text_modality=10,
        tabular_in_dim=1711,
        share_n_e_each_layer: list = [512, 512],
        specific_n_e_list_each_layer: list = [
            [128, 128, 128],
            [128, 128, 128],
        ],
        e_dim=512,
        layers=[2048, 1024, 512, 256],
        dropout_prob=0.1,
        norm="layer_norm",
        loss_type="mse",
        increase_first_commitment_loss=False,
        simple_decoder=True,
        tabular_double_stream=False,
        tabular_seperate=False,
        num_tabular_list=[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
        # ...
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=[0.0, 0.0],
        sk_iters=100,
        use_linear=0,
        sk_epsilons_tabular=[0.0, 0.0],
        # ---
        inference_mode=False,
    ):
        r"""
        Args:
            share_n_e_each_layer: list of int, the number of codebook vectors to share in each layer
                Samples: [512, 512], 2 layers, and each layer has 512 share codes.
            specific_n_e_list_each_layer: list of list of int, the number of codebook vectors to use in each layer
                Samples: [[256, 256, 256, 256, 256], [256, 256, 256, 256, 256]], 2 layers, 5 modalities,
                         and each modality has 256 specific codes.
            inference_mode: if True, the model is in inference mode, no decoder.
            ...

            simple_decoder: if True, the decoder is a simple two-layer MLP.
            tabular_seperate: text_modalities and the tabular modality do not share codebook.
            num_tabular_list: if tabular_seperate, we apply seperated codebbok for tabular.

            ...
        """
        super(RQVAE, self).__init__()
        self.num_text_modality = num_text_modality
        self.tabular_seperate = tabular_seperate
        assert self.tabular_seperate
        self.tabular_double_stream = tabular_double_stream
        if self.tabular_double_stream:
            raise NotImplementedError
            self.num_tabular_modality = (
                2  # double stream (mse + corr) for tabular data
            )
        else:
            self.num_tabular_modality = 1
        if not self.tabular_seperate:
            self.num_total_modality = (
                self.num_text_modality + self.num_tabular_modality
            )
        else:
            self.num_total_modality = self.num_text_modality

        self.in_dim = in_dim
        self.tabular_in_dim = tabular_in_dim

        self.share_n_e_each_layer = share_n_e_each_layer
        self.specific_n_e_list_each_layer = specific_n_e_list_each_layer
        assert len(self.share_n_e_each_layer) == len(
            self.specific_n_e_list_each_layer
        )
        for l in self.specific_n_e_list_each_layer:
            if not tabular_seperate:
                assert (
                    len(l)
                    == self.num_text_modality + self.num_tabular_modality
                )
            else:
                assert len(l) == self.num_text_modality
        self.total_n_e_each_layer = [
            share + sum(specific)
            for share, specific in zip(
                self.share_n_e_each_layer, self.specific_n_e_list_each_layer
            )
        ]
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.norm = norm
        self.loss_type = loss_type
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters

        self.sk_epsilons = sk_epsilons
        assert len(sk_epsilons) == len(share_n_e_each_layer)

        self.sk_iters = sk_iters
        self.use_linear = use_linear
        self.increase_first_commitment_loss = increase_first_commitment_loss
        self.inference_mode = inference_mode

        self.embedding_encode_layer_dims = (
            [self.in_dim] + self.layers + [self.e_dim]
        )
        self.tabular_encode_layer_dims = (
            [self.tabular_in_dim] + self.layers + [self.e_dim]
        )
        self.text_encoders = nn.ModuleList(
            [
                MLPLayers(
                    layers=self.embedding_encode_layer_dims,
                    dropout=self.dropout_prob,
                    norm=self.norm,
                    add_final_norm=False,
                    add_final_dropout=True,
                )
                for _ in range(self.num_text_modality)
            ]
        )
        self.tabular_encoder = MLPLayers(
            layers=self.tabular_encode_layer_dims,
            dropout=self.dropout_prob,
            norm=self.norm,
            add_final_norm=False,
            add_final_dropout=True,
        )

        self.rq_layers = GroupResidualVectorQuantizer(
            share_n_e_each_layer=self.share_n_e_each_layer,
            specific_n_e_list_each_layer=self.specific_n_e_list_each_layer,
            sk_epsilons=self.sk_epsilons,
            e_dim=self.e_dim,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_iters=self.sk_iters,
            use_linear=self.use_linear,
            increase_first_commitment_loss=self.increase_first_commitment_loss,
        )

        if self.tabular_seperate:
            # apply seperated codebook for tabular
            # double stream for tabular
            self.num_tabular_list = num_tabular_list
            self.sk_epsilons_tabular = sk_epsilons_tabular
            assert len(self.sk_epsilons_tabular) == len(self.num_tabular_list)

            self.rq_tabular = nn.ModuleList(
                [
                    ResidualVectorQuantizer(
                        n_e_list=self.num_tabular_list,
                        e_dim=self.e_dim,
                        kmeans_init=self.kmeans_init,
                        kmeans_iters=self.kmeans_iters,
                        sk_epsilons=self.sk_epsilons_tabular,
                        sk_iters=self.sk_iters,
                        use_linear=self.use_linear,
                    )
                    for _ in range(self.num_tabular_modality)
                ]
            )

        if not self.inference_mode:  # no decoder in inference mode
            self.simple_decoder = simple_decoder
            if not self.simple_decoder:
                self.text_decoder_layers = self.embedding_encode_layer_dims[
                    ::-1
                ]
                self.tabular_decoder_layers = self.tabular_encode_layer_dims[
                    ::-1
                ]
            else:
                # simple decoder. 2 layer
                self.text_decoder_layers = [
                    self.e_dim,
                    self.in_dim,
                    self.in_dim,
                ]
                self.tabular_decoder_layers = [
                    self.e_dim,
                    self.tabular_in_dim,
                    self.tabular_in_dim,
                ]
            self.decoders = nn.ModuleList(
                [
                    MLPLayers(
                        layers=self.text_decoder_layers,
                        dropout=self.dropout_prob,
                        norm=self.norm,
                        add_final_norm=False,
                    )
                    for _ in range(self.num_text_modality)
                ]
                + [
                    MLPLayers(
                        layers=self.tabular_decoder_layers,
                        dropout=self.dropout_prob,
                        norm=self.norm,
                        add_final_norm=False,
                    )
                    for _ in range(self.num_tabular_modality)
                ]
            )

    def forward(
        self, text_x, tabular, text_mask, modality_split_index, use_sk=False
    ):
        r"""
        Args:
            text: [batch_size * num_text_modality (dynamic), embedding_dim]
            tabular: [batch_size, tabular_dim]
            text_mask: [batch_size, num_text_modality]
            modality_split_index: [num_text_modality + 1], mark the split index of modalities
        """
        x_index = torch.zeros_like(
            text_mask.view(-1), dtype=torch.long, device=text_mask.device
        ).fill_(-1)
        x_index[text_mask.view(-1)] = torch.arange(
            text_x.shape[0], device=text_mask.device
        )

        target_x = []  # x of each modality, for reconstruction loss
        xs = []
        modality_select = []
        for i in range(self.num_text_modality):
            select = torch.zeros_like(
                text_mask, dtype=torch.bool, device=text_mask.device
            )
            select[
                :, modality_split_index[i] : modality_split_index[i + 1]
            ] = True
            select = select.reshape(-1)
            this_modality_index = x_index[select]
            this_modality_index = this_modality_index[this_modality_index >= 0]
            modality_select.append(this_modality_index)
            this_modality_x = text_x[this_modality_index]
            target_x.append(this_modality_x)
            xs.append(self.text_encoders[i](this_modality_x))
        tabular_x = self.tabular_encoder(tabular)

        if not self.tabular_seperate:
            xs.append(tabular_x)

        # split_index = torch.cumsum(
        #     torch.tensor([0] + [x.shape[0] for x in xs], device=xs[0].device), dim=0
        # ) # cannot jit, replace to the following code
        lengths = torch.stack(
            [m.new_zeros(1, device=m.device).fill_(m.shape[0]) for m in xs]
        ).squeeze(-1)
        split_index = torch.cat(
            [torch.zeros(1, device=xs[0].device), lengths.cumsum(0)]
        ).to(torch.int64)

        xs = torch.cat(xs, dim=0)
        x_q, residual, text_indices, distances, text_rq_losses = (
            self.rq_layers(xs, split_index, use_sk=use_sk)
        )

        if self.tabular_seperate:
            tabular_rq_loss = []
            tabular_indices = []
            tabular_distances = []
            (
                _tabular_x_q,
                _tabular_x_residual,
                _tabular_rq_loss,
                _tabular_indices,
                _tabular_distances,
            ) = self.rq_tabular[0](tabular_x, use_sk=use_sk)
            x_q.append(_tabular_x_q)
            tabular_rq_loss.append(_tabular_rq_loss)
            tabular_indices.append(_tabular_indices)
            tabular_distances.append(_tabular_distances)
        else:
            tabular_rq_loss = None
            tabular_indices = None

        if self.inference_mode:
            # In inference mode, the returned text indices
            #       are rearanged to [batch_size, 280 (day), 4 (layer)]
            # The indices of incomplete modality are padded with  [-1, -1, ..., -1]

            # 1. modality split format -> sample split format, squeezed
            reversed_text_indices = torch.zeros_like(
                text_indices, dtype=torch.long, device=text_indices.device
            )
            # text_indices = text_indices.split(lengths.long().tolist()) # cannot trace.
            split_text_indices = []
            for i, (start, end) in enumerate(
                zip(split_index[:-1], split_index[1:])
            ):
                split_text_indices.append(text_indices[start:end])
            text_indices = split_text_indices

            for select, ind in zip(modality_select, text_indices):
                reversed_text_indices[select] = ind

            # 2. unsqueeze, apply [-1, -1, ..., -1] for incomplete modality
            size1 = torch.zeros(
                1, device=text_mask.device, dtype=torch.long
            ).fill_(text_mask.shape[0] * text_mask.shape[1])
            sample_text_indices = torch.zeros(
                size1,  # type: ignore
                text_indices[0].shape[1],
                dtype=torch.long,
                device=text_mask.device,
            ).fill_(-1)
            sample_text_indices[x_index >= 0] = reversed_text_indices
            sample_text_indices = sample_text_indices.view(
                text_mask.shape[0], text_mask.shape[1], -1
            )
            return sample_text_indices, tabular_indices

        # decoder
        decoder_out = []
        for _x, decoder in zip(x_q, self.decoders):
            decoder_out.append(decoder(_x))

        text_out = decoder_out[: -self.num_tabular_modality]
        tabular_out = decoder_out[-self.num_tabular_modality :]

        return (
            target_x,
            text_out,
            tabular_out,
            text_rq_losses,
            tabular_rq_loss,
            text_indices,
            tabular_indices,
        )

    @torch.no_grad()
    def get_codebook(self):
        text_codebook = self.rq_layers.get_codebook()

        if self.tabular_seperate:
            tabular_codebook = self.rq_tabular[0].get_codebook()
        else:
            tabular_codebook = None

        return text_codebook, tabular_codebook

    @torch.no_grad()
    def text_indices_to_histogram(self, indices):
        share_range = self.share_n_e_each_layer[0]
        specific_ranges = self.specific_n_e_list_each_layer[0]

        # 1. share
        share_indices = []
        specific_indices = []
        for _indices in indices:
            share_mask = _indices < share_range
            share_indices.append(_indices[share_mask])
            specific_indices.append(_indices[~share_mask])

        share_indices = torch.cat(share_indices)
        histograms = []
        min_index = 0
        for inds, nums in zip(
            [share_indices] + specific_indices, [share_range] + specific_ranges
        ):
            # histogram = torch.histc(inds.float(), bins=nums, min=min_index, max=min_index + nums).long()
            histogram = torch.bincount(inds - min_index, minlength=nums).long()
            histograms.append(histogram)
            min_index += nums
        return histograms

    @torch.no_grad()
    def tabular_indices_to_histogram(self, indices, num=1024):
        r"""
        Args:
            indices: Tensor [batch_size]
        """
        # histogram = torch.histc(indices.float(), bins=num, min=0, max=num - 1).long()
        histogram = torch.bincount(indices, minlength=num).long()
        return histogram

    @torch.no_grad()
    def respawn_dead_codebook(
        self, text_x, tabular, text_mask, modality_split_index, ratio=0.3
    ):
        r"""
        Args:
            Same as self.forward.
            ratio: float, the ratio of dead codebook to respawn.
        """
        world_size: int = atorch.world_size()  # type: ignore
        rank: int = atorch.rank()  # type: ignore
        # import pdb

        # if rank == 0:
        #     pdb.set_trace()

        (
            xs,
            text_indices,
            tabular_x,
            tabular_indices_mse,
            tabular_indices_corr,
        ) = self.record_codebook_usage(
            text_x,
            tabular,
            text_mask,
            modality_split_index,
        )
        # print("record codebook usage")

        # xs
        padded_xs_for_gather = []
        for i, x in enumerate(xs):
            _x = torch.zeros(50, x.shape[1], device=x.device)
            _x[: x.shape[0]] = x
            padded_xs_for_gather.append(_x)
        padded_xs_for_gather = torch.stack(padded_xs_for_gather, dim=0)
        if rank == 0:
            gather_list = [
                torch.zeros_like(padded_xs_for_gather)
                for _ in range(world_size)
            ]
            torch.distributed.gather(
                padded_xs_for_gather,
                gather_list=gather_list if rank == 0 else None,
                dst=0,
            )
            xs = (
                torch.stack(gather_list, dim=0)
                .transpose(0, 1)
                .reshape(self.num_total_modality, -1, xs[0].shape[1])
            )
            # print(f"{xs.shape=}")
            gathered_xs = []
            for x in xs:
                gathered_xs.append(x[x.sum(dim=1) != 0])

            share_xs = torch.cat(
                gathered_xs
            )  # random collect from all modalities
            share_xs = share_xs[
                torch.randperm(share_xs.shape[0], device=share_xs[0].device)
            ]
            share_xs = share_xs[: self.share_n_e_each_layer[0]]
        else:
            torch.distributed.gather(
                padded_xs_for_gather, gather_list=None, dst=0
            )

        # indices
        padded_indices_for_gather = []
        for index in text_indices:
            _x = torch.zeros(50, dtype=torch.long, device=index.device) - 1
            _x[: index.shape[0]] = index
            padded_indices_for_gather.append(_x)
        padded_indices_for_gather = torch.stack(
            padded_indices_for_gather, dim=0
        )
        if rank == 0:
            gather_list = [
                torch.zeros_like(padded_indices_for_gather)
                for _ in range(world_size)
            ]
            torch.distributed.gather(
                padded_indices_for_gather,
                gather_list=gather_list if rank == 0 else None,
                dst=0,
            )
            text_indices = (
                torch.stack(gather_list, dim=0)
                .transpose(0, 1)
                .reshape(self.num_total_modality, -1)
            )
            # print(f"{text_indices.shape=}")
            gathered_indices = []
            for indices in text_indices:
                mask = indices != -1
                gathered_indices.append(indices[mask])

            histograms = self.text_indices_to_histogram(gathered_indices)
        else:
            torch.distributed.gather(
                padded_indices_for_gather, gather_list=None, dst=0
            )

        if rank == 0:
            start = 0
            for x, hist, num in zip(
                [share_xs] + gathered_xs,
                histograms,
                [self.share_n_e_each_layer[0]]
                + self.specific_n_e_list_each_layer[0],
            ):
                part = hist < ratio * hist.sum() / hist.shape[0]
                # part may have more codes, than x.shape[0], random to the exact size
                index = torch.arange(part.shape[0], device=part.device)[part]
                index = index[
                    torch.randperm(index.shape[0], device=index.device)[
                        : x.shape[0]
                    ]
                ]
                # index to mask
                mask = torch.zeros_like(
                    part, dtype=torch.bool, device=part.device
                )
                mask[index] = True
                all_mask = torch.zeros(
                    self.total_n_e_each_layer[0],
                    dtype=torch.bool,
                    device=hist.device,
                )
                all_mask[start : start + num] = mask
                # print(f"{rank=}, enter respawn dead codebook")
                self.rq_layers.respawn_dead_codebook(x, all_mask)
                start += num
        else:
            for _ in [
                self.share_n_e_each_layer[0]
            ] + self.specific_n_e_list_each_layer[0]:
                # print(f"{rank=}, enter respawn dead codebook")
                self.rq_layers.respawn_dead_codebook(None, None)
        # print("rq layers respawn dead codebook")
        # print(f"{rank=},text respawn dead codebook finished.")

        # if self.tabular_seperate:
        #     # print(tabular_x.shape)
        #     gather_list = [torch.zeros_like(tabular_x) for _ in range(world_size)]
        #     torch.distributed.gather(
        #         tabular_x, gather_list=gather_list if rank == 0 else None, dst=0
        #     )
        #     tabular_x = torch.cat(gather_list, dim=0)
        #     print(tabular_x.shape)
        #     # tabular_x = torch.cat(torch.distributed.nn.all_gather(tabular_x), dim=0)
        #     # print("gather tabular features")
        #     # print(tabular_x.shape)
        #     # print(tabular_indices_mse.shape)
        #     gather_list = [torch.zeros_like(tabular_indices_mse) for _ in range(world_size)]
        #     torch.distributed.gather(
        #         tabular_indices_mse, gather_list=gather_list if rank == 0 else None, dst=0
        #     )
        #     tabular_indices_mse = torch.cat(gather_list, dim=0)
        #     print(tabular_indices_mse.shape)
        #     # tabular_indices_mse = torch.cat(
        #     #     torch.distributed.nn.all_gather(tabular_indices_mse), dim=0
        #     # )
        #     # print(tabular_indices_mse.shape)
        #     # print("gather tabular indices mse")
        #     # print(tabular_indices_mse.shape)
        #     # tabular_indices_corr = torch.cat(
        #     #     torch.distributed.nn.all_gather(tabular_indices_corr), dim=0
        #     # )
        #     if rank == 0:
        #         print("Start respawn dead codebook for tabular mse")
        #         tabular_histogram = self.tabular_indices_to_histogram(
        #             tabular_indices_mse, num=self.num_tabular_list[0]
        #         )
        #         print("tabular histogram.")
        #         print(tabular_histogram.shape)
        #         print(tabular_histogram.device)
        #         mask_t = (
        #             tabular_histogram
        #             < ratio * tabular_histogram.sum() / tabular_histogram.shape[0]
        #         )
        #         print("calc tabular mask")
        #         print(tabular_x.shape)

        #         index = torch.arange(mask_t.shape[0], device=mask_t.device)[mask_t]
        #         index = index[
        #             torch.randperm(index.shape[0], device=index.device)[: tabular_x.shape[0]]
        #         ]
        #         mask_t = torch.zeros_like(mask_t, dtype=torch.bool, device=mask_t.device)
        #         mask_t[index] = True
        #         print(f"{rank=}, enter respawn dead codebook")
        #         self.rq_tabular[0].respawn_dead_codebook(tabular_x, mask)  # type: ignore
        #     else:
        #         print(f"{rank=}, enter respawn dead codebook")
        #         self.rq_tabular[0].respawn_dead_codebook(None, None)  # type: ignore

        #     # tabular_histogram = self.tabular_indices_to_histogram(
        #     #     tabular_indices_corr, num=self.num_tabular_list[0]
        #     # )
        #     # mask = (
        #     #     tabular_histogram < ratio * tabular_histogram.sum() / tabular_histogram.shape[0]
        #     # )  # type: ignore
        #     # self.rq_tabular[1].respawn_dead_codebook(tabular_x, mask)  # type: ignore

        #     print("tablar respawn dead codebook finished.")

    @torch.no_grad()
    def record_codebook_usage(
        self, text_x, tabular, text_mask, modality_split_index
    ):
        r"""
        currenly, only used in the first rq layer.
        """
        # print("enter record codebook usage")
        # text
        x_index = torch.zeros_like(
            text_mask.view(-1), dtype=torch.long, device=text_mask.device
        ).fill_(-1)
        x_index[text_mask.view(-1)] = torch.arange(
            text_x.shape[0], device=text_mask.device
        )

        xs = []
        for i in range(self.num_text_modality):
            select = torch.zeros_like(
                text_mask, dtype=torch.bool, device=text_mask.device
            )
            select[
                :, modality_split_index[i] : modality_split_index[i + 1]
            ] = True
            this_modality_index = x_index[select.view(-1)]
            this_modality_index = this_modality_index[this_modality_index >= 0]
            this_modality_x = text_x[this_modality_index]
            xs.append(self.text_encoders[i](this_modality_x))
        tabular_x = self.tabular_encoder(tabular)

        if not self.tabular_seperate:
            xs.append(tabular_x)
        split_index = torch.cumsum(
            torch.tensor([0] + [len(x) for x in xs], device=xs[0].device),
            dim=0,
        )
        xs = torch.cat(xs, dim=0)
        text_indices = self.rq_layers.record_codebook_usage(xs, split_index)

        split = [
            int(x1.item()) - int(x2.item())
            for x1, x2 in zip(split_index[1:], split_index[:-1])
        ]
        xs = torch.split(xs, split)
        text_indices = torch.split(text_indices, split)
        choose_indices = [torch.randperm(x.shape[0])[:50] for x in xs]
        xs = [xs[i][choose_indices[i]] for i in range(len(xs))]
        text_indices = [
            text_indices[i][choose_indices[i]]
            for i in range(len(text_indices))
        ]

        if not self.tabular_seperate:
            return (
                xs,
                text_indices,
                None,
                None,
                None,
            )
        else:
            tabular_indices_mse = self.rq_tabular[0].record_codebook_usage(
                tabular_x
            )  # type: ignore
            tabular_indices_corr = None
            # tabular_indices_corr = self.rq_tabular[1].record_codebook_usage(tabular_x)  # type: ignore
            return (
                xs,
                text_indices,
                tabular_x,
                tabular_indices_mse,
                tabular_indices_corr,
            )
