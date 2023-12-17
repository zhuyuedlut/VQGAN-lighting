# -*- coding: utf-8 -*-
"""
# @project    : VQGAN-lighting
# @author     : https://github.com/zhuyuedlut
# @date       : 2023/12/13 11:17
# @brief      : 
"""
import itertools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MeanMetric
from torchvision import utils as vutils

from src.models.loss.lpips import LPIPS


class VQGANModule(LightningModule):
    def __init__(
            self,
            generator: nn.Module,
            discriminator: nn.Module,
            optimizer1: torch.optim.Optimizer,
            optimizer2: torch.optim.Optimizer,
            disc_factor: float,
            disc_start: int,
            perceptual_loss_factor: float,
            l2_loss_factor: float,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['generator', 'discriminator'])

        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.perceptual_loss = LPIPS().eval().to(self.device)

        self.train_vq_loss = MeanMetric()
        self.train_gan_loss = MeanMetric()

        self.val_vq_loss = MeanMetric()
        self.val_gan_loss = MeanMetric()

        self.automatic_optimization = False

    def setup(self, stage: str) -> None:
        self.train_dataset_length = len(self.trainer.datamodule.train_dataset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vqgan(x)

    def on_train_start(self) -> None:
        self.val_vq_loss.reset()
        self.val_gan_loss.reset()

    def training_step(self, imgs: torch.Tensor, batch_idx: int):
        decoded_images, _, q_loss = self.generator(imgs)
        perceptual_loss = self.perceptual_loss(imgs, decoded_images)
        rec_loss = torch.abs(imgs - decoded_images)
        nll_loss = self.hparams.perceptual_loss_factor * perceptual_loss + self.hparams.l2_loss_factor * rec_loss
        nll_losss = nll_loss.mean()

        disc_real = self.discriminator(imgs)
        disc_fake = self.discriminator(decoded_images)
        disc_factor = self.generator.adopt_weight(
            self.hparams.disc_factor, self.current_epoch * self.train_dataset_length + batch_idx,
            threshold=self.hparams.disc_start
        )

        g_loss = -torch.mean(disc_fake)
        λ = self.generator.calculate_lambda(nll_losss, g_loss)
        loss_vq = nll_losss + q_loss + disc_factor * λ * g_loss
        self.train_vq_loss.update(loss_vq)
        self.log("train/vq_loss", self.train_vq_loss.compute(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        d_loss_real = torch.mean(F.relu(1. - disc_real))
        d_loss_fake = torch.mean(F.relu(1. + disc_fake))
        loss_gan = disc_factor * .5 * (d_loss_real + d_loss_fake)
        self.train_gan_loss.update(loss_gan)
        self.log("train/gan_loss", self.train_gan_loss.compute(), on_step=True, on_epoch=True, prog_bar=True,
                 logger=True)

        optimizer1, optimizer2 = self.optimizers()
        optimizer1.zero_grad()
        loss_vq.backward(retain_graph=True)

        optimizer2.zero_grad()
        loss_gan.backward()

        optimizer1.step()
        optimizer2.step()

        if batch_idx % 10 == 0:
            with torch.no_grad():
                both = torch.cat((imgs[:4], decoded_images.add(1).mul(0.5)[:4]))
                os.makedirs("results", exist_ok=True)
                vutils.save_image(both, os.path.join("results", f"{self.current_epoch}_{batch_idx}.jpg"), nrow=4)

    def validation_step(self, imgs: torch.Tensor, batch_idx: int) -> None:
        with torch.enable_grad():
            decoded_images, _, q_loss = self.generator(imgs)

            disc_real = self.discriminator(imgs)
            disc_fake = self.discriminator(decoded_images)
            disc_factor = self.hparams.disc_factor
            perceptual_loss = self.perceptual_loss(imgs, decoded_images)
            rec_loss = torch.abs(imgs - decoded_images)
            nll_loss = self.hparams.perceptual_loss_factor * perceptual_loss + self.hparams.l2_loss_factor * rec_loss
            nll_losss = nll_loss.mean()
            g_loss = -torch.mean(disc_fake)

            λ = self.generator.calculate_lambda(nll_losss, g_loss)
            loss_vq = nll_losss + q_loss + disc_factor * λ * g_loss
            self.val_vq_loss.update(loss_vq)
            self.log("val/vq_loss", self.val_vq_loss.compute(), on_step=False, on_epoch=True, prog_bar=True,
                     logger=True)

            d_loss_real = torch.mean(F.relu(1. - disc_real))
            d_loss_fake = torch.mean(F.relu(1. + disc_fake))
            loss_gan = disc_factor * .5 * (d_loss_real + d_loss_fake)
            self.val_gan_loss.update(loss_gan)
            self.log("val/gan_loss", self.val_gan_loss.compute(), on_step=True, on_epoch=True, prog_bar=True,
                     logger=True)

            optimizer1, optimizer2 = self.optimizers()
            optimizer1.zero_grad(set_to_none=True)
            optimizer2.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer1 = self.hparams.optimizer1(
            params=list(itertools.chain(
                self.generator.encoder.parameters(),
                self.generator.decoder.parameters(),
                self.generator.codebook.parameters(),
                self.generator.quant_conv.parameters(),
                self.generator.post_quant_conv.parameters()
            )),
        )
        optimizer2 = self.hparams.optimizer2(params=self.discriminator.parameters())

        return optimizer1, optimizer2
