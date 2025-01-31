
from . import utils
from .nn import conv_nd
from .unet import ResBlock, Upsample
import torch.nn as nn
import functools
import torch
import numpy as np
from torch.nn.functional import unfold


class Generator(nn.Module):
    def __init__(self, generator_unet, latent_seed_network, side_latent_network):
        super(Generator, self).__init__()
        self.generator_unet = generator_unet
        self.latent_seed_network = latent_seed_network
        self.side_latent_network = side_latent_network

    def forward(self, latent, cond, side_latent):
        random_channel = self.latent_seed_network(latent)
        side_latent = self.side_latent_network(side_latent)
        x = self.generator_unet(random_channel, cond, side_latent)
        return x


class Discriminator(nn.Module):
    def __init__(
        self, discriminator_backbone, patch_classifier, global_classifier, patch_size
    ):
        super(Discriminator, self).__init__()
        self.discriminator_backbone = discriminator_backbone
        self.patch_classifier = patch_classifier
        self.global_classifier = global_classifier

        self.patch_size = patch_size

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        x = self.discriminator_backbone(x)
        patch_results = self.patch_classifier(self.create_patches(x))
        global_results = self.global_classifier(x)
        return global_results, patch_results

    def create_patches(self, backbone_outputs):
        patches = unfold(
            backbone_outputs,
            kernel_size=self.patch_size[-2],
            stride=self.patch_size[-2],
        )
        patches = patches.permute(0, 2, 1)
        return patches.reshape(
            -1, backbone_outputs.shape[1], self.patch_size[-2], self.patch_size[-2]
        )


class cWGAN_GP(nn.Module):
    """Conditional Wasserstein GAN with Gradient Penalty"""

    def __init__(self, config):
        super().__init__()

        self.input_dim = None
        self.nf = None

        self.generator = self._build_generator(config)
        config.model = config.discriminator
        self.discriminator = self._build_discriminator(config)

    def _build_generator(self, config):
        data_config = config.data
        generator_unet = utils.create_model(config)

        dropout = config.model.dropout
        use_checkpoint = False
        use_scale_shift_norm = False

        output_dim = 1  # TODO: needs fixing for new variables

        ResnetBlock = functools.partial(
            ResBlock,
            act=nn.SiLU,
            emb_channels=0,
            dropout=dropout,
            num_embeddings=0,
            dims=2,
            use_checkpoint=use_checkpoint,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        input_dim = data_config.image_size**2 // (4) ** 2
        self.input_dim = input_dim

        ### latent seed architecture loosely based on stylegan1 fig.
        num_dims = 32
        latent_seed_network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Unflatten(1, (4, 16, 16)),
            Upsample(channels=4, use_conv=True),
            conv_nd(2, 4, 32, 3, padding=1),
            ResnetBlock(channels=num_dims, out_channels=num_dims, up=True),
            ResnetBlock(channels=num_dims, out_channels=num_dims, up=True),
            conv_nd(2, num_dims, 1, 1),
        )

        nf = config.model.nf
        self.nf = nf
        temb_latent_network = nn.Sequential(
            nn.Linear(nf, nf),
            nn.SiLU(),
            nn.Linear(nf, nf),
            nn.SiLU(),
            nn.Linear(nf, nf),
            nn.SiLU(),
            nn.Linear(nf, nf),
            nn.SiLU(),
            nn.Linear(nf, nf),
        )

        generator = Generator(generator_unet, latent_seed_network, temb_latent_network)
        return generator

    def _build_discriminator(self, config):
        data_config = config.data
        img_dim = data_config.image_size
        # discriminator_unet = utils.create_model(config)
        cond_var_channels, output_channels = list(map(len, config.data.variables))
        dropout = config.model.dropout
        patch_size = (1, img_dim // 4, img_dim // 4)
        dims = 2

        ResnetBlock = functools.partial(
            ResBlock,
            act=nn.LeakyReLU,
            emb_channels=0,
            dropout=dropout,
            num_embeddings=0,
            dims=dims,
            use_checkpoint=False,
            use_scale_shift_norm=False,
        )

        output_dim = output_channels + cond_var_channels
        shared_extractor = nn.Sequential(
            conv_nd(2, output_dim, 128, 3, padding=1),
            ResnetBlock(channels=128, out_channels=128),
            ResnetBlock(channels=128, out_channels=128),
        )

        patch_discriminator = nn.Sequential(
            ResnetBlock(channels=128, out_channels=128, down=True),
            ResnetBlock(channels=128, out_channels=128, down=True),
            conv_nd(2, 128, 1, 3
            ),
            # nn.Sigmoid(),
        )
        global_discriminator_classifier = nn.Sequential(
            ResnetBlock(channels=128, out_channels=128),
            nn.MaxPool2d(2),
            ResnetBlock(channels=128, out_channels=128),
            nn.MaxPool2d(2),
            conv_nd(2, 128, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(data_config.image_size**2 // 4**3, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            # nn.Sigmoid(),
        )

        discriminator = Discriminator(
            shared_extractor,
            patch_discriminator,
            global_discriminator_classifier,
            patch_size,
        )
        return discriminator

    def forward(self, x):
        condition = x
        random_image_seed = torch.randn((condition.shape[0], self.input_dim)).to(
            condition.device
        )

        random_embedding = torch.randn((condition.shape[0], self.nf)).to(
            condition.device
        )

        return self.generator(random_image_seed, condition, random_embedding)
