import torch.nn as nn
import torch
import torch.nn.functional as F

from torchmetrics.classification import BinaryAccuracy
import matplotlib.pyplot as plt

from .base import LightningBase

class LightningGAN(LightningBase):
    """
    Lightning implementation of a customised Generative Adversarial Network.

    This class is a custom implementation of a GAN, which uses a custom generator and discriminator
    architecture. The model is trained using a specified loss function, and the model architecture
    is provided by the user.

    We implement a range of modifications that in theory stabilise GAN training:
    - Gradient penalty
    - PatchGAN discriminator
    - Content loss

    We expect a range of weightings for each of these contributions, provided by a loss_config.
    This is implemented in the set_loss_function method.

    We override much of LightningBase as GAN training complicates matters significantly.
    """

    def __init__(self, model, loss_config, optimizer_config, **kwargs):
        super().__init__(**kwargs)

        self.automatic_optimization = False
        self.model = model
        self.repeat_factor = None

        self.optimizer_config = optimizer_config
        self.loss_function = self.set_loss_function(loss_config)

        self.accuracy = BinaryAccuracy()

        self.is_generator_warming_up = lambda train_step: (train_step > 5000)

    def forward(self, x):
        return self.model(x)

    def compute_gradient_penalty(self, model, real_samples, fake_samples, condition):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((real_samples.size(0), 1, 1, 1)).to(real_samples.device)
        # Get random interpolation between real and fake samples
        interpolates = (
            alpha * real_samples + ((1 - alpha) * fake_samples)
        ).requires_grad_(True)
        interpolates = interpolates.to(real_samples.device)
        d_interpolates = model.discriminator(condition, interpolates)[0]
        fake = (
            torch.Tensor(real_samples.shape[0], 1)
            .fill_(1.0)
            .to(real_samples.device)
            .requires_grad_(True)
        )
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1).to(real_samples.device)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def set_loss_function(self, loss_config):
        lambda_gp = loss_config.lambda_gp
        content_loss_scaling = loss_config.lambda_content
        lambda_patch = 1.0
        self.repeat_factor = 1
        if loss_config.content_loss:
            repeat_factor = loss_config.content_batch
            self.repeat_factor = repeat_factor

            preprocess_inputs = lambda x: x.repeat(repeat_factor, 1, 1, 1)

            def content_loss_callable(prediction, target):
                ensemble_mean = prediction.reshape(
                    (
                        repeat_factor,
                        -1,
                    )
                    + prediction.shape[1:]
                ).mean(dim=0)
                return F.l1_loss(ensemble_mean, target)

        else:
            preprocess_inputs = nn.Identity()
            content_loss_callable = lambda x, y: torch.Tensor([0.0]).to(x.device)

        def loss(model, target, condition, batch_idx, train=True, train_disc=False):
                
            img_dim = target.shape[-1]

            opt_g, opt_d = self.optimizers()
            condition = preprocess_inputs(condition)
            generated_images = model(condition)
            repeated_target = preprocess_inputs(target)
            disc_loss_weight = 1.0
            if batch_idx % 2 == 0 or not train_disc:
                self.toggle_optimizer(opt_g)

                content_loss = content_loss_callable(generated_images, target)
                if not train_disc:
                    total_loss = content_loss_scaling * content_loss

                    if train:
                        opt_g.zero_grad()
                        self.manual_backward(total_loss)
                        opt_g.step()
                        
                    loss_dict = {
                        "loss": total_loss,
                        "d_loss": 0.0,
                        "g_loss": 0.0,
                        "content": content_loss,
                        "grad_pen": 0.0,
                        "d_acc": 0.0,
                    }
                    self.untoggle_optimizer(opt_g)
                    return loss_dict, generated_images.detach().squeeze()


                labels = torch.ones(repeated_target.size(0)).to(
                    repeated_target.device
                )  # reward generator for fooling discriminator
                # hence positive sign here
                disc_outputs = model.discriminator(generated_images, condition)
                g_loss = -(F.softplus(disc_outputs[0]).mean() + lambda_patch * F.softplus(disc_outputs[1]).mean())
                if train:
                    adaptive_weight = self.calculate_adaptive_weight(content_loss_scaling * content_loss, g_loss, last_layer=self.get_last_layer())
                else:
                    adaptive_weight = 1.0
                scaled_g_loss = disc_loss_weight*adaptive_weight * g_loss 
                total_loss = scaled_g_loss + content_loss_scaling * content_loss

                acc = self.accuracy(
                    disc_outputs[0].squeeze(),
                    labels)

                if train:
                    opt_g.zero_grad()
                    self.manual_backward(total_loss)
                    opt_g.step()
                self.untoggle_optimizer(opt_g)
                loss_dict = {
                    "loss": total_loss,
                    "g_loss": scaled_g_loss,
                    "content": content_loss,
                    "d_acc": acc,
                }
            if batch_idx % 2 == 1:
                self.toggle_optimizer(opt_d)

                generated_images = generated_images.detach()
                real_labels = torch.ones(repeated_target.size(0)).to(repeated_target.device)

                disc_outputs_real = model.discriminator(repeated_target, condition)
                real_image_loss = -(
                    F.softplus(disc_outputs_real[0]).mean() + lambda_patch * F.softplus(disc_outputs_real[1]).mean()
                )

                disc_outputs_fake = model.discriminator(generated_images, condition)

                fake_labels = torch.zeros(repeated_target.size(0)).to(
                    repeated_target.device
                )
                fake_image_loss = (
                    F.softplus(disc_outputs_fake[0]).mean() + lambda_patch * F.softplus(disc_outputs_fake[1]).mean()
                )
                acc = self.accuracy(
                    torch.cat(
                        (disc_outputs_real[0].squeeze(), disc_outputs_fake[0].squeeze())
                    ),
                    torch.cat((real_labels, fake_labels)),
                )

                if train and not lambda_gp == 0:
                    gradient_penalty = self.compute_gradient_penalty(
                        model, repeated_target, generated_images, condition
                    )
                    d_loss = (real_image_loss + fake_image_loss) + gradient_penalty * lambda_gp
                else:
                    gradient_penalty = 0.0
                    d_loss = disc_loss_weight*(real_image_loss + fake_image_loss)

                if train:
                    opt_d.zero_grad()
                    self.manual_backward(d_loss)
                    opt_d.step()
                self.untoggle_optimizer(opt_d)
                total_loss = d_loss
                loss_dict = {
                    "loss": -d_loss,
                    "d_loss":d_loss,
                    "d_acc": acc,
                }

            return loss_dict, generated_images.detach().squeeze()

        return loss

    def shared_step(self, batch, batch_idx, eval_type="", train=True, train_disc=False):
        """"""
        loss, images = self.loss_function(self.model, batch[1], batch[0][0], batch_idx, train, train_disc)
        return {f"{eval_type}{key}": value for key, value in loss.items()}, images

    def training_step(self, batch, batch_idx):
        bs = batch[0][0].shape[0]
        train_disc =  self.is_generator_warming_up(self.trainer.global_step)# and batch_idx % 2 == 0)
        loss, images = self.shared_step(batch, batch_idx, train_disc=train_disc)

        self._log_loss(loss, batch_size=bs)
        if batch_idx - 10 % 1000 == 0:
            self.log_generator_images(images)
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        bs = batch[0][0].shape[0]
        train_disc =  self.is_generator_warming_up(self.trainer.global_step)
        loss, _ = self.shared_step(batch, batch_idx, "val_", train=False, train_disc=train_disc)
        self.model.train()
        self._log_loss(loss, "val_", batch_size=bs)

        return loss["val_loss"]

    def _log_loss(self, loss, eval_type: str = "", batch_size=None):

        for l in loss.keys():
            self.log(l, loss[l], batch_size=batch_size)

    def log_generator_images(self, images):
        images = images.cpu()
        repeated_images = images.reshape(
            (
                -1,
                self.repeat_factor,
            )
            + images.shape[1:]
        )
        for condition_idx in range(repeated_images.shape[1]):
            fig, axs = plt.subplots(
                1, self.repeat_factor, figsize=(self.repeat_factor * 6, 6)
            )
            if self.repeat_factor != 1:
                condition_imgs = repeated_images[:, condition_idx]
                for idx in range(self.repeat_factor):
                    axs[idx].imshow(
                        10**(2*condition_imgs[idx].float().numpy()) -1, vmin=0, vmax=30, cmap="Blues"
                    )
            else:
                axs.imshow(
                    10**( 2* repeated_images[condition_idx, 0].float().numpy()) - 1, vmin=0, vmax=30, cmap="Blues"
                )
            self.logger.experiment.log(
                {f"plot_{condition_idx}": fig}
            )  # wandb specific code
            plt.close()

    def configure_optimizers(self):

        lr = 1.0e-5

        opt_g = torch.optim.Adam(self.model.generator.parameters(), lr=2e-4)
        opt_d = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr)
        return (opt_g, opt_d)

    def get_last_layer(self):
        return self.model.generator.generator_unet.out[-1].weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 100.0).detach()
        d_weight = d_weight
        return d_weight