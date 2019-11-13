from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

import util


class Loss(object):
    """
    Abstract loss class.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def compute_dis_loss(
        self, y_real: torch.Tensor, y_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute discriminator loss.

        Parameters
        ----------
        y_real: torch.Tensor
            Discriminator output against real samples

        y_fake: torch.Tensor
            Discriminator output against fake samples

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_gen_loss(
        self, y_fake_i: torch.Tensor, y_fake_v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute generator loss.

        Parameters
        ----------
        y_fake_i: torch.Tensor
            Output of the image discriminator.

        y_fake_v: torch.Tensor
            Output of the video discriminator.

        Returns
        -------
        loss : torch.Tensor
            Loss value.
        """
        raise NotImplementedError()


class AdversarialLoss(Loss):
    """
    Adversarial loss, a implementation of Loss class.
    """

    def __init__(self):
        super().__init__()
        self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.device = util.current_device()

    def compute_dis_loss(
        self, y_real: torch.Tensor, y_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adversarial loss for the discriminator.

        Parameters
        ----------
        y_real: torch.Tensor
            Discriminator output against real samples

        y_fake: torch.Tensor
            Discriminator output against fake samples

        Returns
        -------
        loss : torch.Tensor
            Adversarial loss.
        """
        ones = torch.ones_like(y_real, device=self.device)
        zeros = torch.zeros_like(y_fake, device=self.device)

        loss = self.loss_func(y_real, ones) / y_real.numel()
        loss += self.loss_func(y_fake, zeros) / y_fake.numel()

        return loss

    def compute_gen_loss(
        self, y_fake_i: torch.Tensor, y_fake_v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adversarial loss for the generator.

        Parameters
        ----------
        y_fake_i: torch.Tensor
            Output of the image discriminator.

        y_fake_v: torch.Tensor
            Output of the video discriminator.

        Returns
        -------
        loss : torch.Tensor
            Adversarial loss.
        """
        ones_i = torch.ones_like(y_fake_i, device=self.device)
        ones_v = torch.ones_like(y_fake_v, device=self.device)

        loss = self.loss_func(y_fake_i, ones_i) / y_fake_i.numel()
        loss += self.loss_func(y_fake_v, ones_v) / y_fake_v.numel()

        return loss


class HingeLoss(Loss):
    """
    Hinge loss, a implementation of Loss class.
    """

    def __init__(self):
        super().__init__()
        self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.device = util.current_device()

    def compute_dis_loss(
        self, y_real: torch.Tensor, y_fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hinge loss for the discriminator.

        Parameters
        ----------
        y_real: torch.Tensor
            Discriminator output against real samples

        y_fake: torch.Tensor
            Discriminator output against fake samples

        Returns
        -------
        loss : torch.Tensor
            Hinge loss.
        """
        loss = torch.mean(nn.functional.relu(1.0 - y_real))
        loss += torch.mean(nn.functional.relu(1.0 + y_fake))

        return loss

    def compute_gen_loss(
        self, y_fake_i: torch.Tensor, y_fake_v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hinge loss for the generator.

        Parameters
        ----------
        y_fake_i: torch.Tensor
            Output of the image discriminator.

        y_fake_v: torch.Tensor
            Output of the video discriminator.

        Returns
        -------
        loss : torch.Tensor
            Hinge loss.
        """
        loss = torch.mean(nn.functional.softplus(-y_fake_i))
        loss += torch.mean(nn.functional.softplus(-y_fake_v))

        return loss
