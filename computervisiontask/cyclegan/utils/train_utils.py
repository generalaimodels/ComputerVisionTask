# utils/train_utils.py

import torch
from torch import nn
from typing import Tuple


def weights_init(m: nn.Module) -> None:
    """
    Initializes model weights.

    Args:
        m (nn.Module): Model layer.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.InstanceNorm2d):
        if m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def get_disc_loss(
    real_X: torch.Tensor,
    fake_X: torch.Tensor,
    disc_X: nn.Module,
    adv_criterion: nn.Module
) -> nn.Module:
    """
    Calculates the discriminator's loss.

    Args:
        real_X (torch.Tensor): Real images from domain X.
        fake_X (torch.Tensor): Fake images generated for domain X.
        disc_X (nn.Module): Discriminator for domain X.
        adv_criterion (nn.Module): Adversarial loss function.

    Returns:
        nn.Module: Discriminator loss.
    """
    disc_fake = disc_X(fake_X)
    disc_fake_loss = adv_criterion(disc_fake, torch.zeros_like(disc_fake).to(fake_X.device))

    disc_real = disc_X(real_X)
    disc_real_loss = adv_criterion(disc_real, torch.ones_like(disc_real).to(real_X.device))

    disc_loss = (disc_fake_loss + disc_real_loss) / 2
    return disc_loss


def get_gen_adversarial_loss(
    real_X: torch.Tensor,
    disc_Y: nn.Module,
    gen_XY: nn.Module,
    adv_criterion: nn.Module
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Calculates the generator's adversarial loss.

    Args:
        real_X (torch.Tensor): Real images from domain X.
        disc_Y (nn.Module): Discriminator for domain Y.
        gen_XY (nn.Module): Generator that maps X to Y.
        adv_criterion (nn.Module): Adversarial loss function.

    Returns:
        Tuple[nn.Module, torch.Tensor]: Adversarial loss and fake images.
    """
    fake_Y = gen_XY(real_X)
    fake_disc_Y = disc_Y(fake_Y)
    adversarial_loss = adv_criterion(fake_disc_Y, torch.ones_like(fake_disc_Y).to(fake_Y.device))
    return adversarial_loss, fake_Y


def get_identity_loss(
    real_X: torch.Tensor,
    gen_YX: nn.Module,
    identity_criterion: nn.Module
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Calculates the identity loss for the generator.

    Args:
        real_X (torch.Tensor): Real images from domain X.
        gen_YX (nn.Module): Generator that maps Y to X.
        identity_criterion (nn.Module): Identity loss function.

    Returns:
        Tuple[nn.Module, torch.Tensor]: Identity loss and preserved images.
    """
    identity_X = gen_YX(real_X)
    identity_loss = identity_criterion(real_X, identity_X)
    return identity_loss, identity_X


def get_cycle_consistency_loss(
    real_X: torch.Tensor,
    fake_Y: torch.Tensor,
    gen_YX: nn.Module,
    cycle_criterion: nn.Module
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Calculates the cycle consistency loss.

    Args:
        real_X (torch.Tensor): Real images from domain X.
        fake_Y (torch.Tensor): Fake images generated for domain Y.
        gen_YX (nn.Module): Generator that maps Y to X.
        cycle_criterion (nn.Module): Cycle consistency loss function.

    Returns:
        Tuple[nn.Module, torch.Tensor]: Cycle consistency loss and cycled images.
    """
    cycle_X = gen_YX(fake_Y)
    cycle_loss = cycle_criterion(real_X, cycle_X)
    return cycle_loss, cycle_X


def get_gen_loss(
    real_A: torch.Tensor,
    real_B: torch.Tensor,
    gen_AB: nn.Module,
    gen_BA: nn.Module,
    disc_A: nn.Module,
    disc_B: nn.Module,
    adv_criterion: nn.Module,
    identity_criterion: nn.Module,
    cycle_criterion: nn.Module,
    lambda_identity: float = 0.1,
    lambda_cycle: float = 10.0
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """
    Calculates the total generator loss.

    Args:
        real_A (torch.Tensor): Real images from domain A.
        real_B (torch.Tensor): Real images from domain B.
        gen_AB (nn.Module): Generator that maps A to B.
        gen_BA (nn.Module): Generator that maps B to A.
        disc_A (nn.Module): Discriminator for domain A.
        disc_B (nn.Module): Discriminator for domain B.
        adv_criterion (nn.Module): Adversarial loss function.
        identity_criterion (nn.Module): Identity loss function.
        cycle_criterion (nn.Module): Cycle consistency loss function.
        lambda_identity (float, optional): Weight for identity loss. Defaults to 0.1.
        lambda_cycle (float, optional): Weight for cycle consistency loss. Defaults to 10.0.

    Returns:
        Tuple[nn.Module, torch.Tensor, torch.Tensor]: Total generator loss and fake images.
    """
    adversarial_loss_AB, fake_B = get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
    adversarial_loss_BA, fake_A = get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
    adversarial_loss = adversarial_loss_AB + adversarial_loss_BA

    identity_loss_A, _ = get_identity_loss(real_A, gen_BA, identity_criterion)
    identity_loss_B, _ = get_identity_loss(real_B, gen_AB, identity_criterion)
    identity_loss = identity_loss_A + identity_loss_B

    cycle_consistency_loss_BA, _ = get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
    cycle_consistency_loss_AB, _ = get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
    cycle_consistency_loss = cycle_consistency_loss_BA + cycle_consistency_loss_AB

    gen_loss = adversarial_loss + lambda_identity * identity_loss + lambda_cycle * cycle_consistency_loss
    return gen_loss, fake_A, fake_B