import torch
import torch.nn as nn


class IdentityPlasticity(nn.Module):

    def __init__(self):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args. 
        """

        super().__init__()

    def forward(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute corrected deformation gradient from deformation gradient tensor.

        Args:
            F (torch.Tensor): deformation gradient tensor (B, 3, 3).

        Returns:
            F_corrected (torch.Tensor): corrected deformation gradient tensor (B, 3, 3).
        """

        F_corrected = F # (B, 3, 3)

        return F_corrected
    
    
class PlasticinePlasticity(nn.Module):

    def __init__(self, youngs_modulus_log: float = {youngs_modulus_log}, poissons_ratio: float = {poissons_ratio}, yield_stress: float = {yield_stress}):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args.

        Args:
            youngs_modulus_log (float): log of Young's modulus.
            poissons_ratio (float): Poisson's ratio.
            yield_stress (float): yield stress.
        """

        super().__init__()
        self.youngs_modulus_log = nn.Parameter(torch.tensor(youngs_modulus_log))
        self.poissons_ratio = nn.Parameter(torch.tensor(poissons_ratio))
        self.yield_stress = nn.Parameter(torch.tensor(yield_stress))

    def forward(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute corrected deformation gradient from deformation gradient tensor.

        Args:
            F (torch.Tensor): deformation gradient tensor (B, 3, 3).

        Returns:
            F_corrected (torch.Tensor): corrected deformation gradient tensor (B, 3, 3).
        """

        youngs_modulus = self.youngs_modulus_log.exp()
        poissons_ratio = self.poissons_ratio
        yield_stress = self.yield_stress

        mu = youngs_modulus / (2 * (1 + poissons_ratio))

        U, sigma, Vh = torch.linalg.svd(F) # (B, 3, 3), (B, 3), (B, 3, 3)

        threshold = 0.01
        sigma = torch.clamp_min(sigma, threshold) # (B, 3)

        epsilon = torch.log(sigma) # (B, 3)
        epsilon_trace = epsilon.sum(dim=1, keepdim=True) # (B, 1)
        epsilon_bar = epsilon - epsilon_trace / 3 # (B, 3)
        epsilon_bar_norm = epsilon_bar.norm(dim=1, keepdim=True) + 1e-5 # (B, 1)

        delta_gamma = epsilon_bar_norm - yield_stress / (2 * mu) # (B, 1)

        yield_epsilon = epsilon - (delta_gamma / epsilon_bar_norm) * epsilon_bar # (B, 3)
        yield_sigma = torch.exp(yield_epsilon) # (B, 3)

        sigma = torch.where((delta_gamma > 0).view(-1, 1), yield_sigma, sigma) # (B, 3)
        F_corrected = torch.matmul(U, torch.matmul(torch.diag_embed(sigma), Vh)) # (B, 3, 3)

        return F_corrected
    
    
class VonMisesPlasticity(nn.Module):

    def __init__(self, youngs_modulus_log: float = {youngs_modulus_log}, poissons_ratio: float = {poissons_ratio}, alpha: float = {alpha}, cohesion: float = {cohesion}):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args.

        Args:
            youngs_modulus_log (float): log of Young's modulus.
            poissons_ratio (float): Poisson's ratio.
            alpha (float): alpha.
            cohesion (float): cohesion.
        """

        super().__init__()
        self.youngs_modulus_log = nn.Parameter(torch.tensor(youngs_modulus_log))
        self.poissons_ratio = nn.Parameter(torch.tensor(poissons_ratio))
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.cohesion = nn.Parameter(torch.tensor(cohesion))

    def forward(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute corrected deformation gradient from deformation gradient tensor.

        Args:
            F (torch.Tensor): deformation gradient tensor (B, 3, 3).

        Returns:
            F_corrected (torch.Tensor): corrected deformation gradient tensor (B, 3, 3).
        """

        youngs_modulus = self.youngs_modulus_log.exp()
        poissons_ratio = self.poissons_ratio
        alpha = self.alpha
        cohesion = self.cohesion

        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))

        U, sigma, Vh = torch.linalg.svd(F) # (B, 3, 3), (B, 3), (B, 3, 3)

        # prevent NaN
        thredhold = 0.05
        sigma = torch.clamp_min(sigma, thredhold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / 3
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)

        expand_epsilon = torch.ones_like(epsilon) * cohesion

        shifted_trace = trace - cohesion * 3
        cond_yield = (shifted_trace < 0).view(-1, 1)

        delta_gamma = epsilon_hat_norm + (3 * la + 2 * mu) / (2 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (torch.clamp_min(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        epsilon = torch.where(cond_yield, compress_epsilon, expand_epsilon)

        F_corrected = torch.matmul(torch.matmul(U, torch.diag_embed(epsilon.exp())), Vh)

        return F_corrected
    
    
class DruckerPragerPlasticity(nn.Module):
    def __init__(self, 
                 youngs_modulus_log: float = math.log(2.0e6), 
                 poissons_ratio: float = 0.4, 
                 friction_angle: float = 25.0, 
                 cohesion: float = 0.0) -> None:
        """
        Differentiable Drucker-Prager elastoplasticity model.

        Args:
            log_E (float): log of Young's modulus.
            nu (float): Poisson's ratio.
            friction_angle (float): Friction angle in degrees.
            cohesion (float): Cohesion (yield stress at zero confining pressure).
        """
        super().__init__()
        self.youngs_modulus_log = nn.Parameter(torch.tensor(youngs_modulus_log))
        self.poissons_ratio = nn.Parameter(torch.tensor(poissons_ratio))
        self.friction_angle = nn.Parameter(torch.tensor(friction_angle))
        self.cohesion = nn.Parameter(torch.tensor(cohesion))
        self.dim = 3  # for 3D tensors

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """
        Compute Drucker-Prager plastic correction for deformation gradient.

        Args:
            F (Tensor): Deformation gradient tensor (B, 3, 3).

        Returns:
            F_corrected (Tensor): Corrected deformation gradient tensor (B, 3, 3).
        """
        # Material parameters
        youngs_modulus = self.youngs_modulus_log.exp()
        poissons_ratio = self.poissons_ratio
        friction_angle = self.friction_angle
        cohesion = self.cohesion

        # Drucker-Prager parameter
        sin_phi = torch.sin(torch.deg2rad(friction_angle))
        alpha = torch.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi)

        mu = youngs_modulus / (2 * (1 + poissons_ratio))
        la = youngs_modulus * poissons_ratio / ((1 + poissons_ratio) * (1 - 2 * poissons_ratio))

        # Ensure broadcast shape
        if mu.dim() != 0:
            mu = mu.view(-1, 1)
        if la.dim() != 0:
            la = la.view(-1, 1)

        # Polar SVD
        U, sigma, Vh = torch.linalg.svd(F)

        # Prevent collapse
        threshold = 0.05
        sigma = torch.clamp_min(sigma, threshold)

        epsilon = torch.log(sigma)
        trace = epsilon.sum(dim=1, keepdim=True)
        epsilon_hat = epsilon - trace / self.dim
        epsilon_hat_norm = torch.linalg.norm(epsilon_hat, dim=1, keepdim=True)
        epsilon_hat_norm = torch.clamp_min(epsilon_hat_norm, 1e-10)  # avoid nan

        # Drucker-Prager yield condition
        shifted_trace = trace - cohesion * self.dim
        cond_yield = (shifted_trace < 0).view(-1, 1)

        # Compute return mapping correction
        delta_gamma = epsilon_hat_norm + (self.dim * la + 2 * mu) / (2 * mu) * shifted_trace * alpha
        compress_epsilon = epsilon - (torch.clamp_min(delta_gamma, 0.0) / epsilon_hat_norm) * epsilon_hat

        # If yield, apply correction; else, expand with cohesion only
        expand_epsilon = torch.ones_like(epsilon) * cohesion
        epsilon_corrected = torch.where(cond_yield, compress_epsilon, expand_epsilon)

        # Reconstruct corrected F
        F_corrected = torch.matmul(U, torch.matmul(torch.diag_embed(epsilon_corrected.exp()), Vh))

        return F_corrected

class SigmaPlasticity(nn.Module):

    def __init__(self):
        """
        Define trainable continuous physical parameters for differentiable optimization.
        Tentatively initialize the parameters with the default values in args.
        """

        super().__init__()

    def forward(self, F: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute corrected deformation gradient from deformation gradient tensor.

        Args:
            F (torch.Tensor): deformation gradient tensor (B, 3, 3).

        Returns:
            F_corrected (torch.Tensor): corrected deformation gradient tensor (B, 3, 3).
        """

        J = torch.det(F).view(-1, 1, 1) # (B, 1, 1)
        Je_1_3 = torch.pow(J, 1 / 3) # (B, 1, 1)
        sigma_corrected = Je_1_3.view(-1, 1).expand(-1, 3) # (B, 3)

        F_corrected = torch.diag_embed(sigma_corrected) # (B, 3, 3)

        return F_corrected
    

