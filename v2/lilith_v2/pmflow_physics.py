"""
Core PMFlow Library (Pushing-Medium Flow).

Implements the optical-mechanical analogy physics engine described in `all_formulas_fixed.tex`.

This is the mathematical heart of the system, replacing simple vector similarity with
active refractive index fields and flow dynamics.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

class PMField(nn.Module):
    """
    Implements a Pushing-Medium Field environment.
    
    Physics Model:
    The latent space is treated as a medium with a variable refractive index n(r).
    Thoughts (rays/particles) travel through this medium, being bent by:
    1. Static Mass (Attractors/Repulsors) -> Changes n(r)
    2. Flow Fields (Frame Dragging) -> Advects the particle
    3. Wave Perturbations (Noise/Inputs) -> Temporal modulation
    
    Formula Reference: all_formulas_fixed.tex
    """
    
    def __init__(self, latent_dim: int, device: str = "cpu"):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        
        # Physics Constants
        self.c = 1.0     # Speed of light in vacuum (baseline speed)
        self.G = 0.1     # Coupling constant
    
    def compute_refractive_index(self, r: torch.Tensor, 
                               centers: torch.Tensor, 
                               mus: torch.Tensor) -> torch.Tensor:
        """
        Formula 1: n(r) = 1 + sum(mu_i / |r - r_i|)
        
        r: [Batch, Dim] location of particle
        centers: [Num_Centers, Dim] locations of masses
        mus: [Num_Centers] strengths (positive=attractor, negative=repulsor)
        """
        n = torch.ones(r.shape[0], device=self.device)
        
        for i in range(centers.shape[0]):
            r_i = centers[i]
            mu_i = mus[i]
            
            # Distance |r - r_i|
            dist = torch.norm(r - r_i, dim=1) + 1e-6 # Avoid singularity
            
            # Perturbation
            dn = mu_i / dist
            n += dn
            
        return n

    def compute_flow_field(self, r: torch.Tensor, 
                         centers: torch.Tensor, 
                         omegas: torch.Tensor) -> torch.Tensor:
        """
        Formula 2: u_g(r) = sum(Omega_i x (r - r_i))
        
        Note: Cross product is strictly 3D. for N-D latent spaces, 
        we generalize "rotation" using a skew-symmetric matrix or simpler 
        tangential velocity model.
        
        Simplified Model: u is tangential flow.
        """
        u = torch.zeros_like(r)
        
        # If D=2, we can do explicit rotation
        if self.latent_dim == 2:
            for i in range(centers.shape[0]):
                r_i = centers[i]
                omega_i = omegas[i] # Scalar for 2D rotation
                
                diff = r - r_i
                
                # Tagent vector [-y, x]
                tangent = torch.stack([-diff[:, 1], diff[:, 0]], dim=1)
                
                # Add flow
                u += omega_i * tangent
                
        return u

    def step(self, r: torch.Tensor, k_hat: torch.Tensor, 
             centers: torch.Tensor, mus: torch.Tensor, omegas: torch.Tensor,
             dt: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evolve the particle state by one timestep using Formulas 5 & 6.
        
        r: position (thought location)
        k_hat: direction (thought Trajectory)
        """
        # 1. Calculate Field Properties at current location
        # We need r to be a leaf node for autograd to work with create_graph=False
        r_leaf = r.detach().requires_grad_(True)
        
        n = self.compute_refractive_index(r_leaf, centers, mus)
        
        # 2. Gradient of Refractive Index (Force)
        # Formula 5: dk/ds = grad_perp(ln(n))
        # We approximate full gradient for simplicity in high dim
        # grad(ln(n)) = grad(n) / n
        grad_n = torch.autograd.grad(n.sum(), r_leaf, create_graph=False)[0]
        grad_ln_n = grad_n / n.unsqueeze(1)
        
        # Update Direction (The "Force" Bending the thought)
        # Project gradient to be perpendicular to current k for pure bending
        # (Though Formula 7 suggests we might want acceleration too)
        # dk/dt approx c * grad_ln_n
        k_new = k_hat + (self.c * grad_ln_n * dt)
        k_new = torch.nn.functional.normalize(k_new, p=2, dim=1)
        
        # 3. Advection (Movement)
        # Formula 6: dr/dt = (c * k / n) + u_g
        # Use simple Euler integration
        u_g = self.compute_flow_field(r, centers, omegas)
        velocity = (self.c * k_new / n.unsqueeze(1)) + u_g
        
        r_new = r + velocity * dt
        
        return r_new, k_new

class ParallelPMField(PMField):
    """
    High-performance batched implementation for Agentic Physics.
    """
    def __init__(self, d_latent=2, n_centers=10, steps=20, dt=0.1, enable_flow=True):
        super().__init__(d_latent)
        self.steps = steps
        self.dt = dt
        self.enable_flow = enable_flow
        
        # Parameter storage (The "Mind State")
        self.centers = torch.zeros(n_centers, d_latent)
        self.mus = torch.zeros(n_centers)       # Gravity
        self.omegas = torch.zeros(n_centers)    # Spin

    def forward(self, x_start: torch.Tensor, return_trajectory=False):
        """
        Simulate the thought process.
        x_start: [Batch, Dim] initial query embedding
        """
        batch_size = x_start.shape[0]
        
        # Initial thought particle
        r = x_start.clone()
        # Random initial direction or directed? Let's assume initially static 
        # but acquiring momentum. For now, random unit vector.
        k = torch.randn_like(r)
        k = torch.nn.functional.normalize(k, dim=1)
        
        traj = [r.clone()]
        
        for _ in range(self.steps):
            # If flow disabled, zero out omegas for calculation
            active_omegas = self.omegas if self.enable_flow else torch.zeros_like(self.omegas)
            
            r, k = self.step(r, k, self.centers, self.mus, active_omegas, self.dt)
            if return_trajectory:
                traj.append(r.clone())
                
        if return_trajectory:
            return torch.stack(traj, dim=0).transpose(0, 1) # [Batch, Time, Dim]
        
        return r

