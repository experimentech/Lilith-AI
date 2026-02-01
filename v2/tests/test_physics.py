import sys
import os
import torch
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from v2.lilith_v2.pmflow_physics import PMField

class TestPMFlowPhysics(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 2
        self.field = PMField(latent_dim=self.latent_dim)

    def test_refractive_index_formula(self):
        """Formula 1: n(r) = 1 + sum(mu/r)"""
        r = torch.tensor([[1.0, 0.0]])
        centers = torch.tensor([[0.0, 0.0]])
        mus = torch.tensor([1.0])
        
        # At distance 1.0, n should be 1 + 1/1 = 2
        n = self.field.compute_refractive_index(r, centers, mus)
        self.assertAlmostEqual(n.item(), 2.0, places=4)

    def test_flow_field_rotation(self):
        """Formula 2: u = Omega x r (Rotation)"""
        r = torch.tensor([[1.0, 0.0]])
        centers = torch.tensor([[0.0, 0.0]])
        omegas = torch.tensor([1.0]) # CCW rotation
        
        # Tangent of [1,0] is [0, 1]
        u = self.field.compute_flow_field(r, centers, omegas)
        self.assertAlmostEqual(u[0, 0].item(), 0.0, places=4)
        self.assertAlmostEqual(u[0, 1].item(), 1.0, places=4)

    def test_step_dynamics(self):
        """Formula 5 & 6: Ray bending and advection"""
        # Particle moving Right ([1,0]) at x=-1
        # Attractor at x=0
        r = torch.tensor([[-1.0, 0.1]]) # Slightly off-axis to see bending
        k = torch.tensor([[1.0, 0.0]])
        
        centers = torch.tensor([[0.0, 0.0]])
        mus = torch.tensor([2.0]) # Strong attractor
        omegas = torch.tensor([0.0])
        
        r_new, k_new = self.field.step(r, k, centers, mus, omegas, dt=0.1)
        
        # 1. Gradient should pull Y towards 0 (bending)
        # k_new Y component should become negative (turning down toward center)
        self.assertTrue(k_new[0, 1].item() < 0.0)
        
        # 2. Position should advance roughly by c*dt/n
        # approx dist 1.0 -> n = 1 + 2 = 3. velocity ~ 1/3 = 0.33
        # dx ~ 0.33 * 0.1 = 0.033
        # r_new x should be > -1.0
        self.assertTrue(r_new[0, 0].item() > -1.0)

if __name__ == '__main__':
    unittest.main()
