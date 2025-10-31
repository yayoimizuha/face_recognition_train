"""
Muon Optimizer Implementation

Muon (MomentUm Orthogonalized by Newton-schulz) is a novel optimizer that combines
momentum-based updates with Newton-Schulz orthogonalization. It's particularly
effective for 2D parameters in neural networks.

Based on the paper and implementations:
- https://github.com/KellerJordan/cifar10-airbench
- https://github.com/kyegomez/zeta

Key Features:
- Newton-Schulz iterations for matrix orthogonalization
- Momentum-based updates (with optional Nesterov)
- Particularly effective for convolutional and transformer architectures
- Better convergence properties than standard SGD/Adam for certain tasks
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Optional, List


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    
    This uses a quintic iteration whose coefficients are selected to maximize the slope at zero.
    The iteration produces approximately UV^T where USV^T = G is the SVD, but with S' being
    diagonal entries uniformly distributed, which doesn't hurt performance.
    
    Args:
        G: Input 2D tensor to orthogonalize
        steps: Number of Newton-Schulz iteration steps (default: 5)
        eps: Small constant for numerical stability (default: 1e-7)
        
    Returns:
        Orthogonalized tensor with same shape as G
    """
    assert len(G.shape) == 2, f"Expected 2D tensor, got shape {G.shape}"
    
    # Coefficients optimized for convergence
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Convert to bfloat16 for efficiency (if supported)
    X = G.bfloat16() if G.device.type == 'cuda' else G.float()
    
    # Normalize to ensure top singular value <= 1
    X = X / (X.norm() + eps)
    
    # Handle non-square matrices by transposing if needed
    transposed = False
    if G.size(0) > G.size(1):
        X = X.T
        transposed = True
    
    # Newton-Schulz iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    # Restore original orientation if transposed
    if transposed:
        X = X.T
    
    # Return in original dtype
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Implements Muon optimizer (MomentUm Orthogonalized by Newton-schulz).
    
    Muon is particularly effective for training deep neural networks with 2D parameters.
    It combines momentum-based gradient descent with Newton-Schulz orthogonalization,
    providing better convergence and training stability.
    
    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups
        lr (float): learning rate (default: 0.02)
        momentum (float): momentum factor (default: 0.95)
        nesterov (bool): enables Nesterov momentum (default: True)
        ns_steps (int): number of Newton-Schulz iteration steps (default: 5)
        eps (float): term added to norm for numerical stability (default: 1e-7)
        
    Example:
        >>> model = MyModel()
        >>> optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
        
    Note:
        Muon is designed primarily for 2D parameters (e.g., weight matrices in Linear and Conv layers).
        For non-2D parameters, it falls back to standard momentum SGD.
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov momentum requires a momentum > 0")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps value: {ns_steps}")
            
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
                
        Returns:
            Optional[float]: The loss value returned by the closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize momentum buffer
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                
                # Apply Nesterov momentum if enabled
                if nesterov:
                    grad_update = grad.add(buf, alpha=momentum)
                else:
                    grad_update = buf
                
                # For 2D parameters, apply Newton-Schulz orthogonalization
                if len(p.shape) == 2:
                    # Reshape to 2D if needed (shouldn't be needed, but defensive)
                    update = zeropower_via_newtonschulz5(
                        grad_update.view(p.shape[0], -1),
                        steps=ns_steps,
                        eps=eps
                    ).view(p.shape)
                else:
                    # For non-2D parameters (e.g., biases, BatchNorm), use standard momentum
                    update = grad_update
                
                # Apply update with learning rate
                p.add_(update, alpha=-lr)
        
        return loss
    
    def zero_grad(self, set_to_none: bool = True):
        """
        Resets the gradients of all optimized parameters.
        
        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
        """
        super().zero_grad(set_to_none=set_to_none)
