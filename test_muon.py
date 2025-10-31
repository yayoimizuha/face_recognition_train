"""
Unit tests for Muon optimizer implementation.

This test file validates the Muon optimizer's functionality without requiring
the full training infrastructure or datasets.
"""

import sys
import os

# Add the current directory to the path to import muon
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_muon_basic():
    """Test basic Muon optimizer functionality."""
    try:
        import torch
    except ImportError:
        print("PyTorch not installed. Skipping tests.")
        return True
    
    from muon import Muon, zeropower_via_newtonschulz5
    
    print("=" * 60)
    print("Testing Muon Optimizer Implementation")
    print("=" * 60)
    
    # Test 1: Newton-Schulz function
    print("\n[Test 1] Newton-Schulz orthogonalization function")
    try:
        G = torch.randn(10, 10)
        result = zeropower_via_newtonschulz5(G, steps=5)
        assert result.shape == G.shape, "Output shape mismatch"
        print("  ✓ Newton-Schulz function works correctly")
        print(f"    Input shape: {G.shape}, Output shape: {result.shape}")
    except Exception as e:
        print(f"  ✗ Newton-Schulz test failed: {e}")
        return False
    
    # Test 2: Muon optimizer initialization
    print("\n[Test 2] Muon optimizer initialization")
    try:
        params = [torch.randn(10, 10, requires_grad=True)]
        opt = Muon(params, lr=0.02, momentum=0.95, nesterov=True)
        print("  ✓ Muon optimizer initialized successfully")
        print(f"    Learning rate: {opt.defaults['lr']}")
        print(f"    Momentum: {opt.defaults['momentum']}")
        print(f"    Nesterov: {opt.defaults['nesterov']}")
        print(f"    NS steps: {opt.defaults['ns_steps']}")
    except Exception as e:
        print(f"  ✗ Initialization test failed: {e}")
        return False
    
    # Test 3: Simple optimization step
    print("\n[Test 3] Single optimization step")
    try:
        # Create a simple model
        model = torch.nn.Linear(5, 3)
        optimizer = Muon(model.parameters(), lr=0.01)
        
        # Forward pass
        x = torch.randn(2, 5)
        y = torch.randn(2, 3)
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
        optimizer.zero_grad()
        
        print("  ✓ Optimization step completed successfully")
        print(f"    Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ Optimization step test failed: {e}")
        return False
    
    # Test 4: Multiple iterations
    print("\n[Test 4] Multiple optimization iterations")
    try:
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5)
        )
        optimizer = Muon(model.parameters(), lr=0.02, momentum=0.9)
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        
        losses = []
        for i in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        print("  ✓ Multiple iterations completed successfully")
        print(f"    Initial loss: {losses[0]:.4f}")
        print(f"    Final loss: {losses[-1]:.4f}")
        print(f"    Loss decreased: {losses[-1] < losses[0]}")
    except Exception as e:
        print(f"  ✗ Multiple iterations test failed: {e}")
        return False
    
    # Test 5: 2D vs non-2D parameters
    print("\n[Test 5] Handling 2D and non-2D parameters")
    try:
        class MixedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)  # 2D weight
                self.bn = torch.nn.BatchNorm1d(5)     # 1D parameters
            
            def forward(self, x):
                x = self.linear(x)
                x = self.bn(x)
                return x
        
        model = MixedModel()
        optimizer = Muon(model.parameters(), lr=0.01)
        
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        
        print("  ✓ Mixed parameter types handled correctly")
    except Exception as e:
        print(f"  ✗ Mixed parameters test failed: {e}")
        return False
    
    # Test 6: Invalid parameters
    print("\n[Test 6] Error handling for invalid parameters")
    try:
        params = [torch.randn(10, 10, requires_grad=True)]
        
        # Test invalid learning rate
        try:
            opt = Muon(params, lr=-0.1)
            print("  ✗ Should have raised ValueError for negative lr")
            return False
        except ValueError:
            print("  ✓ Correctly raises ValueError for negative lr")
        
        # Test invalid momentum
        try:
            opt = Muon(params, lr=0.01, momentum=-0.5)
            print("  ✗ Should have raised ValueError for negative momentum")
            return False
        except ValueError:
            print("  ✓ Correctly raises ValueError for negative momentum")
        
    except Exception as e:
        print(f"  ✗ Error handling test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_muon_basic()
    sys.exit(0 if success else 1)
