from abc import ABC, abstractmethod
import tensorflow as tf


class DEProblem(ABC):
    """
    Abstract interface for a 2nd-order DE problem.

    Subclasses implement:
       - residual(net): returns the ODE/PDE residual tensor given the network callable
       - constraints(net): returns (predicted, target) tensors for boundary/initial conditions
       - domain(): returns the sampling bounds as a dict consumed by the trainer
       - true_solution (optional): callable for error evaluation
    """

    @abstractmethod
    def residual(self, net: callable, x: tf.Tensor):
        """Evaluate DE residual at collocation point x."""
    
    @abstractmethod
    def constraints(self, net: callable):
        """Return (u_pred, u_target) for all imposed conditions."""
    
    @abstractmethod
    def domain(self):
        """Return {'lb': float, 'ub': float} (ODE) or spatial bounds (PDE)."""
    
    def true_solution(self, x):
        return None