"""Full-batch GD, heavy-ball momentum and Adam, implemented from equations."""
from dataclasses import dataclass
import numpy as np
from model import accuracy, initialize, loss_and_gradient

DEFAULT_LEARNING_RATES={'gd':1.,'momentum':.2,'adam':.03}


@dataclass
class TrainingResult:
    theta: np.ndarray
    loss: float
    accuracy: float
    iterations: int
    success: bool
    reason: str
    history: np.ndarray  # iteration, mean BCE, gradient norm


def train(method='adam',seed=0,initial=None,learning_rate=None,
          max_iterations=10000,target_loss=.02):
    if method not in DEFAULT_LEARNING_RATES:
        raise ValueError('unknown optimizer')
    rate=DEFAULT_LEARNING_RATES[method] if learning_rate is None else learning_rate
    if not np.isfinite(rate) or rate<=0 or max_iterations<0:
        raise ValueError('invalid learning rate or iteration cap')
    theta=initialize(seed) if initial is None else np.asarray(initial,float).copy()
    velocity=np.zeros(9);second=np.zeros(9)
    history=[];reason='iteration_cap';success=False
    for iteration in range(max_iterations+1):
        loss,gradient=loss_and_gradient(theta)
        norm=float(np.linalg.norm(gradient))
        history.append([iteration,loss,norm])
        if not (np.isfinite(loss) and np.isfinite(gradient).all()):
            reason='nonfinite_loss_or_gradient';break
        if loss<=target_loss and accuracy(theta)==1.:
            success=True;reason='target_loss_and_accuracy';break
        if norm<1e-12:
            reason='stationary_before_target';break
        if iteration==max_iterations:break
        if method=='gd':
            step=rate*gradient
        elif method=='momentum':
            velocity=.9*velocity+gradient
            step=rate*velocity
        else:
            velocity=.9*velocity+.1*gradient
            second=.999*second+.001*gradient**2
            first_hat=velocity/(1-.9**(iteration+1))
            second_hat=second/(1-.999**(iteration+1))
            step=rate*first_hat/(np.sqrt(second_hat)+1e-8)
        candidate=theta-step
        if not np.isfinite(candidate).all():
            reason='nonfinite_update';break
        theta=candidate
    return TrainingResult(theta,loss,accuracy(theta),iteration,success,reason,np.array(history))
