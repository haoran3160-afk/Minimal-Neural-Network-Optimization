"""Nine-parameter 2-2-1 tanh network with an explicit, hand-written gradient."""
import numpy as np

# Centered XOR, four truth-table entries. No train/test generalization claim.
X = np.array([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])
Y = np.array([0.,1.,1.,0.])
PARAMETER_NAMES = ['W1_00','W1_01','W1_10','W1_11','b1_0','b1_1','w2_0','w2_1','b2']


def unpack(theta):
    theta=np.asarray(theta,float)
    if theta.shape!=(9,) or not np.isfinite(theta).all():
        raise ValueError('theta must contain nine finite parameters')
    return theta[:4].reshape(2,2),theta[4:6],theta[6:8],theta[8]


def initialize(seed=0):
    """Small independent biases break hidden-unit and centered-XOR symmetries."""
    rng=np.random.default_rng(seed)
    return np.r_[rng.normal(0,1/np.sqrt(2),4),rng.normal(0,.1,2),
                 rng.normal(0,1/np.sqrt(2),2),0.]


def sigmoid(z):
    return np.exp(-np.logaddexp(0.,-np.asarray(z,float)))


def forward(theta,x=X):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all():
        raise ValueError('x must be a finite N by 2 matrix')
    w1,b1,w2,b2=unpack(theta)
    hidden=np.tanh(x@w1+b1)
    logits=hidden@w2+b2
    return logits,hidden


def predict_proba(theta,x=X):
    return sigmoid(forward(theta,x)[0])


def loss_and_gradient(theta,x=X,y=Y):
    """Mean BCE from logits; analytic backprop, without an autodiff library."""
    x,y=np.asarray(x,float),np.asarray(y,float)
    if y.shape!=(len(x),) or not len(y) or not np.isfinite(y).all() or ((y<0)|(y>1)).any():
        raise ValueError('y must have one target in [0,1] per example')
    logits,hidden=forward(theta,x)
    _,_,w2,_=unpack(theta)
    loss=float(np.mean(np.logaddexp(0.,logits)-y*logits))
    delta=(sigmoid(logits)-y)/len(y)
    grad_w2=hidden.T@delta
    grad_b2=float(delta.sum())
    hidden_delta=delta[:,None]*w2[None,:]*(1-hidden**2)
    grad_w1=x.T@hidden_delta
    grad_b1=hidden_delta.sum(axis=0)
    gradient=np.r_[grad_w1.ravel(),grad_b1,grad_w2,grad_b2]
    return loss,gradient


def accuracy(theta,x=X,y=Y):
    return float(np.mean((predict_proba(theta,x)>=.5)==np.asarray(y)))
