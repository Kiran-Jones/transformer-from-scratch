import numpy as np

class Adam:
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        self._update_recursive(self.model)

    def zero_grad(self):
        self._zero_grad_recursive(self.model)

    def _update_param(self, param, grad):
        if param is None or grad is None:
            return

        param_id = id(param)
        
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param)
            self.v[param_id] = np.zeros_like(param)
            
        m = self.m[param_id]
        v = self.v[param_id]
        
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        
        self.m[param_id] = m
        self.v[param_id] = v

    def _update_recursive(self, module):
        
        if hasattr(module, 'weight'):
            grad = getattr(module, 'grad_weights', None)
    
            if grad is not None:
                self._update_param(module.weight, grad)

        if hasattr(module, 'bias') and hasattr(module, 'grad_bias'):
            self._update_param(module.bias, module.grad_bias)

        if hasattr(module, 'gamma') and hasattr(module, 'grad_gamma'):
            self._update_param(module.gamma, module.grad_gamma)
            
        if hasattr(module, 'beta') and hasattr(module, 'grad_beta'):
            self._update_param(module.beta, module.grad_beta)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            if isinstance(attr, list):
                for item in attr:
                    if hasattr(item, '__class__'):
                        self._update_recursive(item)
            
            elif hasattr(attr, '__class__') and not isinstance(attr, (np.ndarray, int, float, str)):
                if hasattr(attr, 'forward'):
                    self._update_recursive(attr)

    def _zero_grad_recursive(self, module):
        if hasattr(module, 'grad_weights'): module.grad_weights = None
        if hasattr(module, 'grad_bias'): module.grad_bias = None
        if hasattr(module, 'grad_gamma'): module.grad_gamma = None
        if hasattr(module, 'grad_beta'): module.grad_beta = None
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, list):
                for item in attr:
                    if hasattr(item, 'forward'):
                        self._zero_grad_recursive(item)
            elif hasattr(attr, 'forward'):
                self._zero_grad_recursive(attr)