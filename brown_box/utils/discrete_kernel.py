
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel

class DiscreteKernel(StationaryKernelMixin, NormalizedKernelMixin,
                     Kernel):
    def __init__(self, kernel, transformer):
        self._kernel = kernel
        self._tr = transformer
    
    def get_params(self, deep=True):
        """Get parameters of this kernel.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return dict(kernel=self._kernel, transformer=self._tr)

    def __call__(self, X, Y=None, eval_gradient=False):
        _X = self._tr.continuous_transform(X)
        _Y = None
        if Y is not None:
            _Y = self._tr.continuous_transform(Y)
        return self._kernel(_X, _Y, eval_gradient)

    def diag(self, X):
        _X = self._tr.continuous_transform(X)
        return self._kernel.diag(_X)