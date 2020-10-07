
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

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.
        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.
        Returns
        -------
        theta : ndarray of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return self._kernel.theta

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.
        Parameters
        ----------
        theta : array of shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        self._kernel.theta = theta

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.
        Returns
        -------
        bounds : array of shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return self._kernel.bounds