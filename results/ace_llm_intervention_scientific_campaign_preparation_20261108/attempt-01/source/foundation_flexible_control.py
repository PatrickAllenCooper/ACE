"""Fixed non-pretrained scalar RBF control for the proposed retention screen.

Only fit-row parent standardization and target centering. No parameter search,
calibration access, generative truth, pretrained weights or file IO.
"""


class RBFControl:
    def fit(self, x, y):
        if hasattr(self, 'model_'):
            raise ValueError('refitting a fitted control is prohibited')
        import numpy as np
        from sklearn.kernel_ridge import KernelRidge
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
            raise ValueError('two or more paired scalar fit rows required')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('finite fit rows required')
        self.center_ = float(np.mean(x))
        self.scale_ = float(np.std(x, ddof=0))
        self.target_center_ = float(np.mean(y))
        if not np.isfinite([self.center_, self.scale_, self.target_center_]).all():
            raise ValueError('nonfinite fit normalization')
        if self.scale_ == 0.:
            self.scale_ = 1.
        z = ((x - self.center_) / self.scale_).reshape(-1, 1)
        self.model_ = KernelRidge(kernel='rbf', gamma=1., alpha=.01).fit(
            z, y - self.target_center_)
        self.fit_rows_ = len(x)
        return self

    def __call__(self, x):
        import numpy as np
        if not hasattr(self, 'model_'):
            raise ValueError('control is not fitted')
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
            raise ValueError('nonempty finite scalar parents required')
        y = self.target_center_ + self.model_.predict(
            ((x - self.center_) / self.scale_).reshape(-1, 1))
        if not np.isfinite(y).all():
            raise ValueError('nonfinite control prediction')
        return tuple(float(v) for v in y)
