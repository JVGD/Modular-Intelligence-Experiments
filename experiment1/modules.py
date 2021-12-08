import torch as T

"""
Modules without learnable parameters, this
modules will act as intrinsic knowledge that
the AI must decide when to use and when not
"""


class Adder(object):
    """Adder module without learnable parameters"""
    def __init__(self) -> None:
        """Init adder module"""
        super().__init__()

    def __call__(self, X: T.Tensor) -> T.Tensor:
        """Performing the adding operation"""
        return X[:, 0] + X[:, 1]


class Substracter(object):
    """Substracter module without learnable parameters"""
    def __init__(self) -> None:
        """Init substracter module"""
        super().__init__()

    def __call__(self, X: T.Tensor) -> T.Tensor:
        """Performing the substracting operation"""
        return X[:, 0] - X[:, 1]


if __name__ == "__main__":
    adder = Adder()
    x = T.rand(2)
    y = adder(x)
    print(x,y)