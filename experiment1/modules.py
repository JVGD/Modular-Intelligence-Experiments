import torch as T


class Adder(object):
    def __init__(self) -> None:
        """Init adder module"""
        super().__init__()

    def __call__(self, X: T.Tensor) -> T.Tensor:
        """Calling the adding operation without gradient"""
        return X[0] + X[1]


class Substracter(object):
    def __init__(self) -> None:
        """Init substracter module"""
        super().__init__()

    def __call__(self, X: T.Tensor) -> T.Tensor:
        """Calling the adding operation without gradient"""
        return X[0] - X[1]


if __name__ == "__main__":
    adder = Adder()
    x = T.rand(2)
    y = adder(x)
    print(x,y)