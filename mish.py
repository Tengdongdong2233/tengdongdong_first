import chainer.functions as F
import chainer


def Mish(input):
    return input * F.tanh(F.softplus(input))
