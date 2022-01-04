import abc

from theory_parameters import VcbData


class AbstractInclusiveVcb(abc.ABC):

    def __init__(self, **kwargs):
        self.data = VcbData(**kwargs)

    @abc.abstractmethod
    def total_rate(self, Vcb, mb, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        pass

    @abc.abstractmethod
    def q2_moment_1(self, q_cut, mb, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        pass

    @abc.abstractmethod
    def q2_moment_2(self, q_cut, mb, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        pass

    @abc.abstractmethod
    def q2_moment_3(self, q_cut, mb, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        pass

    @abc.abstractmethod
    def q2_moment_4(self, q_cut, mb, mc, muG, sB, rE, sqB, sE, rG, rhoD, mupi):
        pass
