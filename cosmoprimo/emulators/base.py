from . import tools


def get_calculator(cosmo):

    def calculator(**params):
        state = {}
        c = cosmo.clone(**params)
        for section in c.engine._Sections:
            tmp = c.engine._Sections[section](c)
            for name, value in tmp.items():
                state['{}.{}'.format(section, name)] = value
        return state

    return calculator



class Emulator(tools.Emulator):

    def set_calculator(self, cosmo, params):
        super(Emulator, self).set_calculator(get_calculator(cosmo), params)


class BaseSampler(tools.BaseSampler):

    def set_calculator(self, cosmo, params):
        super(BaseSampler, self).set_calculator(get_calculator(cosmo), params)


class GridSampler(BaseSampler, tools.GridSampler):

    pass


class QMCSampler(BaseSampler, tools.QMCSampler):

    pass