import inspect

import numpy as np
from cosmoprimo.interpolator import PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D, get_default_k_callable, get_default_z_callable


var_types = ['POSITIONAL_OR_KEYWORD', 'KEYWORD_ONLY', 'VAR_POSITIONAL', 'VAR_KEYWORD']

VarType = type('VarType', (), {**dict(zip(var_types, var_types)), 'ALL': var_types})


def function_signature(func):
    if not inspect.isfunction(func):
        raise ValueError('input object is not a function')
    name = func.__name__
    if name.startswith('<') and name.endswith('>'):
        raise ValueError('input object has no valuable name, e.g. may be a lambda expression?')
    sig = inspect.signature(func)
    parameters, vartypes, dlocals = [], {}, {}
    for param in sig.parameters.values():
        vartypes[param.name] = {param.POSITIONAL_OR_KEYWORD: VarType.POSITIONAL_OR_KEYWORD,
                                param.KEYWORD_ONLY: VarType.KEYWORD_ONLY,
                                param.VAR_POSITIONAL: VarType.VAR_POSITIONAL,
                                param.VAR_KEYWORD: VarType.VAR_KEYWORD}[param.kind]
        default = param.default
        if default is not inspect._empty:
            try:
                param = param.replace(default='#{}#'.format(param.name))
                dlocals[param.name] = default
            except ValueError:
                pass
        parameters.append(param)
    sig = sig.replace(parameters=parameters)
    sig = str(sig)
    for param in dlocals:
        sig = sig.replace("'#{}#'".format(param), param)
    return name, sig, vartypes, dlocals


def get_section_state(section, **kwargs):

    kwargs.setdefault('z', get_default_z_callable())
    state, meta = {}, {}
    for name in dir(section.__class__):
        attr = getattr(section, name)
        if callable(attr):
            name, signature, vartypes, dlocals = function_signature(attr)
            kwds = {}
            for name, vartype in vartypes:
                if vartype == VarType.POSITIONAL_OR_KEYWORD and name in kwargs:
                    kwds[name] = kwargs[name]
            if (section.__class__.__name__, name) == ():
                pass
            value = attr(**kwds)
            meta[name] = {'type': 'callable', 'name': name, 'signature': signature, 'vartypes': vartypes, 'dlocals': dlocals}
        else:
            value = attr
            meta[name] = {}
        if isinstance(value, (PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D)):
            state[name + '.k'] = k = get_default_k_callable()
            state[name + '.pk'] = value(k)
            meta[name].update({'result': value.__class__.__name__})
        else:
            state[name] = value

    return state, meta


def set_section_state(section_name, state, meta):

    class_state = {}
    for name, meta in meta.items():
        if meta:
            if meta['type'] == 'callable':
                code = "return interpolate.interp1d(self._state['z'], self._state[{}])(z)".format(name)
                code = 'def {}{}:\n{}'.format(meta['name'], meta['signature'], code)
                scope = {}
                exec(code, dict(meta['dlocals']), scope)  # exec fills dlocals, so let's make a copy
                class_state[name] = scope[name]
        else:
            class_state[name] = state[name]