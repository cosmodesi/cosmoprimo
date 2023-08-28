
def test_cobaya():

    from cosmoprimo.fiducial import DESI
    cosmo = DESI()

    for engine in ['class', 'camb', 'isitgr']:

        params = {'Omega_m': {'prior': {'min': 0.1, 'max': 1.},
                          'ref': {'dist': 'norm', 'loc': 0.3, 'scale': 0.01},
                          'latex': '\Omega_{m}'},
            'omega_b': cosmo['omega_b'],
            'H0': cosmo['H0'],
            'A_s': cosmo['A_s'],
            'n_s': cosmo['n_s'],
            'tau_reio': cosmo['tau_reio']}

        extra_args = {'m_ncdm': cosmo['m_ncdm'], 'N_ur': cosmo['N_ur']}

        if engine == 'isitgr':
            params['Q1'] = 1.5
            params['Q2'] = 0.5
            extra_args.update(parameterization='QD', binning='hybrid', z_div=1, z_TGR=2, z_tw=0.05,
                              k_c=0.01, Q3=1.5, Q4=0.5, D1=1, D2=1, D3=1, D4=1)
        info = {'params': params, 'debug': True,
                'likelihood': {'sn.pantheon': None, 'H0.riess2020': None, 'bao.sdss_dr12_consensus_final': None, 'planck_2018_highl_plik.TTTEEE': None},
                'theory': {'cosmoprimo.bindings.cobaya.cosmoprimo': {'engine': engine, 'stop_at_error': True, 'extra_args': extra_args}}}
        info_sampler = {'evaluate': {}}

        from cobaya.model import get_model
        from cobaya.sampler import get_sampler
        model = get_model(info)
        get_sampler(info_sampler, model=model).run()


def test_cosmosis():

    import os
    import cosmoprimo
    os.environ['COSMOPRIMO_DIR'] = os.path.dirname(os.path.dirname(cosmoprimo.__file__))
    print(os.environ['COSMOPRIMO_DIR'])
    from cosmosis.main import run_cosmosis
    run_cosmosis('cosmosis_config.ini')


if __name__ == '__main__':

    test_cobaya()
    test_cosmosis()