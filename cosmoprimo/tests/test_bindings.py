
def test_cobaya():

    from cosmoprimo.fiducial import DESI
    cosmo = DESI()

    for engine in ['class', 'camb', 'isitgr', 'mochiclass']:

        params = {'Omega_m': {'prior': {'min': 0.1, 'max': 1.},
                          'ref': {'dist': 'norm', 'loc': 0.3, 'scale': 0.01},
                          'latex': '\Omega_{m}'}} | {name: float(cosmo[name]) for name in ['omega_b', 'H0', 'A_s', 'n_s', 'tau_reio']}

        extra_args = {'m_ncdm': list(map(float, cosmo['m_ncdm'])), 'N_ur': float(cosmo['N_ur'])}

        if engine == 'isitgr':
            params['Q1'] = 1.5
            params['Q2'] = 0.5
            extra_args.update(parameterization='QD', z_div=1, z_TGR=2, z_tw=0.05,
                              k_c=0.01, Q3=1.5, Q4=0.5, D1=1, D2=1, D3=1, D4=1)
        if engine == 'mochiclass':
            params.update(w0={'prior': {'min': -3, 'max': 1}, 'ref': -0.99, 'proposal': 0.1, 'latex': 'w_0', 'drop': True},
                          wa={'prior': {'min': -3, 'max': 2}, 'ref': 0., 'proposal': 0.1, 'latex': 'w_a', 'drop': True},
                          expansion_smg={'value': 'lambda w0,wa: str(0.5)+","+str(w0)+","+str(wa)', 'derived': False},
                          cK={'value': 1, 'drop': True},
                          cB={'drop': True, 'latex': 'c_B', 'prior': {'min': -10, 'max': 10}, 'ref': 0, 'proposal': 0.1},
                          cM={'drop': True, 'latex': 'c_M', 'prior': {'min': -10, 'max': 10}, 'ref': 0, 'proposal': 0.1},
                          cT={'drop': True, 'latex': 'c_T', 'value': 0},
                          parameters_smg={'value': 'lambda cK, cB, cM, cT,: str(cK) + "," + str(cB) + "," + str(cM) + "," + str(cT) + "," + str(1)', 'derived': False})
            extra_args.update(Omega_Lambda=0, Omega_fld=0.0, Omega_smg=-1, gravity_model='propto_omega', expansion_model='wowa')

        info = {'params': params,
                'likelihood': {'sn.pantheon': None, 'H0.riess2020': None, 'bao.sdss_dr12_consensus_final': None},
                'theory': {'cosmoprimo.bindings.cobaya.cosmoprimo': {'engine': engine, 'stop_at_error': True, 'extra_args': extra_args}}}
        info_sampler = {'evaluate': {}}

        from cobaya.model import get_model
        from cobaya.sampler import get_sampler
        model = get_model(info)
        get_sampler(info_sampler, model=model).run()


def test_cobaya2():
    test = ['camb', 'class'][:1]

    if 'camb' in test:
        params = {'logA': {'prior': {'min': 1.61, 'max': 3.91}, 'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.001}, 'proposal': 0.001, 'latex': '\\ln(10^{10} A_\\mathrm{s})', 'drop': True},
                'As': {'value': 'lambda logA: 1e-10*np.exp(logA)', 'latex': 'A_\\mathrm{s}'},
                'ns': {'prior': {'min': 0.8, 'max': 1.2}, 'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004}, 'proposal': 0.002, 'latex': 'n_\\mathrm{s}'},
                'H0': {'prior': {'min': 20, 'max': 100}, 'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01}, 'latex': 'H_0'},
                'ombh2': {'prior': {'min': 0.005, 'max': 0.1}, 'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001}, 'proposal': 0.0001, 'latex': '\\Omega_\\mathrm{b} h^2'},
                'omch2': {'prior': {'min': 0.001, 'max': 0.99}, 'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001}, 'proposal': 0.0005, 'latex': '\\Omega_\\mathrm{c} h^2'},
                'tau': {'latex': '\\tau_\\mathrm{reio}', 'value': 0.0544},
                #'mnu': {'latex': '\\sum m_\\nu', 'value': 0.2},
                #'nnu': {'latex': 'N_\\mathrm{eff}', 'value': 3.044}
                }
        info = {'params': params,
                #'likelihood': {'bao.sdss_dr12_consensus_final': None},
                'likelihood': {'planck_2018_highl_plik.TTTEEE_lite_native': None},
                #'theory': {'cosmoprimo.bindings.cobaya.cosmoprimo': {'engine': 'camb', 'stop_at_error': True, 'extra_args': {'extra_params': {'halofit_version': 'mead', 'bbn_predictor': 'PArthENoPE_880.2_standard.dat', 'lens_potential_accuracy': 1, 'num_massive_neutrinos': 0}}}}}
                'theory': {'cosmoprimo.bindings.cobaya.cosmoprimo': {'engine': 'camb', 'stop_at_error': True, 'extra_args': {'non_linear': 'mead2016', 'extra_params': {'bbn_predictor': 'PArthENoPE_880.2_standard.dat'}}}}}
        from cobaya.model import get_model
        model = get_model(info)
        params = {'logA': 3.057147, 'ns': 0.9657119, 'H0': 70., 'ombh2': 0.02246306, 'omch2': 0.1184775, 'A_planck': 1.} #, 'nnu': 3.044} #, 'A_planck': 1.}
        logpost = model.logposterior(params)
        print(logpost.loglike)

        info['theory'] = {'camb': {'extra_args': {'mnu': 0., 'halofit_version': 'mead2016', 'bbn_predictor': 'PArthENoPE_880.2_standard.dat', 'num_massive_neutrinos': 0}}}
        #info['theory'] = {'camb': {'extra_args': {'bbn_predictor': 'PArthENoPE_880.2_standard.dat', 'lens_potential_accuracy': 1, 'num_massive_neutrinos': 1}}}
        model = get_model(info)
        logpost = model.logposterior(params)
        print(logpost.loglike)

    if 'class' in test:
        params = {'logA': {'prior': {'min': 1.61, 'max': 3.91}, 'ref': {'dist': 'norm', 'loc': 3.036, 'scale': 0.001}, 'proposal': 0.001, 'latex': '\\ln(10^{10} A_\\mathrm{s})', 'drop': True},
                'A_s': {'value': 'lambda logA: 1e-10*np.exp(logA)', 'latex': 'A_\\mathrm{s}'},
                'n_s': {'prior': {'min': 0.8, 'max': 1.2}, 'ref': {'dist': 'norm', 'loc': 0.9649, 'scale': 0.004}, 'proposal': 0.002, 'latex': 'n_\\mathrm{s}'},
                'H0': {'prior': {'min': 20, 'max': 100}, 'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 0.01}, 'latex': 'H_0'},
                'omega_b': {'prior': {'min': 0.005, 'max': 0.1}, 'ref': {'dist': 'norm', 'loc': 0.02237, 'scale': 0.0001}, 'proposal': 0.0001, 'latex': '\\Omega_\\mathrm{b} h^2'},
                'omega_cdm': {'prior': {'min': 0.001, 'max': 0.99}, 'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001}, 'proposal': 0.0005, 'latex': '\\Omega_\\mathrm{c} h^2'},
                'tau_reio': {'latex': '\\tau_\\mathrm{reio}', 'value': 0.0544},
                #'nnu': {'latex': 'N_\\mathrm{eff}', 'value': 3.044}
                }
        info = {'params': params,
                'likelihood': {'planck_2018_highl_plik.TTTEEE_lite_native': None, 'act_dr6_lenslike.ACTDR6LensLike': {'lens_only': False, 'variant': 'actplanck_baseline', 'lmax': 4000, 'version': 'v1.2'}},
                'theory': {'cosmoprimo.bindings.cobaya.cosmoprimo': {'engine': 'class', 'stop_at_error': True, 'extra_args': {'non_linear': 'hmcode', 'N_ur': 3.044}}}}
        from cobaya.model import get_model
        model = get_model(info)
        params = {'logA': 3.057147, 'n_s': 0.9657119, 'H0': 70., 'omega_b': 0.02246306, 'omega_cdm': 0.1184775, 'A_planck': 1.} #, 'nnu': 3.044} #, 'A_planck': 1.}
        logpost = model.logposterior(params)
        print(logpost.loglike)
        info['theory'] = {'classy': {'stop_at_error': True, 'extra_args': {'N_ncdm': 0, 'N_ur': 3.044}}}
        model = get_model(info)
        logpost = model.logposterior(params)
        print(logpost.loglike)


def test_cosmosis():

    import os
    import cosmoprimo

    os.environ['COSMOPRIMO_DIR'] = os.path.dirname(os.path.dirname(cosmoprimo.__file__))
    print(os.environ['COSMOPRIMO_DIR'])
    from cosmosis.main import run_cosmosis
    for engine in ['class', 'camb', 'isitgr']:
        with open('cosmosis_config.ini', 'r') as file:
            config = file.read()
        config = config.replace('engine = class', 'engine = {}'.format(engine))
        tmp_fn = 'tmp.ini'
        with open(tmp_fn, 'w') as file:
            file.write(config)
        run_cosmosis(tmp_fn)
        os.remove(tmp_fn)


if __name__ == '__main__':

    test_cobaya()
    test_cobaya2()
    #test_cosmosis()