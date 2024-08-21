# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Plot gradient reports
~~~~~~~~~~~~~~~~~~~~~
Visualise and save gradient reports
"""
import pickle
import matplotlib.pyplot as plt


def main():
    for info in ('grad_info_L', 'grad_info_R', 'updates_info_L', 'updates_info_R'):
        with open(f'/tmp/{info}.pkl', 'rb') as f:
            grad_info = pickle.load(f)
        for module in ('contractive', 'expansive', 'resample', 'ingress', 'readout'):
            data = grad_info[f'{module}.relnorm']
            plt.figure(figsize=(12, 8))
            plt.plot(data)
            if info[:4] == 'grad':
                plt.ylim(-0.5, 3)
            plt.legend(range(len(data)))
            plt.savefig(f'/tmp/gradinfo_{info}_{module}_relnorm.png')


if __name__ == '__main__':
    main()

