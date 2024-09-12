# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
HCP behavioural filter
~~~~~~~~~~~~~~~~~~~~~~
Select from the HCP tabular data only the behavioural variables that attained
a high predictability from connectome features according to Kong et al.
(2021).

We use a subset of 10 task performance measures, one self-report / survey
measure, and 1 other measure, as many of the original 58 variables were poorly
predicted by connectome features.

Task performance:
Vocabulary (picture matching)
Reading (pronunciation)
Relational processing
Working memory (n-back)
Story comprehension
Spatial orientation
Processing speed
Fluid intelligence (PMAT)
Visual episodic memory
Cognitive flexibility

Survey:
Openness (NEO)

Other:
Delay discounting
"""
from typing import Any, Sequence, Tuple
import pandas as pd


INPUT_FRAME = '/Users/rastkociric/Downloads/HCPtabular.csv'
OUTPUT_FRAME = '/tmp/HCPtabularfiltered.tsv'
HCP_MEASURES = {
    'Vocabulary (picture matching)': 'PicVocab_Unadj',
    'Reading (pronunciation)': 'ReadEng_Unadj',
    'Relational processing': 'Relational_Task_Acc',
    'Working memory (n-back)': 'WM_Task_Acc',
    'Story comprehension': 'Language_Task_Story_Avg_Difficulty_Level',
    'Spatial orientation': 'VSPLOT_TC',
    'Processing speed': 'ProcSpeed_Unadj',
    'Fluid intelligence (PMAT)': 'PMAT24_A_CR',
    'Visual episodic memory': 'PicSeq_Unadj',
    'Cognitive flexibility': 'CardSort_Unadj',
    'Openness (NEO)': 'NEOFAC_O',
    'Delay discounting': 'DDisc_AUC_40K',
}


def fetch_subject(
    ref: str | pd.DataFrame,
    subject: str | int,
    categoricals: Sequence[str] = (),
    subject_key = 'Subject',
) -> Tuple[Any, Sequence[int]]:
    import jax.numpy as jnp
    import numpy as np
    subject = int(subject)
    if isinstance(ref, str):
        ref = pd.read_csv(ref, sep='\t').set_index(subject_key)
    ref = ref.loc[subject]
    categoricals = {
        e: [
            k
            for k in ref.to_dict()
            if e in k
        ]
        for e in categoricals
    }
    all_categoricals = sum([e for e in categoricals.values()], [])
    continuous = [e for e in ref.to_dict() if e not in all_categoricals]
    continuous = ref[continuous].values.astype(float)
    categoricals = {
        k: ref[v].values.astype(float)
        for k, v in categoricals.items()
    }
    data = np.concatenate((continuous, *[e for e in categoricals.values()]))
    data = np.where(np.isnan(data), 0., data)
    return (
        jnp.asarray(data),
        (len(continuous), *[len(e) for e in categoricals.values()][:-1]),
    )


def filter_df_vars(
    remove_mean_continuous: bool = True,
    remove_mean_categorical: bool = False,
):
    df = pd.read_csv(INPUT_FRAME)
    # From the LUT in Kong et al. (2021) supplement (Table S1), we select the
    # following columns that correspond to the above variables.
    columns = [
        'Subject',
        'Age',
    ] + list(HCP_MEASURES.values())
    df = df[columns]
    df = pd.get_dummies(df)
    # There are not enough 36+ subjects to warrant a separate category.
    df['Age_30+'] = df['Age_31-35'] | df['Age_36+']
    df = df.drop(columns=['Age_31-35', 'Age_36+'])
    if remove_mean_continuous or remove_mean_categorical:
        dff = df.set_index('Subject').select_dtypes(float)
        dfc = df.set_index('Subject').select_dtypes(bool)
        if remove_mean_continuous:
            dff = dff - dff.mean()
        if remove_mean_categorical:
            dfc = dfc.astype(float) * 2 - 1
        df = pd.concat([dff, dfc], axis=1).reset_index()
    df.to_csv(OUTPUT_FRAME, sep='\t', index=False)


def main():
    filter_df_vars(
        remove_mean_continuous=True,
        remove_mean_categorical=False,
    )
    ref = fetch_subject(
        ref=OUTPUT_FRAME,
        subject=100307,
        categoricals=['Age'],
    )
    breakpoint()


if __name__ == '__main__':
    main()
