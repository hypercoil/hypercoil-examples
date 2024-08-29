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
import pandas as pd


INPUT_FRAME = '/Users/rastkociric/Downloads/HCPtabular.csv'


def main():
    df = pd.read_csv(INPUT_FRAME)
    # From the LUT in Kong et al. (2021) supplement (Table S1), we select the
    # following columns that correspond to the above variables.
    columns = [
        'Subject',
        'Age',
        'PicVocab_Unadj',
        'ReadEng_Unadj',
        'Relational_Task_Acc',
        'WM_Task_Acc',
        'Language_Task_Story_Avg_Difficulty_Level',
        'VSPLOT_TC',
        'ProcSpeed_Unadj',
        'PMAT24_A_CR',
        'PicSeq_Unadj',
        'CardSort_Unadj',
        'NEOFAC_O',
        'DDisc_AUC_40K',
    ]
    df = df[columns]
    df = pd.get_dummies(df)
    # There are not enough 36+ subjects to warrant a separate category.
    df['Age_30+'] = df['Age_31-35'] | df['Age_36+']
    df = df.drop(columns=['Age_31-35', 'Age_36+'])
    df.to_csv('/tmp/HCPtabularfiltered.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
