from datetime import datetime
import json
from pathlib import Path
from pyexpat import features
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fastai.vision.learner import load_learner
import torch

from marugoto.data import SKLearnEncoder

from ._mil import train, deploy
from .data import get_cohort_df, get_target_enc

__all__ = [
    'train_categorical_model_', 'deploy_categorical_model_', 'categorical_crossval_']


PathLike = Union[str, Path]


def train_categorical_model_(
    clini_table: PathLike,
    slide_csv: PathLike,
    feature_dir: PathLike,
    output_path: PathLike,
    *,
    target_label: str,
    cat_labels: Sequence[str] = [],
    cont_labels: Sequence[str] = [],
    categories: Optional[Iterable[str]] = None,
) -> None:
    """Train a categorical model on a cohort's tile's features.

    Args:
        clini_table:  Path to the clini table.
        slide_csv:  Path to the slide tabel.
        target_label:  Label to train for.
        categories:  Categories to train for, or all categories appearing in the
            clini table if none given (e.g. '["MSIH", "nonMSIH"]').
        feature_dir:  Path containing the features.
        output_path:  File to save model in.
    """
    feature_dir = Path(feature_dir)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # just a big fat object to dump all kinds of info into for later reference
    # not used during actual training
    from datetime import datetime
    info = {
        'description': 'MIL training',
        'clini': str(Path(clini_table).absolute()),
        'slide': str(Path(slide_csv).absolute()),
        'feature_dir': str(feature_dir.absolute()),
        'target_label': str(target_label),
        'cat_labels': [str(c) for c in cat_labels],
        'cont_labels': [str(c) for c in cont_labels],
        'output_path': str(output_path.absolute()),
        'datetime': datetime.now().astimezone().isoformat()}

    model_path = output_path/'export.pkl'
    if model_path.exists():
        print(f'{model_path} already exists. Skipping...')
        return

    clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(clini_table, dtype=str)
    slide_df = pd.read_csv(slide_csv, dtype=str)
    df = clini_df.merge(slide_df, on='PATIENT')

    # filter na, infer categories if not given
    df = df.dropna(subset=target_label)

    # TODO move into get_cohort_df
    if not categories:
        categories = df[target_label].unique()
    categories = np.array(categories)
    info['categories'] = list(categories)

    df = get_cohort_df(clini_table, slide_csv, feature_dir, target_label, categories)

    print('Overall distribution')
    print(df[target_label].value_counts())
    info['class distribution'] = {'overall': {  # type: ignore
        k: int(v) for k, v in df[target_label].value_counts().items()}}

    # Split off validation set
    train_patients, valid_patients = train_test_split(df.PATIENT, stratify=df[target_label])
    train_df = df[df.PATIENT.isin(train_patients)]
    valid_df = df[df.PATIENT.isin(valid_patients)]
    train_df.drop(columns='slide_path').to_csv(output_path/'train.csv', index=False)
    valid_df.drop(columns='slide_path').to_csv(output_path/'valid.csv', index=False)

    info['class distribution']['training'] = {      # type: ignore
        k: int(v) for k, v in train_df[target_label].value_counts().items()}
    info['class distribution']['validation'] = {    # type: ignore
        k: int(v) for k, v in valid_df[target_label].value_counts().items()}

    with open(output_path/'info.json', 'w') as f:
        json.dump(info, f)

    target_enc = OneHotEncoder(sparse=False).fit(categories.reshape(-1, 1))

    add_features = []
    if cat_labels: add_features.append((_make_cat_enc(train_df, cat_labels), df[cat_labels].values))
    if cont_labels: add_features.append((_make_cont_enc(train_df, cont_labels), df[cont_labels].values))

    learn = train(
        bags=df.slide_path.values,
        targets=(target_enc, df[target_label].values),
        add_features=add_features,
        valid_idxs=df.PATIENT.isin(valid_patients).values,
        path=output_path,
    )

    # save some additional information to the learner to make deployment easier
    learn.target_label = target_label
    learn.cat_labels, learn.cont_labels = cat_labels, cont_labels

    learn.export()


def _make_cat_enc(df, cats) -> SKLearnEncoder:
    # create a scaled one-hot encoder for the categorical values
    #
    # due to weirdeties in sklearn's OneHotEncoder.fit we fill NAs with other values
    # randomly sampled with the same probability as their distribution in the
    # dataset.  This is necessary for correctly determining StandardScaler's weigth
    fitting_cats = []
    for cat in cats:
        weights = df[cat].value_counts(normalize=True)
        non_na_samples = df[cat].fillna(pd.Series(np.random.choice(weights.index, len(df), p=weights)))
        fitting_cats.append(non_na_samples)
    cat_samples = np.stack(fitting_cats, axis=1)
    cat_enc = make_pipeline(
        OneHotEncoder(sparse=False, handle_unknown='ignore'),
        StandardScaler(),
    ).fit(cat_samples)
    return cat_enc


def _make_cont_enc(df, conts) -> SKLearnEncoder:
    cont_enc = make_pipeline(
        StandardScaler(),
        SimpleImputer(fill_value=0)
    ).fit(df[conts].values)
    return cont_enc


def deploy_categorical_model_(
    clini_table: PathLike,
    slide_csv: PathLike,
    feature_dir: PathLike,
    model_path: PathLike,
    output_path: PathLike,
    *,
    target_label: Optional[str] = None,
    cat_labels: Optional[str] = None,
    cont_labels: Optional[str] = None,
) -> None:
    """Deploy a categorical model on a cohort's tile's features.

    Args:
        clini_excel:  Path to the clini table.
        slide_csv:  Path to the slide tabel.
        target_label:  Label to train for.
        feature_dir:  Path containing the features.
        model_path:  Path of the model to deploy.
        output_path:  File to save model in.
    """
 
    feature_dir = Path(feature_dir)
    model_path = Path(model_path)
    output_path = Path(output_path)
    if (preds_csv := output_path/'patient-preds.csv').exists():
        print(f'{preds_csv} already exists!  Skipping...')
        return

    learn = load_learner(model_path)
    target_enc = get_target_enc(learn)
    categories = target_enc.categories_[0]

    target_label = target_label or learn.target_label
    cat_labels = cat_labels or learn.cat_labels
    cont_labels = cont_labels or learn.cont_labels

    test_df = get_cohort_df(clini_table, slide_csv, feature_dir, target_label, categories)
    patient_preds_df = deploy(test_df=test_df, learn=learn, target_label=target_label)
    output_path.mkdir(parents=True, exist_ok=True)
    patient_preds_df.to_csv(preds_csv, index=False)


def categorical_crossval_(
    clini_excel: PathLike, slide_csv: PathLike, feature_dir: PathLike, output_path: PathLike,
    *,
    target_label: str,
    cat_labels: Sequence[str] = [],
    cont_labels: Sequence[str] = [],
    n_splits: int = 5,
    fixed_folds: Optional[PathLike] = None, #added option to use fixed folds from previous experiment
    categories: Optional[Iterable[str]] = None,
) -> None:
    """Performs a cross-validation for a categorical target.

    Args:
        clini_excel:  Path to the clini table.
        slide_csv:  Path to the slide tabel.
        feature_dir:  Path containing the features.
        target_label:  Label to train for.
        output_path:  File to save model and the results in.
        n_splits:  The number of folds.
        fixed_folds: Path to the folds.pt splits you want to use
        categories:  Categories to train for, or all categories appearing in the
            clini table if none given (e.g. '["MSIH", "nonMSIH"]').
    """
    feature_dir = Path(feature_dir)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # just a big fat object to dump all kinds of info into for later reference
    # not used during actual training
    info = {
        'description': 'MIL cross-validation',
        'clini': str(Path(clini_excel).absolute()),
        'slide': str(Path(slide_csv).absolute()),
        'feature_dir': str(feature_dir.absolute()),
        'target_label': str(target_label),
        'cat_labels': [str(c) for c in cat_labels],
        'cont_labels': [str(c) for c in cont_labels],
        'output_path': str(output_path.absolute()),
        'n_splits': n_splits,
        'datetime': datetime.now().astimezone().isoformat()}

    clini_df = pd.read_csv(clini_excel, dtype=str) if Path(clini_excel).suffix == '.csv' else pd.read_excel(clini_excel, dtype=str)
    slide_df = pd.read_csv(slide_csv, dtype=str)
    df = clini_df.merge(slide_df, on='PATIENT')

    # filter na, infer categories if not given
    df = df.dropna(subset=target_label)

    if not categories:
        categories = df[target_label].unique()
    categories = np.array(categories)
    info['categories'] = list(categories)

    df = get_cohort_df(clini_excel, slide_csv, feature_dir, target_label, categories)

    info['class distribution'] = {'overall': {
        k: int(v) for k, v in df[target_label].value_counts().items()}}

    target_enc = OneHotEncoder(sparse=False).fit(categories.reshape(-1, 1))

    if (fold_path := output_path/'folds.pt').exists():
        folds = torch.load(fold_path)
    
    elif (fixed_folds is not None):
        folds = torch.load(fixed_folds)
        torch.save(folds, output_path/'folds.pt')
        print(f"Successfully loaded and saved fixed folds from {fixed_folds}")

    else:
        #added shuffling with seed 1337
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1337)
        patient_df = df.groupby('PATIENT').first().reset_index()
        folds = tuple(skf.split(patient_df.PATIENT, patient_df[target_label]))
        torch.save(folds, fold_path)

    info['folds'] = [
        {
            part: list(df.PATIENT[folds[fold][i]])
            for i, part in enumerate(['train', 'test'])
        }
        for fold in range(n_splits) ]

    with open(output_path/'info.json', 'w') as f:
        json.dump(info, f)

    for fold, (train_idxs, test_idxs) in enumerate(folds):
        fold_path = output_path/f'fold-{fold}'
        if (preds_csv := fold_path/'patient-preds.csv').exists():
            print(f'{preds_csv} already exists!  Skipping...')
            continue
        elif (fold_path/'export.pkl').exists():
            learn = load_learner(fold_path/'export.pkl')
        else:
            fold_train_df = df.iloc[train_idxs]
            learn = _crossval_train(
                fold_path=fold_path, fold_df=fold_train_df, fold=fold, info=info,
                target_label=target_label, target_enc=target_enc,
                cat_labels=cat_labels, cont_labels=cont_labels)
            learn.export()

        fold_test_df = df.iloc[test_idxs]
        fold_test_df.drop(columns='slide_path').to_csv(fold_path/'test.csv', index=False)
        patient_preds_df = deploy(
            test_df=fold_test_df, learn=learn,
            target_label=target_label, cat_labels=cat_labels, cont_labels=cont_labels)
        patient_preds_df.to_csv(preds_csv, index=False)


def _crossval_train(
    *, fold_path, fold_df, fold, info, target_label, target_enc, cat_labels, cont_labels
):
    """Helper function for training the folds."""
    assert fold_df.PATIENT.nunique() == len(fold_df)
    fold_path.mkdir(exist_ok=True, parents=True)

    info['class distribution'][f'fold {fold}'] = {'overall': {
        k: int(v) for k, v in fold_df[target_label].value_counts().items()}}

    train_patients, valid_patients = train_test_split(
        fold_df.PATIENT, stratify=fold_df[target_label], random_state=1337)
    train_df = fold_df[fold_df.PATIENT.isin(train_patients)]
    valid_df = fold_df[fold_df.PATIENT.isin(valid_patients)]
    train_df.drop(columns='slide_path').to_csv(fold_path/'train.csv', index=False)
    valid_df.drop(columns='slide_path').to_csv(fold_path/'valid.csv', index=False)

    info['class distribution'][f'fold {fold}']['training'] = {
        k: int(v) for k, v in train_df[target_label].value_counts().items()}
    info['class distribution'][f'fold {fold}']['validation'] = {
        k: int(v) for k, v in valid_df[target_label].value_counts().items()}

    add_features = []
    if cat_labels: add_features.append((_make_cat_enc(train_df, cat_labels), fold_df[cat_labels].values))
    if cont_labels: add_features.append((_make_cont_enc(train_df, cont_labels), fold_df[cont_labels].values))

    learn = train(
        bags=fold_df.slide_path.values,
        targets=(target_enc, fold_df[target_label].values),
        add_features=add_features,
        valid_idxs=fold_df.PATIENT.isin(valid_patients),
        path=fold_path)
    learn.target_label = target_label
    learn.cat_labels, learn.cont_labels = cat_labels, cont_labels

    return learn
