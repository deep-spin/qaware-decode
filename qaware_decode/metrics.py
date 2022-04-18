import numpy as np
from typing import List

from joblib import delayed
from tqdm import tqdm
from functools import partial

from .utils import ProgressParallel


def build_metric_fn(
    metric_name: str,
    comet_dir: str = None,
    bleurt_dir: str = None,
    n_cpus=1,
    n_gpus=1,
    batch_size: int = 64,
    progress_bar: bool = True,
    only_sentence_level: bool = True,
):
    if metric_name == "comet":
        assert comet_dir is not None
        return partial(
            comet,
            comet_dir=comet_dir,
            comet_bsize=batch_size,
            progress_bar=progress_bar,
        )
    elif metric_name == "bleurt":
        assert bleurt_dir is not None
        return partial(bleurt, bleurt_dir=bleurt_dir, bleurt_bsize=batch_size)
    elif metric_name == "bleu":
        return partial(
            bleu,
            progress_bar=progress_bar,
            parallel=n_cpus,
            only_sentence_level=only_sentence_level,
        )


def comet(
    hyps: List[str],
    refs: List[str],
    srcs: List[str],
    comet_dir: str = None,
    comet_model: str = "wmt20-comet-da",
    comet_bsize: int = 64,
    progress_bar: bool = True,
):
    from comet import download_model, load_from_checkpoint

    # download comet and load
    comet_path = download_model(comet_model, comet_dir)
    comet_model = load_from_checkpoint(comet_path)
    comet_input = [
        {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
    ]
    # sentence-level and corpus-level COMET
    return comet_model.predict(
        comet_input,
        batch_size=comet_bsize,
        sort_by_mtlen=True,
        progress_bar=progress_bar,
    )


def bleurt(
    hyps: List[str],
    refs: List[str],
    srcs: List[str] = None,
    bleurt_dir: str = None,
    bleurt_bsize: str = 64,
):
    from bleurt import score

    bleurt_scorer = score.LengthBatchingBleurtScorer(bleurt_dir)
    bleurt_scores = bleurt_scorer.score(
        references=refs,
        candidates=hyps,
        batch_size=bleurt_bsize,
    )
    assert type(bleurt_scores) == list
    return bleurt_scores, np.array(bleurt_scores).mean()


def bleu(
    hyps: List[str],
    refs: List[str],
    srcs: List[str] = None,
    progress_bar: bool = True,
    parallel: int = 1,
    only_sentence_level: bool = True,
):
    import sacrebleu

    bleu_fn = lambda *args: sacrebleu.sentence_bleu(*args).score
    iterator = (
        delayed(bleu_fn)(hyp, [ref]) if parallel > 1 else bleu_fn(hyp, [ref])
        for hyp, ref in zip(hyps, refs)
    )

    if parallel > 1 and progress_bar:
        iterator = ProgressParallel(
            total=len(hyps),
            n_jobs=parallel,
            batch_size=50000,
            pre_dispatch="16*n_jobs",
        )(iterator)
    elif progress_bar:
        iterator = tqdm(iterator, total=len(hyps))

    sentence_scores = list(iterator)

    corpus_score = None
    if not only_sentence_level:
        corpus_score = sacrebleu.corpus_bleu(hyps, [refs]).score

    return sentence_scores, corpus_score
