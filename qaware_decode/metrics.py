import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import ProgressParallel


def comet(
    hyps: list[str],
    refs: list[str],
    srcs: list[str],
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
    hyps: list[str],
    refs: list[str],
    srcs: list[str] = None,
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
    hyps: list[str],
    refs: list[str],
    srcs: list[str] = None,
    progress_bar: bool = True,
    parallel: int = 1,
):
    import sacrebleu

    bleu_fn = lambda *args: sacrebleu.sentence_bleu(*args).score
    iterator = (
        delayed(bleu_fn)(ref, [hyp]) if parallel > 1 else bleu_fn(ref, [hyp])
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
    return sentence_scores
