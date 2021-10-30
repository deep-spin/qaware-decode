import numpy as np

COMET_MODEL = "wmt20-comet-da"
COMET_BATCH_SIZE = 64

BLEURT_BATCH_SIZE = 64


def comet(hyps, refs, srcs, comet_dir=None):
    from comet import download_model, load_from_checkpoint

    # download comet and load
    comet_path = download_model(COMET_MODEL, comet_dir)
    comet_model = load_from_checkpoint(comet_path)
    comet_input = [
        {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
    ]
    # sentence-level and corpus-level COMET
    return comet_model.predict(
        comet_input, batch_size=COMET_BATCH_SIZE, sort_by_mtlen=True
    )


def bleurt(hyps, refs, srcs=None, bleurt_dir=None):
    from bleurt import score

    bleurt_scorer = score.LengthBatchingBleurtScorer(bleurt_dir)
    bleurt_scores = bleurt_scorer.score(
        references=refs, candidates=hyps, batch_size=BLEURT_BATCH_SIZE
    )
    assert type(bleurt_scores) == list
    return bleurt_scores, np.array(bleurt_scores).mean()


def bleu(hyps, refs, srcs=None):
    import sacrebleu

    sentence_scores = [
        sacrebleu.sentence_bleu(ref, [hyp]).score for hyp, ref in zip(hyps, refs)
    ]
    corpus_score = sacrebleu.corpus_bleu(refs, [hyps]).score
    return sentence_scores, corpus_score
