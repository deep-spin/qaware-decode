import numpy as np


def comet_qe(
    hyps: list[str],
    srcs: list[str],
    cometqe_dir: str = None,
    cometqe_model: str = "wmt20-comet-qe-da",
    cometqe_bsize: int = 64,
    progress_bar: bool = True,
):
    from comet import download_model, load_from_checkpoint

    # download comet and load
    cometqe_path = download_model(cometqe_model, cometqe_dir)
    cometqe_model = load_from_checkpoint(cometqe_path)
    cometqe_input = [{"src": src, "mt": mt} for src, mt in zip(srcs, hyps)]
    # sentence-level and corpus-level COMET
    return cometqe_model.predict(
        cometqe_input, batch_size=cometqe_bsize, progress_bar=progress_bar
    )


def mbart_qe(
    hyps: list[str],
    srcs: list[str],
    langpair: str,
    mbartqe_dir: str = None,
    mbartqe_model: str = "wmt21-mbart-m2",
    mbartqe_bsize: int = 64,
    progress_bar: bool = True,
):
    from mbart_qe import download_mbart_qe, load_mbart_qe

    mbart_path = download_mbart_qe(mbartqe_model, mbartqe_dir)
    mbart = load_mbart_qe(mbart_path)

    mbart_input = [
        {"src": src, "mt": mt, "lp": langpair} for src, mt in zip(srcs, hyps)
    ]
    _, segment_scores = mbart.predict(
        mbart_input, show_progress=progress_bar, batch_size=mbartqe_bsize
    )
    mbart_score = [s[0] for s in segment_scores]
    mbart_uncertainty = [s[1] for s in segment_scores]
    return {"score": mbart_score, "uncertainty": mbart_uncertainty}, {
        "score": np.mean(mbart_score)
    }
