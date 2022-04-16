import numpy as np
from functools import partial
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hyps",
        type=str,
        help="File containing all hypothesis grouped per sentence, with ``num_samples*sentences`` ",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of hypothesis per sentence",
    )
    parser.add_argument(
        "--metric",
        default="comet",
        choices=["bleu", "comet", "bleurt"],
        help="Metric to use. Currently only bleu, comet and bleurt are supported. Check `qaware_decode/metrics.py` for more details.",
    )
    parser.add_argument(
        "--num-subsamples",
        type=int,
        default=None,
        help="Number of subsamples to use for MBR expectation",
    )
    parser.add_argument(
        "--comet-dir",
        default=None,
        help="Directory containing the comet models. Only necessary if metric is comet",
    )
    parser.add_argument(
        "--bleurt-dir",
        default=None,
        help="Directory containing the bleurt models. Only necessary if metric is bleurt",
    )
    parser.add_argument(
        "--n-cpus",
        default=1,
        type=int,
        help="number of cpus to use for cpu based metrics",
    )
    parser.add_argument(
        "-n-gpus",
        default=1,
        type=int,
        help="number of gpus to use for gpu based metrics",
    )
    parser.add_argument(
        "--batch-size", default=64, type=int, help="batch size for gpu-based metrics"
    )
    parser.add_argument(
        "--src",
        default=None,
        help="File containing source sentences. Only necessary if metric is comet",
    )
    parser.add_argument(
        "--save-mbr-utils",
        default=None,
        help="File to save utility scores, one per hypothesis",
    )
    parser.add_argument("--seed")
    return parser.parse_args()


def mbr_corpus(
    hyps: list[list[str]],
    metric: callable,
    srcs: list[str] = None,
    num_subsamples: int = None,
    aggregation: str = "mean",
    scores: list[list[float]] = None,
) -> list[list[float]]:
    """
    Computes per-sample MBR for a corpus. Returns the (negative) risk of each sample

    Args:
        hyps: list of hypotheses for each sample
        metric: metric to compute MBR
        srcs: source for each sample. only used for src-based metrics (comet)
        num_subsamples: number of subsamples to use for MBR
        aggregation: how to aggregate the subsamples. "mean" or "max"

    Returns:
        neg_risk: negative risk of each sample
    """

    if srcs is not None:
        assert len(hyps) == len(srcs), f"{len(hyps)} != {len(srcs)}"

    assert (
        metric is not None or metric_matrixes is not None
    ), "metric or matrixes must be specified"

    num_samples = len(hyps[0])
    use_subsampling = num_subsamples is not None and num_subsamples < num_samples

    # flattens the source
    cands = []
    refs = []
    dup_srcs = [] if srcs is not None else None
    for i, samples in enumerate(hyps):
        indices = (
            np.random.choice(num_samples, num_subsamples, replace=False)
            if use_subsampling
            else list(range(num_samples))
        )
        for cand in samples:
            for ref_id in indices:
                cands.append(cand)
                refs.append(samples[ref_id])
                if srcs is not None:
                    dup_srcs.append(srcs[i])

    flat_metric_matrixes, _ = metric(cands, refs, srcs=dup_srcs)

    # unflattens the metrics into a N*S*T tensor
    metric_matrixes = []
    for i, _ in enumerate(hyps):
        metric_matrixes.append([])
        for j in range(num_samples):
            metric_matrixes[i].append([])
            for k in range(num_subsamples if use_subsampling else num_samples):
                metric_matrixes[i][j].append(
                    flat_metric_matrixes[
                        i * num_samples * num_samples + j * num_samples + k
                    ]
                )

    metric_matrixes = np.array(metric_matrixes)

    if aggregation == "mean":
        neg_risks = metric_matrixes.mean(axis=2).tolist()
    elif aggregation == "weighted_mean":
        assert scores is not None
        # TODO implemented
        raise ValueError("weighted_mean not implemented")
    else:
        raise ValueError(f"aggregation {aggregation} not implemented")

    return neg_risks


def main():
    args = parse_args()

    with open(args.hyps, encoding="utf-8") as hyp_f:
        flat_hyps = [line.strip() for line in hyp_f.readlines()]
        assert len(flat_hyps) % args.num_samples == 0

        # unflatten the hypotheses
        hyps = []
        for i in range(0, len(flat_hyps) // args.num_samples):
            hyps.append([])
            for j in range(args.num_samples):
                hyps[i].append(flat_hyps[i * args.num_samples + j])

    srcs = None
    if args.src is not None:
        with open(args.src, encoding="utf-8") as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

    if args.metric == "comet":
        from qaware_decode.metrics import comet

        assert (
            args.comet_dir is not None
        ), "comet_dir needs to specified for comet metric"
        metric = partial(comet, comet_dir=args.comet_dir)

    if args.metric == "bleurt":
        from qaware_decode.metrics import bleurt

        assert (
            args.bleurt_dir is not None
        ), "bleurt_dir needs to specified for bleurt metric"

        metric = partial(bleurt, bleurt_dir=args.bleurt_dir)

    if args.metric == "bleu":
        from qaware_decode.metrics import bleu

        metric = partial(bleu, parallel=args.n_cpus)

    neg_risk = mbr_corpus(
        hyps,
        metric=metric,
        srcs=srcs,
        num_subsamples=args.num_subsamples,
    )

    if args.save_mbr_utils is not None:
        mbr_utils = open(args.save_mbr_utils, "w")

    # print best candidates
    for sample_hyps, sample_utilities in zip(hyps, neg_risk):
        print(sample_hyps[np.argmax(sample_utilities)])
        if args.save_mbr_utils is not None:
            for util in sample_utilities:
                print(f"mbr-util={util}", file=mbr_utils)


if __name__ == "__main__":
    main()
