import numpy as np
import argparse


def mbr_corpus(
    hyps: list[list[str]],
    metric: callable = None,
    metric_matrixes: np.array = None,
    srcs: list[str] = None,
    num_subsamples: int = None,
    aggregation: str = "mean",
    scores: list[list[float]] = None,
) -> list[list[float]]:
    """
    Computes per-sample MBR for a corpus. Returns the (negative) risk of each sample

    :param hyps: list of list containing hypotheses sampled for each source in the cropus
    :param metric: function that computes a metric/utility to estimate the risk
    :param metric_matrixes: matrix of metric values for each hypothesis
    :param srcs (optional): list of sources. used for metrics that depend on the source
    :param subsample (optional): number of num_subsamples to estimate the risk with. if None, use all samples
    """

    if srcs is not None:
        assert len(hyps) == len(srcs), f"{len(hyps)} != {len(srcs)}"

    assert (
        metric is not None or metric_matrixes is not None
    ), "metric or matrixes must be specified"

    num_samples = len(hyps[0])
    use_subsampling = num_subsamples is not None and num_subsamples < num_samples

    if metric_matrixes is None:
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
        # TODO implemet
        raise ValueError("weighted_mean not implemented")
    else:
        raise ValueError(f"aggregation {aggregation} not implemented")

    return neg_risks, metric_matrixes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyps")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument(
        "--metric", default="comet", choices=["bleu", "comet", "bleurt"]
    )
    parser.add_argument("--num-subsamples", type=int, default=None)
    parser.add_argument("--metric-matrixes", default=None)
    parser.add_argument("--comet-dir", default=None)
    parser.add_argument("--bleurt-dir", default=None)
    parser.add_argument("--src", default=None)
    parser.add_argument("--save-metric-matrixes", default=None)
    parser.add_argument("--seed")
    args = parser.parse_args()

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

    if args.metric_matrixes is not None:
        metric = None
        metric_matrixes = np.load(args.metric_matrixes)
    else:
        metric_matrixes = None
        if args.metric == "comet":
            from metrics import comet

            assert (
                args.comet_dir is not None
            ), "comet_dir needs to specified for comet metric"
            metric = lambda cands, refs, srcs: comet(
                cands, refs, srcs, comet_dir=args.comet_dir
            )

        if args.metric == "bleurt":
            from metrics import bleurt

            assert (
                args.bleurt_dir is not None
            ), "bleurt_dir needs to specified for bleurt metric"
            metric = lambda cands, refs, srcs: bleurt(
                cands, refs, srcs, bleurt_dir=args.bleurt_dir
            )

        if args.metric == "bleu":
            from metrics import bleu

            metric = bleu

    neg_risk, metric_matrixes = mbr_corpus(
        hyps,
        metric=metric,
        metric_matrixes=metric_matrixes,
        srcs=srcs,
        num_subsamples=args.num_subsamples,
    )

    if args.save_metric_matrixes is not None:
        np.save(args.save_metric_matrixes, metric_matrixes)

    # print best candidates
    for sample_hyps, sample_utilities in zip(hyps, neg_risk):
        print(sample_hyps[np.argmax(sample_utilities)])


if __name__ == "__main__":
    main()
