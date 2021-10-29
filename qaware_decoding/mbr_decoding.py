import numpy as np
import argparse


def mbr_corpus(
    hyps: list[list[str]],
    metric: callable,
    srcs: list[str] = None,
    num_subsamples: int = None,
) -> list[list[float]]:
    """
    Computes per-sample MBR for a corpus. Returns the (negative) risk of each sample

    :param hyps: list of list containing hypotheses sampled for each source in the cropus
    :param metric: function that computes a metric/utility to estimate the risk
    :param srcs (optional): list of sources. used for metrics that depend on the source
    :param subsample (optional): number of num_subsamples to estimate the risk with. if None, use all samples
    """
    if srcs is not None:
        assert len(hyps) == len(srcs), f"{len(hyps)} != {len(srcs)}"

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

    flat_metric_scores, _ = metric(cands, refs, srcs=dup_srcs)

    # unflattens the metrics into a N*S*T tensor
    metric_scores = []
    for i, _ in enumerate(hyps):
        metric_scores.append([])
        for j in range(num_samples):
            metric_scores[i].append([])
            for k in range(num_subsamples if use_subsampling else num_samples):
                metric_scores[i][j].append(
                    flat_metric_scores[
                        i * num_samples * num_samples + j * num_samples + k
                    ]
                )

    metric_scores = np.array(metric_scores)

    # TODO: add other pondering functions other than just mean
    return metric_scores.mean(axis=2).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyps")
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--num-subsamples", type=int, default=None)
    parser.add_argument(
        "--metric", default="comet", choices=["bleu", "comet", "bleurt"]
    )
    parser.add_argument("--comet-dir", default=None)
    parser.add_argument("--bleurt-dir", default=None)
    parser.add_argument("--src", default=None)
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

    if args.metric == "comet":
        from qaware_decoding.metrics import comet

        assert (
            args.comet_dir is not None
        ), "comet_dir needs to specified for comet metric"
        metric = lambda cands, refs, srcs: comet(
            cands, refs, srcs, comet_dir=args.comet_dir
        )

    if args.metric == "bleurt":
        from qaware_decoding.metrics import bleurt

        assert (
            args.bleurt_dir is not None
        ), "bleurt_dir needs to specified for bleurt metric"
        metric = lambda cands, refs, srcs: bleurt(
            cands, refs, srcs, bleurt_dir=args.bleurt_dir
        )

    if args.metric == "bleu":
        from qaware_decoding.metrics import bleu

        metric = bleu

    neg_risk = mbr_corpus(hyps, metric, srcs=srcs, num_subsamples=args.num_subsamples)

    # print best candidates
    for sample_hyps, sample_utilities in zip(hyps, neg_risk):
        print(sample_hyps[np.argmax(sample_utilities)])


if __name__ == "__main__":
    main()
