import argparse
import sys
from collections import defaultdict
import tempfile
import numpy as np

from qaware_decode.metrics import build_metric_fn
from qaware_decode.qe_metrics import comet_qe, mbart_qe, transquest
from qaware_decode.utils import wrapped_partial

from typing import List, Dict
import json
import os
import subprocess


def parse_weights(weights_str: str):
    try:
        with open(weights_str) as f:
            return json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        try:
            return {
                parts.split("=")[0]: parts.split("=")[0]
                for parts in weights_str.split(" ")
            }
        except (IndexError, ValueError):
            raise ValueError("Could not parse weights")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "hyps",
        type=str,
        help="File containing all hypothesis grouped per sentence, with ``num_samples*sentences`` ",
    )
    parser.add_argument(
        "--src",
        required=True,
        help="File containing source sentences.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        required=True,
        help="Number of hypothesis per sentence",
    )
    parser.add_argument(
        "--refs",
        default=None,
        type=str,
        help="File containing reference translations. If passed, will be used for evaluating the chosen hypothesis. "
        "Also necessary if training a reranker",
    )

    parser.add_argument(
        "--scores",
        type=str,
        default=None,
        help="File containing scores (for example, probs or logprobs) for each sentence",
    )
    parser.add_argument(
        "--qe-metrics",
        default=["comet_qe"],
        choices=["comet_qe", "mbart_qe", "transquest"],
        nargs="+",
        help="Metric to use. Currently only bleu, comet and bleurt are supported. Check `qaware_decode/metrics.py` for more details.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Weights to combine the qe-metrics. If not provided, equal weights are used. Also used to initialize the trained reranked",
    )

    parser.add_argument(
        "--train-reranker",
        type=str,
        default=None,
        help="If set, optimizes reranked (qe metric) weights using provided references",
    )
    parser.add_argument(
        "--rerank-metric",
        default="comet",
        choices=["bleu", "comet", "bleurt"],
        help="When training reranker, metric to optimize.",
    )
    parser.add_argument(
        "--eval-metrics",
        default=["bleu", "comet"],
        choices=["bleu", "comet", "bleurt"],
        help="Metric(s) to evaluate the chosen hypothesis",
        nargs="+",
    )

    parser.add_argument(
        "--langpair",
        default=None,
        type=str,
        help="Language pair for source/target. Necessary for mbart-qe",
    )
    parser.add_argument(
        "--comet-dir",
        default=".cache/qaware_decode/comet",
        help="Directory containing the comet models.",
    )
    parser.add_argument(
        "--mbartqe-dir",
        default=".cache/qaware_decode/mbart-qe",
        help="Directory containing the mbart-qe models.",
    )
    parser.add_argument(
        "--bleurt-dir",
        default=None,
        help="Directory containing the bleurt models",
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
        "--travatar-dir",
        default=None,
        help="Directory containing the compiled travatar source code. If not set, uses $TRAVATAR_DIR",
    )

    return parser.parse_args()


def rerank(
    hyps: List[List[str]],
    srcs: List[str],
    qe_metrics: List[callable],
    scores: List[List[float]] = None,
    weights: List[float] = None,
    return_features: bool = False,
) -> List[List[float]]:
    """
    Rerank hypotheses using qe_metrics.
    """
    if weights is None:
        if len(qe_metrics) > 1:
            print("No weights provided. Using equal weights...", file=sys.stderr)
        weights = defaultdict(lambda: 1)
        if scores is not None:
            print(
                "scores were passed but not weights. Scores will be ignored...",
                file=sys.stderr,
            )
            weights["score"] = 0.0

    # flatten hyps
    flat_hyps = [hyp for sent_hyps in hyps for hyp in sent_hyps]
    dup_srcs = [src for src in srcs for _ in range(len(hyps[0]))]

    all_features = [[{} for _ in hyp] for hyp in hyps]
    for qe_metric in qe_metrics:
        qe_scores, _ = qe_metric(flat_hyps, dup_srcs)
        multiscore = isinstance(qe_scores, dict)
        for i, sent_hyps in enumerate(hyps):
            for j, hyp in enumerate(sent_hyps):
                if multiscore:
                    for submetric, subscores in qe_scores.items():
                        all_features[i][j][
                            f"{qe_metric.__name__}_{submetric}"
                        ] = subscores[i * len(sent_hyps) + j]
                else:
                    all_features[i][j][f"{qe_metric.__name__}"] = qe_scores[
                        i * len(sent_hyps) + j
                    ]

    if scores is not None:
        for i, sent_hyps in enumerate(hyps):
            for j, hyp in enumerate(sent_hyps):
                all_features[i][j]["score"] = scores[i][j]

    weighted_scores = [
        [
            sum(weights[name] * score for name, score in features.items())
            for features in sentence_feat
        ]
        for sentence_feat in all_features
    ]

    return (weighted_scores, all_features) if return_features else weighted_scores


def compute_hyps_metric(
    hyps: List[List[str]],
    srcs: List[str],
    refs: List[str],
    metric: callable,
) -> float:
    flat_hyps = [hyp for sent_hyps in hyps for hyp in sent_hyps]
    dup_srcs = [src for src in srcs for _ in range(len(hyps[0]))]
    dup_refs = [ref for ref in refs for _ in range(len(hyps[0]))]
    flat_scores, _ = metric(flat_hyps, dup_refs, srcs=dup_srcs)
    return [
        [flat_scores[i * len(sent_hyps) + j] for j, _ in enumerate(sent_hyps)]
        for i, sent_hyps in enumerate(hyps)
    ]


def train_reranker(
    hyps: List[List[str]],
    features: List[List[Dict[str, float]]],
    ref_file: str,
    metric_scores: List[List[float]],
    initial_weights: Dict[str, float] = None,
    seed: int = 9,
    travatar_dir: str = None,
    restarts: int = 1000,
    threads: int = 1,
):
    if initial_weights is None:
        print(
            "No initial weights for training provided. Using equal weights...",
            file=sys.stderr,
        )
        initial_weights = {name: 1 for name in features[0][0].keys()}

    features_file = tempfile.NamedTemporaryFile(mode="w")
    scores_file = tempfile.NamedTemporaryFile(mode="w")
    weights_in_file = tempfile.NamedTemporaryFile(mode="w")
    for sent, (sent_hyps, sent_feats, sent_scores) in enumerate(
        zip(hyps, features, metric_scores)
    ):
        for hyp, hyp_feats, hyp_score in zip(sent_hyps, sent_feats, sent_scores):
            parts = [
                str(sent),
                hyp,
                f"{hyp_feats['score']}" if "score" in hyp_feats else "0",
            ]
            feature_strs = []
            for name, value in hyp_feats.items():
                feature_strs.append(f"{name}={value}")

            parts.append(" ".join(feature_strs))
            print(" ||| ".join(parts), file=features_file)
            print(f"{hyp_score} 1", file=scores_file)
    print(
        " ".join(f"{name}={value}" for name, value in initial_weights.items()),
        file=weights_in_file,
    )
    features_file.flush()
    scores_file.flush()
    weights_in_file.flush()

    if travatar_dir is None:
        assert (
            "TRAVATAR_DIR" in os.environ
        ), "travatar_dir was not provided and $TRAVATAR_DIR is not set"
        travatar_dir = os.environ["TRAVATAR_DIR"]

    weights_out_file = tempfile.NamedTemporaryFile(mode="r+")

    # fmt: off
    subprocess.call(
        [
            os.path.join(travatar_dir, "src/bin/batch-tune"),
            "-nbest", features_file.name,
            "-algorithm", "mert",
            "-weight_in", weights_in_file.name,
            "-eval", "zeroone", "-stat_in", scores_file.name,
            "-restarts", str(restarts),
            "-threads", str(threads),
            "-rand_seed", str(seed),
            ref_file
        ],
        stdout=weights_out_file
    )
    # fmt: on
    weights_out_file.seek(0)
    learned_weights = {
        feat.split("=")[0]: float(feat.split("=")[1])
        for feat in weights_out_file.readlines()[0].split(" ")
    }
    return learned_weights


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

    with open(args.src, encoding="utf-8") as src_f:
        srcs = [line.strip() for line in src_f.readlines()]

    # read logprobs scores if any are provided
    if args.scores is not None:
        with open(args.scores, encoding="utf-8") as score_f:
            flat_scores = [float(line.strip()) for line in score_f.readlines()]
            assert len(flat_scores) % args.num_samples == 0

            # unflatten the scores
            scores = []
            for i in range(0, len(flat_scores) // args.num_samples):
                scores.append([])
                for j in range(args.num_samples):
                    scores[i].append(flat_scores[i * args.num_samples + j])

    # TODO: make builder function
    qe_metrics = []
    for qe_metric in args.qe_metrics:
        if qe_metric == "comet_qe":
            assert args.comet_dir is not None
            qe_metrics.append(wrapped_partial(comet_qe, cometqe_dir=args.comet_dir))
        elif qe_metric == "mbart_qe":
            assert args.mbartqe_dir is not None
            qe_metrics.append(
                wrapped_partial(
                    mbart_qe, mbartqe_dir=args.mbartqe_dir, langpair=args.langpair
                )
            )
        elif qe_metric == "tranquest":
            qe_metrics.append(transquest)

    weighted_scores, features = rerank(
        hyps=hyps,
        srcs=srcs,
        scores=scores if args.scores is not None else None,
        qe_metrics=qe_metrics,
        weights=parse_weights(args.weights) if args.weights is not None else None,
        return_features=True,
    )

    predictions = []
    for sent_hyps, sent_scores in zip(hyps, weighted_scores):
        predictions.append(sent_hyps[np.argmax(sent_scores)])
        print(predictions[-1])

    if args.refs is not None:
        with open(args.refs, encoding="utf-8") as ref_f:
            refs = [line.strip() for line in ref_f.readlines()]

        assert len(refs) == len(srcs)

        decode_metrics = []
        for metric in args.eval_metrics:
            metric_fn = build_metric_fn(
                metric,
                comet_dir=args.comet_dir,
                bleurt_dir=args.bleurt_dir,
                n_cpus=args.n_cpus,
                n_gpus=args.n_gpus,
                only_sentence_level=False,
            )
            decode_metrics.append(
                f"{metric}={metric_fn(predictions, refs, srcs=srcs)[1]}"
            )

        print(" ".join(decode_metrics), file=sys.stderr)

    if args.train_reranker is not None:
        assert args.refs is not None

        # compute metric scores
        metric_fn = build_metric_fn(
            args.rerank_metric,
            comet_dir=args.comet_dir,
            bleurt_dir=args.bleurt_dir,
            n_cpus=args.n_cpus,
            n_gpus=args.n_gpus,
        )
        metric_scores = compute_hyps_metric(hyps, srcs, refs, metric=metric_fn)

        learned_weights = train_reranker(
            hyps=hyps,
            features=features,
            ref_file=args.refs,
            metric_scores=metric_scores,
            initial_weights=parse_weights(args.weights)
            if args.weights is not None
            else None,
            travatar_dir=args.travatar_dir,
        )

        with open(args.train_reranker, "w") as weights_out_file:
            json.dump(learned_weights, weights_out_file)


if __name__ == "__main__":
    main()
