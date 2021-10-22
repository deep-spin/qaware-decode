import argparse
from collections import Counter

from comet import download_model, load_from_checkpoint

import sacrebleu
from sacrebleu.compat import sentence_bleu


COMET_MODEL="wmt20-comet-da"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=str)
    parser.add_argument("ref", type=str)
    parser.add_argument("--comet", type=str)
    parser.add_argument("--src", type=str)
    parser.add_argument("--save-segment-level", default=None)

    args = parser.parse_args()

    with open(args.hyp, encoding='utf-8') as hyp_f:
        hyps = [line.strip() for line in hyp_f.readlines()]
    with open(args.ref, encoding='utf-8') as ref_f:
        refs = [line.strip() for line in ref_f.readlines()]

    sentence_metrics = [[] for _ in range(len(refs))]

    print(sacrebleu.corpus_bleu(hyps, [refs]).format())
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        sentence_metrics[i].append(("bleu", sacrebleu.sentence_bleu(hyp, [ref]).score))

    if args.comet is not None:
        assert args.src is not None, "source needs to be provided to use COMET"
        with open(args.src) as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        # download comet and load
        comet_path = download_model(COMET_MODEL, args.comet)
        comet_model = load_from_checkpoint(comet_path)

        print("running comet evaluation....")
        comet_input = [
            {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
        ]
        comet_sentscores, comet_score = comet_model.predict(comet_input, num_workers=4)
        for i, comet_sentscore in enumerate(comet_sentscores):
            sentence_metrics[i].append(("comet", comet_sentscore))

        print(f"COMET = {comet_score:.4f}")

    if args.save_segment_level is not None:
        with open(args.save_segment_level, "w") as f:
            for metrics in sentence_metrics:
                print(" ".join(f"{metric_name}={value}" for metric_name, value in metrics), file=f)

if __name__ == "__main__":
    main()