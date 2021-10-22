import argparse
import numpy as np
import sacrebleu
from comet import download_model, load_from_checkpoint

from bleurt import score



COMET_MODEL = "wmt20-comet-da"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=str)
    parser.add_argument("ref", type=str)
    parser.add_argument("--comet_model_dir", type=str, default=None)
    parser.add_argument("--bleurt_model_dir", type=str, default=None)
    parser.add_argument("--src", type=str)
    parser.add_argument("--save_segment_level", default=None)

    args = parser.parse_args()

    with open(args.hyp, encoding='utf-8') as hyp_f:
        hyps = [line.strip() for line in hyp_f.readlines()]
    with open(args.ref, encoding='utf-8') as ref_f:
        refs = [line.strip() for line in ref_f.readlines()]

    sentence_metrics = [[] for _ in range(len(refs))]

    # gets corpus-level non-ml evaluation metrics
    # corpus-level BLEU
    print(sacrebleu.corpus_bleu(hyps, [refs]).format())
    # corpus-level chrF
    print(sacrebleu.corpus_chrf(hyps, [refs]).format())
    # corpus-level TER
    print(sacrebleu.corpus_ter(hyps, [refs]).format())

    if args.save_segment_level is not None:
        # gets sentence-level non-ml metrics
        for i, (hyp, ref) in enumerate(zip(hyps, refs)):
            sentence_metrics[i].append(("bleu", sacrebleu.sentence_bleu(hyp, [ref]).score))
            sentence_metrics[i].append(("chrf", sacrebleu.sentence_chrf(hyp, [ref]).score))
            sentence_metrics[i].append(("ter", sacrebleu.sentence_ter(hyp, [ref]).score))

    if args.comet_model_dir is not None:
        assert args.src is not None, "source needs to be provided to use COMET"
        with open(args.src) as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        # download comet and load
        comet_path = download_model(COMET_MODEL, args.comet_model_dir)
        comet_model = load_from_checkpoint(comet_path)

        print("Running COMET evaluation...")
        comet_input = [
            {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
        ]
        # sentence-level and corpus-level COMET
        comet_sentscores, comet_score = comet_model.predict(comet_input)
        for i, comet_sentscore in enumerate(comet_sentscores):
            sentence_metrics[i].append(("comet", comet_sentscore))

        print(f"COMET = {comet_score:.4f}")

    # gets BLEURT scores
    if args.bleurt_model_dir is not None:
        checkpoint = args.bleurt_model_dir
      
        bleurt_scorer = score.BleurtScorer(checkpoint)
        bleurt_scores = bleurt_scorer.score(references=refs, candidates=hyps)
        assert type(bleurt_scores) == list
        # corpus-level BLEURT
        print(f"BLEURT = {np.array(bleurt_scores).mean()}")
        for i, bleurt_score in enumerate(bleurt_scores):
            sentence_metrics[i].append(("bleurt", bleurt_score))

    # saves segment-level scores to the disk
    if args.save_segment_level is not None:
        with open(args.save_segment_level, "w") as f:
            for metrics in sentence_metrics:
                print(" ".join(f"{metric_name}={value}" for metric_name, value in metrics), file=f)

if __name__ == "__main__":
    main()
