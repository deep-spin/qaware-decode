import argparse

import torch
import sentencepiece as spm

from comet import download_model, load_from_checkpoint
from transquest.algo.sentence_level.monotransquest.run_model import (
    MonoTransQuestModel,
    MonoTransQuestArgs,
)
from mbart_qe import download_mbart_qe, load_mbart_qe


COMETSRC_MODEL = "wmt20-comet-qe-da"
COMETSRC_BATCH_SIZE = 64
TRANSQUEST_MODEL = "TransQuest/monotransquest-da-multilingual"
TRANSQUEST_BATCH_SIZE = 64
MBART_BATCH_SIZE = 64

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyps")
    parser.add_argument("scores")
    parser.add_argument("formatted")
    parser.add_argument("--spm", default=False)
    parser.add_argument("--nbest", required=True, type=int)
    parser.add_argument("--add-cometsrc", default=None)
    parser.add_argument("--add-transquest", action="store_true")
    parser.add_argument("--add-mbart-qe", action="store_true")
    parser.add_argument("--comet-path", default=None)
    parser.add_argument("--src", default=None)
    parser.add_argument("--lp", default=None)
    args = parser.parse_args()

    with open(args.hyps, encoding="utf-8") as hyp_f:
        hyps = [line.strip() for line in hyp_f.readlines()]

    with open(args.scores, encoding="utf-8") as score_f:
        scores = [float(line.strip()) for line in score_f.readlines()]

    def src_hyp_iterator(srcs, hyps):
        assert len(srcs) * args.nbest == len(
            hyps
        ), f"{len(srcs) * args.nbest} != {len(hyps)}"
        for i, src in enumerate(srcs):
            for j in range(args.nbest):
                hyp = hyps[i * args.nbest + j]
                yield src, hyp

    if args.add_cometsrc is not None:
        assert args.src is not None, "source needs to be provided to use COMET"
        with open(args.src, encoding="utf-8") as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        # download comet and load
        comet_path = download_model(COMETSRC_MODEL, args.add_cometsrc)
        comet_model = load_from_checkpoint(comet_path)
        comet_input = [
            {"src": src, "mt": mt} for src, mt in src_hyp_iterator(srcs, hyps)
        ]
        comet_scores, _ = comet_model.predict(
            comet_input,
            num_workers=4,
            batch_size=COMETSRC_BATCH_SIZE,
            sort_by_mtlen=True,
        )
        torch.cuda.empty_cache()

    if args.add_transquest:
        assert args.src is not None, "source needs to be provided to use Transquest"
        with open(args.src, encoding="utf-8") as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        transquest_args = MonoTransQuestArgs(eval_batch_size=TRANSQUEST_BATCH_SIZE)
        transquest_model = MonoTransQuestModel(
            "xlmroberta",
            TRANSQUEST_MODEL,
            num_labels=1,
            use_cuda=torch.cuda.is_available(),
            args=transquest_args,
        )
        transquest_input = [[src, mt] for src, mt in src_hyp_iterator(srcs, hyps)]
        transquest_scores, _ = transquest_model.predict(transquest_input)
        torch.cuda.empty_cache()

    if args.add_mbart_qe is not None:
        assert args.src is not None, "source needs to be provided to use MBART-QE"
        assert args.lp is not None, "MBART-QE requires the language pair to be passed as argument"
        with open(args.src, encoding='utf-8') as src_f:
            srcs = [line.strip() for line in src_f.readlines()]
        
        mbart_path = download_mbart_qe("wmt21-mbart-m2")
        mbart = load_mbart_qe(mbart_path)
        mbart_input = [
            {"src": src, "mt": mt, "lp": args.lp} 
            for src, mt in src_hyp_iterator(srcs, hyps)
        ]
        _, segment_scores = mbart.predict(
            mbart_input, show_progress=True, batch_size=MBART_BATCH_SIZE
        )
        mbart_score = [s[0] for s in segment_scores]
        mbart_uncertainty = [s[1] for s in segment_scores]

    with open(args.formatted, "w", encoding="utf-8") as formatted_f:
        for i, (hyp, score) in enumerate(zip(hyps, scores)):
            sample = i // args.nbest
            parts = [str(sample), hyp, f"{score}"]
            features = [f"logprob={score}"]

            if args.add_cometsrc is not None:
                features.append(f"cometsrc={comet_scores[i]}")

            if args.add_transquest:
                features.append(f"transquest={transquest_scores[i]}")

            if args.add_mbart_qe is not None:
                features.append(f"mbart-uncertainty={mbart_uncertainty[i]}")
                features.append(f"mbart-prediction={mbart_score[i]}")

            parts.append(" ".join(features))
            print(" ||| ".join(parts), file=formatted_f)

if __name__ == "__main__":
    main()
