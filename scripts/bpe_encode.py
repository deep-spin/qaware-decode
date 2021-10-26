import argparse
import sys

import sentencepiece as spm
import fastBPE
import sacremoses


def load_model(model_path, bpe_type, lang=None):
    if bpe_type == "sentencepiece":
        sp = spm.SentencePieceProcessor()
        sp.Load(model_path)
        return sp
    if bpe_type == "fastbpe":
        tok = sacremoses.MosesTokenizer(lang=lang or "en")
        fastbpe = fastBPE.fastBPE(model_path)
        return (tok, fastbpe)


def encode(model, bpe_type, sentence):
    if bpe_type == "sentencepiece":
        return " ".join(model.encode(sentence, out_type=str))
    if bpe_type == "fastbpe":
        tok, fastbpe = model
        return fastbpe.apply([tok.tokenize(sentence, return_str=True)])[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=str, nargs="+", default=[None])
    parser.add_argument("--outputs", type=str, nargs="+", default=[None])
    parser.add_argument(
        "--bpe-type",
        type=str,
        default="sentencepiece",
        choices=["sentencepiece", "fastbpe"],
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs)

    model = load_model(args.model, args.bpe_type, args.lang)
    for inp, out in zip(args.inputs, args.outputs):
        if inp is not None and out is not None:
            inp_f = open(inp, "r")
            out_f = open(out, "w")
        else:
            inp_f = sys.stdin
            out_f = sys.stdout

        for line in inp_f:
            print(encode(model, args.bpe_type, line), file=out_f)


if __name__ == "__main__":
    main()
