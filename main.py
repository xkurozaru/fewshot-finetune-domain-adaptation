import os
import sys
import warnings

from common import param, set_seed
from core import cosine_tune, dann_tune, dist_tune, finetune, pretrain, triplet_tune


def main():
    warnings.filterwarnings("ignore")
    os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu_ids
    set_seed(param.seed)

    args = sys.argv
    if len(args) == 1:
        print("Please input argument.")
        return

    if args[1] == "pretrain":
        pretrain()
    elif args[1] == "finetune":
        finetune()
    elif args[1] == "dist_tune":
        dist_tune()
    elif args[1] == "triplet_tune":
        triplet_tune()
    elif args[1] == "cosine_tune":
        cosine_tune()
    elif args[1] == "dann_tune":
        dann_tune()
    else:
        print("Invalid argument.")
        return


if __name__ == "__main__":
    main()
