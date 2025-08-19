import argparse
from pathlib import Path

import esm
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
from utils import generate_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta", type=Path, help="Path to the fasta file to create embeddings for.", required=True
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory path to use for embeddings.",
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=1022
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(exist_ok=True, parents=True)

    # The following if from esm: https://github.com/facebookresearch/esm/blob/main/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py
    url = "tcp://localhost:23456"
    torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

    model_name = "esm2_t48_15B_UR50D"
    model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)

    fsdp_params = dict(
        mixed_precision=True,
        flatten_parameters=True,
        state_dict_device=torch.device("cpu"),
        cpu_offload=True,
    )
    with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
        model, _ = esm.pretrained.load_model_and_alphabet_core(
            model_name, model_data, regression_data
        )
        model.eval()

        # Wrap each layer in FSDP separately
        for name, child in model.named_children():
            if name == "layers":
                for layer_name, layer in child.named_children():
                    wrapped_layer = wrap(layer)
                    setattr(child, layer_name, wrapped_layer)
        model = wrap(model)
    # end of esm code

    generate_embeddings(
        fasta_path=args.fasta,
        toks_per_batch=512,
        repr_layers=48,
        model=model,
        output_dir=args.output_dir,
        truncation_seq_length=args.truncation_seq_length
    )


if __name__ == "__main__":
    main()
