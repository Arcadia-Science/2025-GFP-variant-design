import torch
from esm import FastaBatchedDataset


def data_loader_for_fasta(fasta, batch_size, alphabet):
    dataset = FastaBatchedDataset.from_file(fasta)
    batches = dataset.get_batch_indices(batch_size, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    return data_loader


def generate_embeddings(
    fasta_path,
    toks_per_batch,
    repr_layers,
    model,
    output_dir,
    truncation_seq_length=1022,
):
    alphabet = model.alphabet
    data_loader = data_loader_for_fasta(fasta_path, toks_per_batch, alphabet)
    num_batches = len(data_loader.batch_sampler)

    assert all(
        -(model.num_layers + 1) <= i <= model.num_layers for i in range(repr_layers)
    )
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in range(repr_layers)
    ]

    model.eval()
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            current = batch_idx + 1
            seqs_in_batch = toks.size(0)
            print(
                f"Processing {current} of {num_batches} batches ({seqs_in_batch} sequences)"
            )
            toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }

            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                output_file = output_dir / f"{label}.pt"
                result = {"label": label}

                result["mean_representations"] = {
                    layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                    for layer, t in representations.items()
                }

                torch.save(
                    result,
                    output_file,
                )
