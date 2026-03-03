# Neural Phoneme Decoder for Speech BCI

A PyTorch implementation of the GRU-based neural phoneme decoder from [Willett et al. (2023)](https://www.nature.com/articles/s41586-023-06377-x), adapted to run on a single consumer GPU.

**78.77% phoneme accuracy — within 1.5% of the Nature 2023 benchmark.**

---

## What This Is

Willett et al. (2023) demonstrated a speech BCI that decodes intended speech from intracortical neural signals into text at 62 words per minute for people with paralysis. The full pipeline has three stages:
```
Stage 1 — GRU Neural Decoder      <- this repo
Stage 2 — Viterbi Search          (not implemented)
Stage 3 — Kaldi Trigram LM        (not implemented)
```

This repo implements Stage 1 only: a bidirectional GRU that maps neural activity recorded from motor cortex (256 electrodes, 20 ms bins) to phoneme probabilities at each 80 ms time step, trained with CTC loss.

The 78.77% reported here is **phoneme accuracy from the decoder stage alone** — the paper reports ~80.3% at this same stage before language model post-processing.

---

## Model
```
Input: T x 256 neural features (threshold crossings + spike band power)
    |
Gaussian Smoothing -> Day-specific Linear Layer -> Softsign
    |
Unfold (32-bin window, stride 4)   [640ms context, 80ms output step]
    |
Bidirectional GRU (5 layers, hidden=512)
    |
Linear -> 41 classes (40 phonemes + CTC blank)
```

**Day-specific input layers** handle electrode signal drift across recording sessions — each day gets its own learned linear transform, initialized to identity.

**CTC loss** allows training without frame-level phoneme alignment labels.

---

## My Changes

The original was designed for a multi-GPU cluster running the full Kaldi pipeline. These adjustments make the GRU decoder stage run on a single consumer GPU:

| Hyperparameter | Original | This Repo |
|---|---|---|
| `nUnits` (GRU hidden size) | 1024 | **512** |
| `batchSize` | 64 | **16** |
| `dropout` | 0.4 | **0.3** |
| Everything else | — | unchanged |

All parameters are set in `train_model.py` with inline comments.

---

## Results

| | This Repo | Willett et al. (2023) |
|---|---|---|
| Metric | Phoneme accuracy (Stage 1 only) | WER (full pipeline) |
| Score | **78.77%** | 9.1% WER / ~80.3% phoneme acc. |
| Hardware | Single consumer GPU | Multi-GPU HPC cluster |

---

## Setup

**Requirements:** Python >= 3.9, PyTorch >= 2.0, CUDA GPU
```bash
git clone https://github.com/YOUR_USERNAME/neural_seq_decoder.git
cd neural_seq_decoder
pip install -e .
```

Download the dataset from [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq).

---

## How to Run

**Step 1 — Format the raw data**
```bash
jupyter notebook notebooks/formatCompetitionData.ipynb
```

Set input and output paths inside the notebook. Produces `ptDecoder_ctc` — a pickle with `train`/`test` splits.

**Step 2 — Set paths and train**

Edit `train_model.py`:
```python
args['outputDir']   = './outputs'
args['datasetPath'] = './data/ptDecoder_ctc'
```

Then run:
```bash
python train_model.py

# or pass paths via CLI
python train_model.py --output_dir ./outputs --dataset_path ./data/ptDecoder_ctc
```

Training prints CTC loss and CER every 100 batches. Best checkpoint saved to `outputDir/modelWeights`.

**Loading a saved model**
```python
from neural_decoder.neural_decoder_trainer import loadModel
model = loadModel('./outputs', nInputLayers=24, device='cuda')
```

---

## References

- Willett et al. (2023). *A high-performance speech neuroprosthesis*. Nature, 620, 1031-1036. https://doi.org/10.1038/s41586-023-06377-x
- PyTorch decoder: [cffan/neural_seq_decoder](https://github.com/cffan/neural_seq_decoder)
- Dataset: [Dryad doi:10.5061/dryad.x69p8czpq](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq)
