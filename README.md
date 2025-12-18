PENet+ Steganalysis (Alaska2 QF90)
==================================

Lightweight snapshot of the PENet steganalysis code focused on the arbitrary-size model `PENet_Arbitrary_layer_pdc_newHPFv3_DepthMHSA_chooseAct_All_cos.py`.

Repository layout
-----------------
- `PENet_Arbitrary_layer_pdc_newHPFv3_DepthMHSA_chooseAct_All_cos.py`: main train/eval script (arbitrary resolution). Handles HPF selection, training, validation, and test.
- `srm_filter_kernel.py`: 30 SRM high-pass filters used to build the HPF bank (plus Gabor filters generated in code).
- `MPNCOV/`: second-order pooling module used by the network.
- `index_list/`: example split files (train/valid/test) stored as NumPy arrays of image basenames (without extension).
- `PENet_Arbitrary_layer_pdc_newHPFv3_DepthMHSA_chooseAct_All_cos/`: output directory created by the main script for checkpoints/logs (named after the script stem).

Environment / install
---------------------
- See `requirements.txt` for pinned versions. Install via:
  ```bash
  pip install -r requirements.txt
  ```
- Core stack: Python 3.8, PyTorch 1.10.2 (choose the wheel matching your CUDA/ROCm), torchvision 0.11.3, fvcore, opencv-python, scipy, einops, numpy.
- Stego generation for scan/test mode uses `jpeglib` and `conseal` (already listed in requirements).

Data preparation
----------------
- Expected layout (relative to repo root by default):
  ```
  Alaska2/QF_90/cover/{id}.jpg
  Alaska2/QF_90/Stego/conseal_UERD_0.4/{id}.jpg
  ```
  Adjust `COVER_DIR` / `STEGO_DIR` inside the script if your paths differ.
- Splits: the script looks for `index_list/train_index_{DATASET_INDEX}.npy`, `valid_index_{DATASET_INDEX}.npy`, `var_test_index_{DATASET_INDEX}.npy`. Each file must contain an array of basenames (e.g., `00001`, `00002`).
  - Provided example: set `--DATASET_INDEX custom` to use `train_index_custom.npy`, `valid_index_custom.npy`, `var_test_index_custom.npy`.

Running (train -> eval -> test)
-----------------------------
Basic command (UERD 0.4 on GPU 0, using provided custom splits):
```bash
python PENet_Arbitrary_layer_pdc_newHPFv3_DepthMHSA_chooseAct_All_cos.py \
  -alg UERD_0.4 -rate 0.4 -i custom -g 0 \
  --batch-size 8 --epochs 50 \
  --hpf-topk 31 --hpf-bank 62 --hpf-gabor 16 \
  --optimizer adamw --lr 5e-4 --weight-decay 1e-2 \
  --act prelu --leaky-slope 0.10
```
Key options:
- `-g/--gpuNum`: CUDA device id (also sets `CUDA_VISIBLE_DEVICES`).
- `-alg/--STEGANOGRAPHY`: experiment tag used in filenames (choices in the script).
- `-rate/--EMBEDDING_RATE`: only used for naming results.
- `-i/--DATASET_INDEX`: selects which index_list file trio to load.
- `--statePath`: resume from a saved checkpoint.
- `--hpf-*`: controls SRM+Gabor bank size and the learned top-K selection.
- `--act` / `--leaky-slope`: activation choice for conv blocks.
- `--no-bn-recalc`: disable BN running-stat recalculation before eval.

Outputs
-------
- Saved under `PENet_Arbitrary_layer_pdc_newHPFv3_DepthMHSA_chooseAct_All_cos/`:
  - `*-params-*.pt`: best checkpoint (highest val accuracy after `--save-after` epochs).
  - `*-process-params-*.pt`: latest checkpoint from the most recent eval.
  - `*-model_log-*.log`: training/eval log.
- Filenames include tags: `{STEGANOGRAPHY}-{EMBEDDING_RATE}-{DATASET_INDEX}-{times}-lr={lr}`.

Notes
-----
- Default test loader uses the same cover/stego pairs as train/valid; adjust paths if you have a separate test split.
- The HPF selection step runs once before training (see `learnable_select_indices`); ensure the training split is accessible for that step.
- If you change image format to PNG, adjust the extensions in `MyDataset` accordingly (commented lines in the script).
