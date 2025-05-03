# Music Source Separation & Enhancement Training Code

This repository is based on [this pipeline for training models for **music source separation**](https://github.com/ZFTurbo/Music-Source-Separation-Training) and includes **audio enhancement** model using the [Apollo](https://github.com/JusperLee/Apollo) architecture.


## Training

### Training Example (Apollo Enhancement - Type 5: On-the-Fly Degradation)

This example trains Apollo to restore quality from simulated MP3 compression applied during training.

```bash
python train.py ^
    --model_type apollo ^
    --config_path configs/config_apollo.yaml ^
    --start_check_point "" ^
    --results_path results/ ^
    --data_path "D:\TrainingDataClean" ^
    --valid_path "D:\Validation" ^
    --dataset_type 5 ^
    --num_workers 4 ^
    --device_ids 0 ^
    --metrics aura_mrstft ^
    --metric_for_scheduler aura_mrstft
```

## Inference

### Inference Example (Apollo Enhancement)

```bash
python inference.py ^
    --model_type apollo ^
    --config_path configs/config_apollo.yaml ^
    --start_check_point results/apollo_stem_enhancement/last_apollo.ckpt ^
    --input_folder "D:\AudioToEnhance" ^
    --store_dir enhanced_results/ ^
    --device_ids 0
```



## Dataset Types (`--dataset_type`)


*   **Types 1, 2, 3, 4:** Designed for **Source Separation**. They expect different structures for loading individual instrument stems and potentially mixtures. See the original [Dataset Types Documentation](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/dataset_types.md) for details.

*   **Type 5:** Designed for **Audio Enhancement (On-the-Fly Degradation)**.
    *   **Use Case:** Training Apollo to restore audio quality lost due to simulated codec compression (e.g., MP3).
    *   **Training Data (`--data_path`):** Should contain folders with **ONLY CLEAN/ORIGINAL** audio files (`.wav`, `.flac`). The dataloader will randomly select chunks and apply simulated MP3 compression (using `pedalboard`). Filenames of the clean files do not matter.
        ```
        D:\TrainingDataClean\
        ├── song1.wav
        ├── song2.flac
        └── ...
        ```
    *   **Validation Data (`--valid_path`):** Should contain folders with **PAIRS** of pre-degraded audio and their corresponding clean originals. **A consistent naming is required** so the script can find the clean original files based on the degraded files' names. The current script supports finding clean files with suffixes `_orig`, `_clean`.
        ```
        D:\ValidationEnhancementPairs\
        ├── SongA\
        │   ├── dirty.wav    # Degraded audio
        │   └── clean.flac   # Original quality audio
        ├── SongB\
        │   ├── dirty.flac   # Degraded audio
        │   └── clean.wav    # Original quality audio
        └── ...
        ```

*   **Type 6:** Designed for **Audio Enhancement (Pre-Defined Pairs)**.
    *   **Use Case:** Training Apollo to enhance audio where you already have explicit pairs of "dirty" and "clean" files. This is ideal for tasks like enhancing stems previously separated by another model, where the original clean stem is available.
    *   **Training Data (`--data_path`):** Should point to a directory containing **subfolders** for each song/example. Inside each subfolder, there must be exactly one "dirty" audio file and one "clean" audio file.
        *   **Naming:** The script looks for files named **exactly** `dirty.wav` (or `.flac`) and `clean.wav` (or `.flac`) within each subfolder.
        ```
        D:\StemEnhancementTrain\
        ├── SongA\
        │   ├── dirty.wav    # Separated vocals
        │   └── clean.flac   # Original studio vocals
        ├── SongB\
        │   ├── dirty.flac   # Separated drums
        │   └── clean.wav    # Original drum stem
        └── ...
        ```
    *   **Validation Data (`--valid_path`):** Similar to Type 5. It must follow the **exact same structure and naming** as the training data (subfolders containing `dirty.wav`/`.flac` and `clean.wav`/`.flac`). The script will process the `dirty` file and compare the output against the `clean` file for metrics.

## Key Differences from Original Apollo Code

*   **Checkpointing:** Saves only model weights (`generator_state_dict` and `discriminator_state_dict` for Apollo) in `.ckpt` files. To use Apollo checkpoints with other tools (like UVR), you may need to extract the `generator_state_dict`.
*   **Dataset Loading:** Integrated into the existing `MSSDataset` class with specific logic for Types 5 and 6, rather than using the original `MusdbMoisesdbDataModule`.
*   **Codec Simulation (Type 5):** Uses `pedalboard` for MP3 simulation for better cross-platform compatibility compared to the original `torchaudio.functional.apply_codec`.

```
