# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import time
import os
import glob
import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm.auto import tqdm
from ml_collections import ConfigDict
from typing import Tuple, Dict, List, Union
import torch.nn as nn

from utils import demix, get_model_from_config, prefer_target_instrument, draw_spectrogram
from utils import normalize_audio, denormalize_audio, apply_tta, read_audio_transposed, load_start_checkpoint
from metrics import get_metrics
import warnings

warnings.filterwarnings("ignore")


def logging(logs: List[str], text: str, verbose_logging: bool = False) -> None:
    """ Log validation information. """
    print(text)
    if verbose_logging:
        logs.append(text)


def write_results_in_file(store_dir: str, logs: List[str]) -> None:
    """ Write logs to a results file. """
    if not store_dir: return # Skip if no directory specified
    try:
        with open(os.path.join(store_dir, 'results.txt'), 'w') as out:
            for item in logs:
                out.write(item + "\n")
    except Exception as e:
        print(f"Error writing results file: {e}")


def get_mixture_paths(
    args,
    verbose: bool,
    config: ConfigDict,
    extension: str, 
    is_enhancement: bool = False
) -> List[str]:
    """
    Retrieve paths to input files for validation.
    For enhancement (Type 6), finds 'dirty.wav'/'dirty.flac' in subfolders if 'clean' pair exists.
    For separation, finds mixture files.
    """
    try:
        valid_path_list = args.valid_path
    except AttributeError:
        print('Error: --valid_path argument missing.')
        return []

    all_input_paths = []
    task_name = "enhancement inputs" if is_enhancement else "mixtures"

    for path_dir_root in valid_path_list: 
        if is_enhancement:
            potential_song_folders = []
            try:
                 items = sorted(glob.glob(os.path.join(path_dir_root, '*')))
                 potential_song_folders = [p for p in items if os.path.isdir(p) and os.path.basename(p)[0] != '.']
            except Exception as e:
                 print(f"Error searching for song folders in {path_dir_root}: {e}")
                 continue

            if verbose: print(f"Found {len(potential_song_folders)} potential song folders in {path_dir_root}. Checking for pairs...")

            file_types = ['wav', 'flac'] 
            dirty_names = [f"dirty.{ext}" for ext in file_types]
            clean_names = [f"clean.{ext}" for ext in file_types]

            for song_folder_path in potential_song_folders:
                dirty_path_found = None
                clean_path_found = None
                # Check for dirty file
                for d_name in dirty_names:
                    potential_path = os.path.join(song_folder_path, d_name)
                    if os.path.isfile(potential_path):
                        dirty_path_found = potential_path
                        break
                # Check for clean file
                for c_name in clean_names:
                    potential_path = os.path.join(song_folder_path, c_name)
                    if os.path.isfile(potential_path):
                        clean_path_found = potential_path
                        break

                if dirty_path_found and clean_path_found:
                    all_input_paths.append(dirty_path_found)
                    if verbose: print(f"  Found valid pair in: {song_folder_path} -> Adding input: {dirty_path_found}")

        else: 
            try:
                target_pattern = os.path.join(path_dir_root, f'*/mixture.{extension}')
                if verbose: print(f"Searching for separation inputs matching: {target_pattern}")
                found_files = sorted(glob.glob(target_pattern))
                all_input_paths.extend([f for f in found_files if os.path.basename(os.path.dirname(f))[0] != '.'])
            except Exception as e:
                 print(f"Error searching path {path_dir_root} for mixtures: {e}")

    all_input_paths = sorted(list(set(all_input_paths)))

    if verbose:
        print(f'Total {task_name} identified to process: {len(all_input_paths)}')
        if len(all_input_paths) == 0:
             print(f"Please check the structure and naming convention in: {valid_path_list}")
        if hasattr(config, 'inference'):
             print(f'Overlap: {config.inference.get("num_overlap", "N/A")} Batch size: {config.inference.get("batch_size", "N/A")}')

    return all_input_paths


def update_metrics_and_pbar(
        track_metrics: Dict,
        all_metrics: Dict,
        instr: str,
        pbar_dict: Dict,
        mixture_paths_iter: Union[List[str], tqdm],
        verbose: bool = False
) -> None:
    """ Update metrics dictionary and progress bar. """
    for metric_name, metric_value in track_metrics.items():
        if verbose:
            print(f"Metric {metric_name:11s} value: {metric_value:.4f}")
        if metric_name not in all_metrics: all_metrics[metric_name] = {}
        if instr not in all_metrics[metric_name]: all_metrics[metric_name][instr] = []
        all_metrics[metric_name][instr].append(metric_value)
        pbar_dict[f'{metric_name}_{instr}'] = f"{metric_value:.4f}" # Format value for display

    if isinstance(mixture_paths_iter, tqdm):
        try:
            mixture_paths_iter.set_postfix(pbar_dict)
        except Exception:
            pass # Ignore if tqdm object is closed or invalid


def process_audio_files(
    input_paths: List[str], 
    model: torch.nn.Module,
    args,
    config,
    device: torch.device,
    verbose: bool = False,
    is_tqdm: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """ Process audio files for validation (separation or enhancement). """
    is_enhancement = config.training.get('is_gan', False) and args.model_type == 'apollo'

    if is_enhancement:
        instruments = ['enhanced']
        print("Validation mode: Enhancement")
    else:
        instruments = prefer_target_instrument(config)
        print(f"Validation mode: Separation for instruments: {instruments}")

    use_tta = getattr(args, 'use_tta', False)
    store_dir = getattr(args, 'store_dir', '')
    input_extension = getattr(args, 'extension', 'wav')
    output_extension = config.get('inference', {}).get('extension', 'wav')
    target_sr = config.get('audio', {}).get('sample_rate', 44100)

    all_metrics = {
        metric: {instr: [] for instr in instruments}
        for metric in args.metrics
    }

    input_paths_iter = tqdm(input_paths, desc="Processing files") if is_tqdm else input_paths

    for path_to_input in input_paths_iter:
        start_time = time.time()
        try:
            input_audio, sr = read_audio_transposed(path_to_input)
            if input_audio is None: raise IOError("Failed to read input audio.")
        except Exception as e:
             print(f"Error reading input {path_to_input}: {e}. Skipping.")
             continue

        original_clean = None
        if is_enhancement:
            folder = os.path.dirname(path_to_input)
            base_name = os.path.splitext(os.path.basename(path_to_input))[0]
            potential_orig_names = [
                f"restored.wav", f"restored.flac",
                f"{base_name}_orig.wav", f"{base_name}_orig.flac",
                f"{base_name}_clean.wav", f"{base_name}_clean.flac",
                f"{base_name}.wav", f"{base_name}.flac"
            ]
            path_to_original = None
            for name in potential_orig_names:
                 potential_path = os.path.join(folder, name)
                 if os.path.exists(potential_path):
                      path_to_original = potential_path
                      if verbose: print(f"Found original reference: {path_to_original}")
                      break

            if path_to_original:
                 original_clean, sr_orig = read_audio_transposed(path_to_original)
                 if original_clean is None:
                      print(f"Warning: Failed to read original audio {path_to_original} for {path_to_input}. Cannot calculate metrics.")
                 elif sr != sr_orig:
                      print(f"Warning: SR mismatch between input ({sr}Hz) and original ({sr_orig}Hz) for {base_name}. Skipping metrics.")
                      original_clean = None
            else:
                 print(f"Warning: Could not find original clean audio for {path_to_input}. Cannot calculate metrics.")

        mix_for_model = input_audio.copy()
        mix_orig_for_metrics = original_clean.copy() if original_clean is not None else None
        original_mixture_for_sep_metrics = input_audio.copy()

        current_sr = sr
        if sr != target_sr:
             orig_length = mix_for_model.shape[-1]
             if verbose: print(f'Resampling input {os.path.basename(path_to_input)} from {sr}Hz to {target_sr}Hz')
             try:
                 mix_for_model = librosa.resample(mix_for_model, orig_sr=sr, target_sr=target_sr, res_type='kaiser_best')
                 if mix_orig_for_metrics is not None:
                     mix_orig_for_metrics = librosa.resample(mix_orig_for_metrics, orig_sr=sr, target_sr=target_sr, res_type='kaiser_best')
                 if not is_enhancement:
                     original_mixture_for_sep_metrics = librosa.resample(original_mixture_for_sep_metrics, orig_sr=sr, target_sr=target_sr, res_type='kaiser_best')

                 target_len = int(orig_length * target_sr / sr)
                 mix_for_model = librosa.util.fix_length(mix_for_model, size=target_len)
                 if mix_orig_for_metrics is not None: mix_orig_for_metrics = librosa.util.fix_length(mix_orig_for_metrics, size=target_len)
                 if not is_enhancement: original_mixture_for_sep_metrics = librosa.util.fix_length(original_mixture_for_sep_metrics, size=target_len)
                 current_sr = target_sr
             except Exception as resample_e:
                  print(f"Error during resampling for {path_to_input}: {resample_e}. Skipping file.")
                  continue

        if verbose:
             folder_name = os.path.abspath(os.path.dirname(path_to_input))
             print(f'File: {os.path.basename(path_to_input)} | Input Shape: {mix_for_model.shape} | SR: {current_sr}Hz')

        norm_params = None
        if config.get('inference', {}).get('normalize', False):
             mix_for_model, norm_params = normalize_audio(mix_for_model)

        try:
             waveforms_orig = demix(config, model, mix_for_model, device, model_type=args.model_type, pbar=(not is_tqdm and verbose))
        except Exception as demix_e:
             print(f"Error during demix for {path_to_input}: {demix_e}. Skipping file.")
             continue

        if use_tta and not is_enhancement:
             waveforms_orig = apply_tta(config, model, mix_for_model, waveforms_orig, device, args.model_type)
        elif use_tta and is_enhancement and verbose:
             print("Skipping TTA for enhancement task.")

        pbar_dict = {}

        for instr in instruments:
            if instr not in waveforms_orig:
                 print(f"Warning: Instrument '{instr}' not found in model output for {path_to_input}. Skipping.")
                 continue
            estimates = waveforms_orig[instr]

            if current_sr != sr:
                 try:
                      estimates = librosa.resample(estimates, orig_sr=current_sr, target_sr=sr, res_type='kaiser_best')
                      estimates = librosa.util.fix_length(estimates, size=input_audio.shape[-1])
                 except Exception as resample_e:
                      print(f"Error resampling output for {path_to_input}/{instr}: {resample_e}. Skipping metric calculation.")
                      continue

            if norm_params:
                 estimates = denormalize_audio(estimates, norm_params)

            if store_dir:
                os.makedirs(store_dir, exist_ok=True)
                file_base = os.path.splitext(os.path.basename(path_to_input))[0]
                out_wav_name = f"{store_dir}/{file_base}_{instr}.{output_extension}"
                try:
                     sf.write(out_wav_name, estimates.T, sr, subtype='FLOAT')
                     if args.draw_spectro > 0:
                         out_img_name = f"{store_dir}/{file_base}_{instr}.jpg"
                         draw_spectrogram(estimates.T, sr, args.draw_spectro, out_img_name)
                         # Draw original/degraded spectrograms
                         orig_to_draw = original_clean if is_enhancement and original_clean is not None else input_audio
                         if orig_to_draw is not None:
                             draw_spectrogram(orig_to_draw.T, sr, args.draw_spectro, f"{store_dir}/{file_base}_{instr}_orig_ref.jpg")
                         if is_enhancement:
                             draw_spectrogram(input_audio.T, sr, args.draw_spectro, f"{store_dir}/{file_base}_{instr}_input_degraded.jpg")
                except Exception as write_e:
                     print(f"Error writing output file {out_wav_name}: {write_e}")


            reference_audio = None
            mixture_context = None
            if is_enhancement:
                reference_audio = mix_orig_for_metrics
                mixture_context = input_audio
            else:
                folder = os.path.dirname(path_to_input)
                # Look for target stem with multiple extensions
                path_to_target_stem = None
                for ext in [input_extension, 'wav', 'flac']:
                     potential_path = os.path.join(folder, f"{instr}.{ext}")
                     if os.path.exists(potential_path):
                          path_to_target_stem = potential_path
                          break
                if path_to_target_stem:
                    reference_audio, sr_ref = read_audio_transposed(path_to_target_stem, instr, skip_err=True)
                    if reference_audio is not None and sr_ref != sr:
                         print(f"Warning: Sample rate mismatch for target stem {instr} ({sr_ref}Hz) vs input ({sr}Hz). Skipping metrics.")
                         reference_audio = None # Invalidate reference
                mixture_context = original_mixture_for_sep_metrics

            if reference_audio is None:
                 if verbose: print(f"Skipping metrics for {instr} - reference audio not available or invalid.")
                 continue

            min_len = min(reference_audio.shape[-1], estimates.shape[-1])
            reference_audio = reference_audio[..., :min_len]
            estimates_for_metric = estimates[..., :min_len]
            if mixture_context is not None:
                 mixture_context = mixture_context[..., :min_len]

            try:
                 track_metrics = get_metrics(
                     args.metrics,
                     reference=reference_audio,
                     estimate=estimates_for_metric,
                     mix=mixture_context,
                     device=device,
                 )
                 update_metrics_and_pbar(
                     track_metrics, all_metrics, instr, pbar_dict,
                     input_paths_iter, verbose=verbose
                 )
            except Exception as metric_e:
                 print(f"Error calculating metrics for {path_to_input}/{instr}: {metric_e}")


        if verbose:
            print(f"Time for file: {time.time() - start_time:.2f} sec")

    return all_metrics


def compute_metric_avg(
    store_dir: str,
    args,
    instruments: List[str],
    config: ConfigDict,
    all_metrics: Dict[str, Dict[str, List[float]]],
    start_time: float
) -> Dict[str, float]:
    """ Calculate and log average metrics. """
    logs = []
    verbose_logging = bool(store_dir)

    if verbose_logging:
        logs.append(str(args))
        if hasattr(config, 'inference'):
             logs.append(f"Num overlap: {config.inference.get('num_overlap', 'N/A')}")

    metric_avg = {}
    if not all_metrics or not any(all_metrics.values()):
         print("No metrics were calculated.")
         return metric_avg

    valid_instruments_count = 0
    for instr in instruments:
        instr_has_metrics = False
        for metric_name in args.metrics:
            if metric_name in all_metrics and instr in all_metrics[metric_name] and all_metrics[metric_name][instr]:
                instr_has_metrics = True
                metric_values = np.array(all_metrics[metric_name][instr])
                # Filter out potential NaNs or Infs before calculating mean/std
                valid_metric_values = metric_values[np.isfinite(metric_values)]
                if valid_metric_values.size == 0:
                    print(f"Warning: No valid metric values for {instr}/{metric_name}")
                    continue

                mean_val = valid_metric_values.mean()
                std_val = valid_metric_values.std()

                log_text = f"Instr {instr} {metric_name}: {mean_val:.4f} (Std: {std_val:.4f}) Count: {len(valid_metric_values)}"
                logging(logs, text=log_text, verbose_logging=verbose_logging)

                if metric_name not in metric_avg: metric_avg[metric_name] = 0.0
                metric_avg[metric_name] += mean_val

        if instr_has_metrics:
            valid_instruments_count += 1

    if valid_instruments_count > 0:
        for metric_name in metric_avg:
            metric_avg[metric_name] /= valid_instruments_count

        if valid_instruments_count > 1 or len(instruments) == 1:
            for metric_name in metric_avg:
                log_text = f'Metric avg {metric_name:11s}: {metric_avg[metric_name]:.4f}'
                logging(logs, text=log_text, verbose_logging=verbose_logging)
    else:
         print("No valid instruments found with metrics to average.")

    log_text = f"Elapsed time: {time.time() - start_time:.2f} sec"
    logging(logs, text=log_text, verbose_logging=verbose_logging)

    if store_dir:
        write_results_in_file(store_dir, logs)

    return metric_avg


def valid(
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device: torch.device,
    verbose: bool = False
) -> Tuple[dict, dict]:
    """ Validate a trained model (separation or enhancement). """
    start_time = time.time()
    model.eval().to(device)

    store_dir = getattr(args, 'store_dir', '')
    extension = getattr(args, 'extension', 'wav') 
    is_enhancement = config.training.get('is_gan', False) and args.model_type == 'apollo'

    all_input_paths = get_mixture_paths(args, verbose, config, extension, is_enhancement)
    if not all_input_paths:
        print("No input files found for validation.")
        return {}, {}

    all_metrics = process_audio_files(all_input_paths, model, args, config, device, verbose, not verbose)

    instruments = ['enhanced'] if is_enhancement else prefer_target_instrument(config)

    return compute_metric_avg(store_dir, args, instruments, config, all_metrics, start_time), all_metrics


def validate_in_subprocess(
    proc_id: int,
    queue: torch.multiprocessing.Queue,
    all_input_paths: List[str],
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device: str,
    return_dict
) -> None:
    """ Perform validation on a subprocess. """
    try: 
        m1 = model.eval().to(device)
        is_enhancement = config.training.get('is_gan', False) and args.model_type == 'apollo'
        instruments = ['enhanced'] if is_enhancement else config.training.instruments

        if proc_id == 0:
            progress_bar = tqdm(total=len(all_input_paths), desc="MultiGPU Validation")

        all_metrics_proc = {
            metric: {instr: [] for instr in instruments}
            for metric in args.metrics
        }

        while True:
            item = queue.get()
            if item is None: break
            current_step, path = item

            try:
                single_file_metrics = process_audio_files([path], m1, args, config, device, False, False)
                pbar_dict = {}
                for metric_name in args.metrics:
                    for instr in instruments:
                        values = single_file_metrics.get(metric_name, {}).get(instr, [])
                        if values:
                            if metric_name not in all_metrics_proc: all_metrics_proc[metric_name] = {}
                            if instr not in all_metrics_proc[metric_name]: all_metrics_proc[metric_name][instr] = []
                            all_metrics_proc[metric_name][instr].extend(values)
                            pbar_dict[f"{metric_name}_{instr}"] = f"{values[0]:.4f}"

                if proc_id == 0:
                    progress_bar.update(1) # Simple update by 1 per item processed
                    progress_bar.set_postfix(pbar_dict)
            except Exception as e:
                 print(f"Error in subprocess {proc_id} processing {path}: {e}")

        return_dict[proc_id] = all_metrics_proc
        if proc_id == 0 and 'progress_bar' in locals(): progress_bar.close()
    except Exception as e:
         print(f"Critical error in subprocess {proc_id}: {e}")
         return_dict[proc_id] = {} # Return empty dict on critical failure
    finally:
         if proc_id == 0 and 'progress_bar' in locals() and not progress_bar.disable:
              try:
                  progress_bar.close()
              except Exception: pass


def run_parallel_validation(
    verbose: bool,
    all_input_paths: List[str],
    config: ConfigDict,
    model: torch.nn.Module,
    device_ids: List[int],
    args,
    return_dict
) -> None:
    """ Run parallel validation using multiple processes. """
    model_cpu = model.to('cpu')
    if isinstance(model_cpu, nn.DataParallel):
        model_cpu = model_cpu.module

    # Use spawn context for better compatibility, especially with CUDA
    mp_context = torch.multiprocessing.get_context('spawn')
    queue = mp_context.Queue()
    processes = []

    num_devices = len(device_ids) if torch.cuda.is_available() else 1

    for i in range(num_devices):
        if torch.cuda.is_available():
            device_str = f'cuda:{device_ids[i]}'
        else:
            device_str = 'cpu'
        p = mp_context.Process(
            target=validate_in_subprocess,
            args=(i, queue, all_input_paths, model_cpu, args, config, device_str, return_dict)
        )
        p.start()
        processes.append(p)

    for i, path in enumerate(all_input_paths):
        queue.put((i + 1, path))

    for _ in range(num_devices):
        queue.put(None)

    for p in processes:
        p.join()

    queue.close()
    queue.join_thread()


def valid_multi_gpu(
    model: torch.nn.Module,
    args,
    config: ConfigDict,
    device_ids: List[int],
    verbose: bool = False
) -> Tuple[Dict[str, float], dict]:
    """ Perform validation across multiple GPUs. """
    start_time = time.time()

    store_dir = getattr(args, 'store_dir', '')
    extension = getattr(args, 'extension', 'wav')
    is_enhancement = config.training.get('is_gan', False) and args.model_type == 'apollo'

    all_input_paths = get_mixture_paths(args, verbose, config, extension, is_enhancement)
    if not all_input_paths:
        print("No input files found for validation.")
        return {}, {}

    mp_context = torch.multiprocessing.get_context('spawn')
    manager = mp_context.Manager()
    return_dict = manager.dict()

    run_parallel_validation(verbose, all_input_paths, config, model, device_ids, args, return_dict)

    aggregated_metrics = {}
    instruments_list = ['enhanced'] if is_enhancement else prefer_target_instrument(config)

    num_processes = len(return_dict)

    for metric in args.metrics:
        aggregated_metrics[metric] = {}
        for instr in instruments_list:
            aggregated_metrics[metric][instr] = []
            for i in range(num_processes):
                # Check if key exists before accessing
                if i in return_dict:
                    proc_metrics = return_dict[i]
                    if metric in proc_metrics and instr in proc_metrics[metric]:
                         aggregated_metrics[metric][instr].extend(proc_metrics[metric][instr])
                else:
                     print(f"Warning: Results from process {i} not found in return_dict.")


    return compute_metric_avg(store_dir, args, instruments_list, config, aggregated_metrics, start_time), aggregated_metrics


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """ Parse command-line arguments for validation. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit, apollo")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Checkpoint to validate")
    parser.add_argument("--valid_path", nargs="+", type=str, help="Validation data path(s)")
    parser.add_argument("--store_dir", type=str, default="", help="Path to store results (wav, spectrograms)")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="Generate spectrograms for N seconds of output stems if --store_dir is set.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='List of GPU IDs')
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader num_workers (Not used in current valid.py)")
    parser.add_argument("--pin_memory", action='store_true', help="Dataloader pin_memory (Not used in current valid.py)")
    parser.add_argument("--extension", type=str, default='wav', help="Input file extension for validation")
    parser.add_argument("--use_tta", action='store_true',
                        help="Use Test-Time Augmentation (not recommended for enhancement)")
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='List of metrics to calculate.')
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Path to LoRA weights checkpoint")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def check_validation(dict_args):
    """ Main function to run validation """
    args = parse_args(dict_args)
    torch.backends.cudnn.benchmark = True 

    model_or_models, config = get_model_from_config(args.model_type, args.config_path)
    is_gan = isinstance(model_or_models, tuple)

    model_to_validate = model_or_models[0] if is_gan else model_or_models

    if args.start_check_point:
        load_start_checkpoint(args, model_or_models, type_='valid')

    print(f"Instruments/Task: {'Enhancement' if is_gan else config.training.get('instruments', 'N/A')}")

    device_ids = args.device_ids
    if torch.cuda.is_available():
        if not device_ids: device_ids = [0]
        device = torch.device(f'cuda:{device_ids[0]}')
        print(f"Using CUDA device(s): {device_ids}")
    else:
        device = torch.device('cpu')
        device_ids = []
        print('CUDA is not available. Running validation on CPU.')

    if torch.cuda.is_available() and len(device_ids) > 1:
        print("Running multi-GPU validation...")
        valid_multi_gpu(model_to_validate, args, config, device_ids, verbose=False)
    else:
        print("Running single-device validation...")
        valid(model_to_validate, args, config, device, verbose=True)


if __name__ == "__main__":
    # Set start method for multiprocessing if running directly
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass # Ignore if already set or not applicable
    check_validation(None)
