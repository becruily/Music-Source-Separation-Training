# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn
from typing import Dict, Union, List, Tuple, Optional 
from ml_collections import ConfigDict 

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils import demix, get_model_from_config, normalize_audio, denormalize_audio, draw_spectrogram
from utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings
warnings.filterwarnings("ignore")


def run_folder(model: nn.Module, args: argparse.Namespace, config: ConfigDict, device: torch.device, verbose: bool = False):
    """
    Process a folder of audio files for source separation or enhancement.
    """
    start_time = time.time()
    model.eval()

    is_enhancement = config.training.get('is_gan', False) and args.model_type == 'apollo'

    if is_enhancement:
        instruments = ['enhanced']
        print("Running inference in Enhancement mode.")
    else:
        instruments = prefer_target_instrument(config)[:]
        if not instruments:
            print("Warning: Instruments list missing in config for separation task. Using generic 'output'.")
            instruments = ['output']
        print(f"Running inference in Separation mode for: {instruments}")

    mixture_paths = sorted(glob.glob(os.path.join(args.input_folder, '*.*')))
    sample_rate = config.get('audio', {}).get('sample_rate', 44100)
    output_extension = config.get('inference', {}).get('extension', 'wav')

    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")

    os.makedirs(args.store_dir, exist_ok=True)

    process_desc = "Processing files"
    mixture_paths_iter = tqdm(mixture_paths, desc=process_desc) if not verbose else mixture_paths

    detailed_pbar = not args.disable_detailed_pbar

    for path in mixture_paths_iter:
        if verbose: print(f"Processing track: {path}")
        try:
            mix, sr = sf.read(path, dtype='float32', always_2d=True)
            mix = mix.T
            if sr != sample_rate:
                print(f"Resampling {os.path.basename(path)} from {sr}Hz to {sample_rate}Hz")
                mix = librosa.resample(mix, orig_sr=sr, target_sr=sample_rate, res_type='kaiser_best')
        except Exception as e:
            print(f'Cannot read track: {path}')
            print(f'Error message: {str(e)}')
            continue

        expected_channels = config.get('discriminator', {}).get('nch', 2) if is_enhancement else config.get('audio',{}).get('num_channels', 2)
        if mix.shape[0] == 1 and expected_channels == 2:
            if verbose: print(f'Converting mono track to stereo...')
            mix = np.concatenate([mix, mix], axis=0)
        elif mix.shape[0] != expected_channels:
             print(f"Warning: Input channel mismatch ({mix.shape[0]}) for {path}, expected {expected_channels}. Skipping.")
             continue

        mix_orig = mix.copy()
        norm_params = None
        if config.get('inference', {}).get('normalize', False):
            mix, norm_params = normalize_audio(mix)

        try:
             waveforms_orig = demix(config, model, mix, device, model_type=args.model_type, pbar=detailed_pbar)
        except Exception as e:
             print(f"Error during demix for {path}: {e}. Skipping.")
             continue

        if args.use_tta and not is_enhancement:
            waveforms_orig = apply_tta(config, model, mix, waveforms_orig, device, args.model_type)
        elif args.use_tta and is_enhancement and verbose:
             print("Skipping TTA for enhancement.")

        if args.extract_instrumental and not is_enhancement:
            target_instr = 'vocals' if 'vocals' in instruments else instruments[0]
            if target_instr in waveforms_orig:
                waveforms_orig['instrumental'] = mix_orig - waveforms_orig[target_instr]
                if 'instrumental' not in instruments:
                    instruments.append('instrumental')
            else:
                print(f"Warning: Cannot extract instrumental, '{target_instr}' not found in output.")

        file_name = os.path.splitext(os.path.basename(path))[0]

        output_base_dir = os.path.join(args.store_dir, file_name) if not is_enhancement else args.store_dir
        os.makedirs(output_base_dir, exist_ok=True)

        for instr in instruments:
            if instr not in waveforms_orig:
                 print(f"Warning: Instrument '{instr}' not in waveforms_orig output dict for {path}. Skipping save.")
                 continue

            estimates = waveforms_orig[instr]
            if norm_params:
                estimates = denormalize_audio(estimates, norm_params)

            if is_enhancement:
                 out_filename = f"{file_name}_{instr}.{output_extension}"
            else:
                 out_filename = f"{instr}.{output_extension}"

            output_path = os.path.join(output_base_dir, out_filename)

            codec = 'flac' if getattr(args, 'flac_file', False) else 'wav'
            if codec == 'flac':
                 subtype = 'PCM_16' if getattr(args, 'pcm_type', 'PCM_24') == 'PCM_16' else 'PCM_24'
            else:
                 subtype = 'FLOAT'

            try:
                 sf.write(output_path, estimates.T, sample_rate, subtype=subtype)
                 if args.draw_spectro > 0:
                     img_base = os.path.splitext(output_path)[0]
                     output_img_path = f"{img_base}.jpg"
                     draw_spectrogram(estimates.T, sample_rate, args.draw_spectro, output_img_path)
            except Exception as e:
                 print(f"Error writing file {output_path}: {e}")

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of bandit, bandit_v2, bs_roformer, htdemucs, mdx23c, mel_band_roformer,"
                             " scnet, scnet_unofficial, segm_models, swin_upernet, torchseg, apollo") # Added apollo
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Checkpoint path")
    parser.add_argument("--input_folder", type=str, help="folder with mixtures/inputs to process")
    parser.add_argument("--store_dir", type=str, default="", help="path to store results")
    parser.add_argument("--draw_spectro", type=float, default=0,
                        help="Generate spectrograms for N seconds of output.")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
    parser.add_argument("--extract_instrumental", action='store_true',
                        help="invert vocals to get instrumental (separation only)")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    parser.add_argument("--force_cpu", action='store_true', help="Force the use of CPU")
    parser.add_argument("--flac_file", action='store_true', help="Output flac file instead of wav")
    parser.add_argument("--pcm_type", type=str, choices=['PCM_16', 'PCM_24'], default='PCM_24',
                        help="PCM type for FLAC files")
    parser.add_argument("--use_tta", action='store_true',
                        help="Use Test-Time Augmentation (separation only)")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Path to LoRA weights checkpoint")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    return args


def proc_folder(dict_args):
    args = parse_args(dict_args)
    device = torch.device("cpu") # Default to CPU
    if not args.force_cpu and torch.cuda.is_available():
        print('CUDA is available.')
        if not args.device_ids: args.device_ids = [0] # Default to GPU 0
        device = torch.device(f'cuda:{args.device_ids[0]}')
    # elif not args.force_cpu and torch.backends.mps.is_available(): # MPS support can be unstable
    #     device = torch.device("mps")

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True # Generally safe for inference

    model_or_models, config = get_model_from_config(args.model_type, args.config_path)
    is_gan = isinstance(model_or_models, tuple)
    model_to_run = model_or_models[0] if is_gan else model_or_models

    if args.start_check_point:
        load_start_checkpoint(args, model_or_models, type_='inference')

    # Safely access instruments/task info
    if is_gan and args.model_type == 'apollo':
        print("Task: Enhancement")
        config.training = config.get('training', ConfigDict()) # Ensure training section exists
        config.training.instruments = ['enhanced'] # Set default for run_folder
    elif hasattr(config, 'training') and hasattr(config.training, 'instruments'):
        print(f"Task: Separation, Instruments: {config.training.instruments}")
    else:
        print("Task: Unknown (Instruments not specified in config)")
        config.training = config.get('training', ConfigDict())
        config.training.instruments = ['output'] # Generic fallback

    # Apply DataParallel only if multiple GPUs requested and CUDA available
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and torch.cuda.is_available() and not args.force_cpu:
        print(f"Using DataParallel on devices: {args.device_ids}")
        model_to_run = nn.DataParallel(model_to_run, device_ids=args.device_ids)

    model_to_run = model_to_run.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model_to_run, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
