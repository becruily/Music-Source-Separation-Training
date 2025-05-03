# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
import os
import soundfile as sf
import matplotlib.pyplot as plt
from ml_collections import ConfigDict
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Union
import loralib as lora
import librosa # Ensure librosa is imported if not already


try:
    from models.apollo.models import Apollo
    from models.apollo.discriminators import MultiFrequencyDiscriminator
except ImportError as e:
    print(f"Warning: Could not import Apollo models. Ensure 'apollo' structure is inside 'models/': {e}")
    class Apollo(nn.Module): pass
    class MultiFrequencyDiscriminator(nn.Module): pass


def load_config(model_type: str, config_path: str) -> Union[ConfigDict, OmegaConf]:
    """
    Load the configuration from the specified path based on the model type.
    Uses ConfigDict for most models, including the new 'apollo'.
    """
    try:
        with open(config_path, 'r') as f:
            if model_type == 'htdemucs':
                 config = OmegaConf.load(config_path)
            else:
                 config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def get_model_from_config(model_type: str, config_path: str) -> Tuple[Union[nn.Module, Tuple[nn.Module, nn.Module]], Union[ConfigDict, OmegaConf]]:
    """
    Load the model specified by the model type and configuration file.
    For 'apollo', returns a tuple: (generator, discriminator).
    """
    config = load_config(model_type, config_path)
    model = None
    discriminator = None

    if model_type == 'apollo':
        generator = Apollo(**config.model)
        discriminator = MultiFrequencyDiscriminator(**config.discriminator)
        return (generator, discriminator), config
    elif model_type == 'mdx23c':
        from models.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        model = TFC_TDF_net(config)
    elif model_type == 'htdemucs':
        from models.demucs4ht import get_model
        model = get_model(config)
    elif model_type == 'segm_models':
        from models.segm_models import Segm_Models_Net
        model = Segm_Models_Net(config)
    elif model_type == 'torchseg':
        from models.torchseg_models import Torchseg_Net
        model = Torchseg_Net(config)
    elif model_type == 'mel_band_roformer':
        from models.bs_roformer import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'mel_band_roformer_experimental':
        from models.bs_roformer.mel_band_roformer_experimental import MelBandRoformer
        model = MelBandRoformer(**dict(config.model))
    elif model_type == 'bs_roformer':
        from models.bs_roformer import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'bs_roformer_experimental':
        from models.bs_roformer.bs_roformer_experimental import BSRoformer
        model = BSRoformer(**dict(config.model))
    elif model_type == 'swin_upernet':
        from models.upernet_swin_transformers import Swin_UperNet_Model
        model = Swin_UperNet_Model(config)
    elif model_type == 'bandit':
        from models.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        model = MultiMaskMultiSourceBandSplitRNNSimple(**config.model)
    elif model_type == 'bandit_v2':
        from models.bandit_v2.bandit import Bandit
        model = Bandit(**config.kwargs)
    elif model_type == 'scnet_unofficial':
        from models.scnet_unofficial import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet':
        from models.scnet import SCNet
        model = SCNet(**config.model)
    elif model_type == 'scnet_tran':
        from models.scnet.scnet_tran import SCNet_Tran
        model = SCNet_Tran(**config.model)
    elif model_type == 'bs_mamba2':
        from models.ts_bs_mamba2 import Separator
        model = Separator(**config.model)
    elif model_type == 'experimental_mdx23c_stht':
        from models.mdx23c_tfc_tdf_v3_with_STHT import TFC_TDF_net
        model = TFC_TDF_net(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model, config


def read_audio_transposed(path: str, instr: str = None, skip_err: bool = False) -> Tuple[np.ndarray, int]:
    """
    Reads an audio file, ensuring mono audio is converted to two-dimensional format,
    and transposes the data to have channels as the first dimension.
    """
    try:
        mix, sr = sf.read(path)
    except Exception as e:
        if skip_err:
            print(f"Skipping stem {instr}: {e}")
            return None, None
        else:
            raise RuntimeError(f"Error reading the file at {path}: {e}")
    else:
        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=-1)
        return mix.T, sr


def normalize_audio(audio: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize an audio signal by subtracting the mean and dividing by the standard deviation.
    """
    mono = audio.mean(0)
    mean, std = mono.mean(), mono.std() + 1e-8 # Add epsilon for stability
    return (audio - mean) / std, {"mean": mean, "std": std}


def denormalize_audio(audio: np.ndarray, norm_params: Dict[str, float]) -> np.ndarray:
    """
    Denormalize an audio signal by reversing the normalization process.
    """
    return audio * norm_params["std"] + norm_params["mean"]


def apply_tta(
        config,
        model: torch.nn.Module,
        mix: torch.Tensor,
        waveforms_orig: Dict[str, torch.Tensor],
        device: torch.device,
        model_type: str
) -> Dict[str, torch.Tensor]:
    """
    Apply Test-Time Augmentation (TTA) for source separation.
    """
    track_proc_list = [mix[::-1].copy(), -1.0 * mix.copy()]

    for i, augmented_mix in enumerate(track_proc_list):
        waveforms = demix(config, model, augmented_mix, device, model_type=model_type)
        for el in waveforms:
            if i == 0:
                waveforms_orig[el] += waveforms[el][::-1].copy()
            else:
                waveforms_orig[el] -= waveforms[el]

    for el in waveforms_orig:
        waveforms_orig[el] /= len(track_proc_list) + 1

    return waveforms_orig


def _getWindowingArray(window_size: int, fade_size: int) -> torch.Tensor:
    """
    Generate a windowing array with a linear fade-in/out.
    """
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)

    window = torch.ones(window_size)
    window[-fade_size:] = fadeout
    window[:fade_size] = fadein
    return window


def demix(
        config: ConfigDict,
        model: torch.nn.Module,
        mix: torch.Tensor,
        device: torch.device,
        model_type: str,
        pbar: bool = False
) -> Dict[str, np.ndarray]:
    """
    Unified function for audio source separation with support for multiple processing modes.
    Always returns a dictionary mapping instruments to numpy arrays.
    """
    mix = torch.tensor(mix, dtype=torch.float32)

    is_enhancement = config.training.get('is_gan', False) and model_type == 'apollo'

    if model_type == 'htdemucs':
        mode = 'demucs'
    else:
        mode = 'generic' 

    if mode == 'demucs':
        chunk_size = config.training.samplerate * config.training.segment
        num_overlap = config.inference.num_overlap
        step = chunk_size // num_overlap
    else:
        chunk_size = config.audio.chunk_size
        num_overlap = config.inference.num_overlap
        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = _getWindowingArray(chunk_size, fade_size).to(device) # Move window to device
        if length_init > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

    if is_enhancement:
        num_instruments = 1 # Only one output: enhanced
    elif mode == 'demucs':
        num_instruments = len(config.training.instruments)
    else: # Generic separation
        num_instruments = len(prefer_target_instrument(config))

    batch_size = config.inference.batch_size
    use_amp = getattr(config.training, 'use_amp', True)

    with torch.cuda.amp.autocast(enabled=use_amp):
        with torch.inference_mode():
            req_shape = (num_instruments,) + mix.shape
            result = torch.zeros(req_shape, dtype=torch.float32).to(device) # Move result to device
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device) # Move counter to device

            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(
                total=mix.shape[1], desc="Processing audio chunks", leave=False, disable=not pbar
            )

            while i < mix.shape[1]:
                part = mix[:, i:i + chunk_size].to(device)
                chunk_len = part.shape[-1]
                pad_len = chunk_size - chunk_len
                if pad_len > 0:
                    if mode == "generic" and chunk_len > chunk_size // 2:
                         pad_mode = "reflect"
                         if chunk_len < pad_len: pad_mode = "constant"
                    else:
                         pad_mode = "constant"
                    part = nn.functional.pad(part, (0, pad_len), mode=pad_mode, value=0)

                batch_data.append(part)
                batch_locations.append((i, chunk_len))
                i += step

                if len(batch_data) >= batch_size or i >= mix.shape[1]:
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr) # Shape: [B, NumInst/1, Chan, Len] or [B, Chan, Len] for Apollo

                    # Ensure x has instrument dimension for consistent processing
                    if is_enhancement and x.dim() == 3: # Apollo might return [B, Chan, Len]
                        x = x.unsqueeze(1) # Add instrument dim -> [B, 1, Chan, Len]
                    elif not is_enhancement and x.dim() == 3: # Some sep models might return [B, Chan, Len] if single target
                        x = x.unsqueeze(1) # Add instrument dim -> [B, 1, Chan, Len]

                    if x.shape[1] != num_instruments:
                        raise RuntimeError(f"Model output instrument count {x.shape[1]} != expected {num_instruments}")

                    for j, (start, seg_len) in enumerate(batch_locations):
                        output_chunk = x[j, ..., :seg_len] # [NumInst/1, Chan, seg_len]
                        if mode == "generic":
                            window = windowing_array # Use the window already on device
                            # Adjust fade in/out for edge chunks
                            current_window = window.clone()
                            if start == 0: current_window[:fade_size] = 1
                            # Check if this is the last segment based on original length and padding
                            effective_end = start + seg_len
                            padded_end = mix.shape[-1]
                            original_end = length_init + (border if length_init > 2 * border and border > 0 else 0)
                            if effective_end >= original_end:
                                current_window[-fade_size:] = 1

                            window_seg = current_window[:seg_len] # Apply window to valid segment length
                            result[..., start:start + seg_len] += output_chunk * window_seg
                            counter[..., start:start + seg_len] += window_seg
                        else: # demucs mode
                            result[..., start:start + seg_len] += output_chunk
                            counter[..., start:start + seg_len] += 1.0

                    batch_data.clear()
                    batch_locations.clear()

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            estimated_sources = result / (counter + 1e-8) # Add epsilon for stability
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if mode == "generic":
                if length_init > 2 * border and border > 0:
                    estimated_sources = estimated_sources[..., border:-border]

    if is_enhancement:
        instruments = ['enhanced']
    elif mode == "demucs":
        instruments = config.training.instruments
    else: # Generic separation
        instruments = prefer_target_instrument(config)

    ret_data = {k: v for k, v in zip(instruments, estimated_sources)}
    return ret_data


def prefer_target_instrument(config: ConfigDict) -> List[str]:
    """
    Return the list of target instruments based on the configuration.
    """
    if getattr(config.training, 'target_instrument', None):
        return [config.training.target_instrument]
    else:
        # Ensure instruments exist in config before returning
        return getattr(config.training, 'instruments', [])


def load_not_compatible_weights(model: torch.nn.Module, weights_or_state_dict: Union[str, Dict], verbose: bool = False, is_state_dict: bool = False) -> None:
    """ Load weights, handling mismatches. Can accept path or state_dict. """
    new_model_state = model.state_dict()
    if is_state_dict:
        old_model_state = weights_or_state_dict
    else:
        old_model_state = torch.load(weights_or_state_dict, map_location='cpu') # Load to CPU

    if isinstance(old_model_state, dict):
        if 'state' in old_model_state: old_model_state = old_model_state['state']
        if 'state_dict' in old_model_state: old_model_state = old_model_state['state_dict']

    loaded_count = 0
    skipped_count = 0
    mismatched_count = 0

    for el in new_model_state:
        if el in old_model_state:
            if new_model_state[el].shape == old_model_state[el].shape:
                if verbose: print(f'Match found for {el}! Action: Copy weights.')
                new_model_state[el] = old_model_state[el].to(new_model_state[el].device).type(new_model_state[el].dtype)
                loaded_count += 1
            else:
                if verbose: print(f'Shape mismatch for {el}: {tuple(new_model_state[el].shape)} != {tuple(old_model_state[el].shape)}')
                ln_new = len(new_model_state[el].shape)
                ln_old = len(old_model_state[el].shape)
                if ln_new == ln_old:
                    try:
                        copy_slice_new = tuple(slice(0, min(new_model_state[el].shape[i], old_model_state[el].shape[i])) for i in range(ln_new))
                        copy_slice_old = tuple(slice(0, min(new_model_state[el].shape[i], old_model_state[el].shape[i])) for i in range(ln_old))

                        new_model_state[el][copy_slice_new] = old_model_state[el][copy_slice_old].to(new_model_state[el].device).type(new_model_state[el].dtype)
                        if verbose: print(f'Action: Partially loaded intersecting shape for {el}.')
                        mismatched_count += 1
                    except Exception as e:
                         if verbose: print(f'Action: Could not partially load {el} due to error: {e}. Skip.')
                         skipped_count += 1
                else:
                    if verbose: print('Action: Different dimension count! Skip.')
                    skipped_count += 1
        else:
            if verbose: print(f'Match not found for {el}! Skip.')
            skipped_count += 1

    model.load_state_dict(new_model_state, strict=False)
    print(f"Weight loading summary: Loaded={loaded_count}, Partially Loaded (mismatch)={mismatched_count}, Skipped={skipped_count}")


def load_lora_weights(model: torch.nn.Module, lora_path: str, device: str = 'cpu') -> None:
    """
    Load LoRA weights into a model.
    """
    lora_state_dict = torch.load(lora_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(lora_state_dict, strict=False)
    if unexpected_keys:
        print(f"Warning: Unexpected keys found while loading LoRA weights: {unexpected_keys}")
    if not missing_keys:
        print("Successfully loaded LoRA weights.")
    else:
        is_expected_missing = all('lora_' not in k for k in missing_keys)
        if is_expected_missing:
             print("Successfully loaded LoRA weights (ignored non-LoRA params).")
        else:
             print(f"Warning: Some LoRA keys might be missing: {missing_keys}")



def load_start_checkpoint(args: argparse.Namespace, model_or_models: Union[torch.nn.Module, Tuple[torch.nn.Module, torch.nn.Module]], type_='train') -> None:
    """ Load the starting checkpoint for standard or GAN models. """
    if not args.start_check_point: return

    print(f'Loading checkpoint: {args.start_check_point} (Type: {type_})')

    is_gan = isinstance(model_or_models, (tuple, list)) and len(model_or_models) == 2

    if type_ == 'train':
        model = model_or_models
        load_not_compatible_weights(model, args.start_check_point, verbose=False)
        if args.lora_checkpoint:
            print(f"Loading LoRA weights from: {args.lora_checkpoint}")
            load_lora_weights(model, args.lora_checkpoint)

    elif type_ == 'train_gan':
        if not is_gan: raise ValueError("Type 'train_gan' requires model_or_models to be (generator, discriminator)")
        generator, discriminator = model_or_models
        checkpoint = torch.load(args.start_check_point, map_location='cpu')
        if 'generator_state_dict' in checkpoint and 'discriminator_state_dict' in checkpoint:
            print("Loading base weights for Generator...")
            load_not_compatible_weights(generator, checkpoint['generator_state_dict'], verbose=False, is_state_dict=True)
            print("Loading weights for Discriminator...")
            load_not_compatible_weights(discriminator, checkpoint['discriminator_state_dict'], verbose=False, is_state_dict=True)
            if args.lora_checkpoint:
                print(f"Loading LoRA weights from: {args.lora_checkpoint} onto Generator")
                load_lora_weights(generator, args.lora_checkpoint)
        else:
            print("Warning: GAN checkpoint format invalid. Expected 'generator_state_dict' and 'discriminator_state_dict'. Loading failed.")

    elif type_ in ['valid', 'inference']:
        model_to_load = model_or_models[0] if is_gan else model_or_models
        device = 'cpu'
        loaded_data = torch.load(args.start_check_point, map_location=device)

        state_dict = None
        if isinstance(loaded_data, dict):
            if 'generator_state_dict' in loaded_data and is_gan:
                state_dict = loaded_data['generator_state_dict']
                print("Loaded generator_state_dict for validation/inference.")
            elif 'state' in loaded_data:
                 state_dict = loaded_data['state']
                 print("Loaded 'state' key for validation/inference.")
            elif 'state_dict' in loaded_data:
                 state_dict = loaded_data['state_dict']
                 print("Loaded 'state_dict' key for validation/inference.")
            else:
                 state_dict = loaded_data
                 print("Loaded raw state_dict for validation/inference.")
        else:
            state_dict = loaded_data
            print("Loaded raw state_dict for validation/inference.")

        if state_dict:
            missing_keys, unexpected_keys = model_to_load.load_state_dict(state_dict, strict=False)
            if unexpected_keys: print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            if missing_keys: print(f"Warning: Missing keys in model: {missing_keys}")
        else:
            print("Warning: Could not extract state_dict from checkpoint.")

        if args.lora_checkpoint:
            print(f"Loading LoRA weights from: {args.lora_checkpoint} for validation/inference")
            load_lora_weights(model_to_load, args.lora_checkpoint)
    else:
        raise ValueError(f"Unknown checkpoint loading type: {type_}")


def bind_lora_to_model(config: Dict[str, Any], model: nn.Module) -> nn.Module:
    """
    Replaces specific layers in the model with LoRA-extended versions.
    """
    if 'lora' not in config:
        raise ValueError("Configuration must contain the 'lora' key with parameters for LoRA.")

    lora_config = config['lora']
    replaced_layers = 0

    for name, module in model.named_modules():
        if '.' not in name and isinstance(module, nn.Linear): # Target direct Linear layers
            try:
                parent_module = model
                layer_name = name

                setattr(
                    parent_module,
                    layer_name,
                    lora.MergedLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        **lora_config
                    )
                )
                replaced_layers += 1
            except Exception as e:
                print(f"Error replacing layer {name}: {e}")
        elif '.' in name: # Handle nested layers
             hierarchy = name.split('.')
             layer_name = hierarchy[-1]
             parent_module = model
             try:
                 for submodule_name in hierarchy[:-1]:
                     parent_module = getattr(parent_module, submodule_name)

                 module_to_replace = getattr(parent_module, layer_name)
                 if isinstance(module_to_replace, nn.Linear):
                      setattr(
                          parent_module,
                          layer_name,
                          lora.MergedLinear(
                              in_features=module_to_replace.in_features,
                              out_features=module_to_replace.out_features,
                              bias=module_to_replace.bias is not None,
                              **lora_config
                          )
                      )
                      replaced_layers += 1
             except AttributeError:
                  pass
             except Exception as e:
                  print(f"Error processing nested layer {name}: {e}")


    if replaced_layers == 0:
        print("Warning: No Linear layers were replaced with LoRA. Check model structure and LoRA target config.")
    else:
        print(f"Number of Linear layers replaced with LoRA: {replaced_layers}")

    return model


def draw_spectrogram(waveform, sample_rate, length, output_file):
    """ Draws and saves a spectrogram """
    try:
        import librosa.display

        max_samples = int(length * sample_rate)
        if waveform.shape[0] > max_samples:
             x = waveform[:max_samples, :]
        else:
             x = waveform

        if x.shape[0] == 0:
             print(f"Warning: Cannot draw spectrogram for empty waveform ({output_file})")
             return

        mono_signal = x.mean(axis=-1)
        fig, ax = plt.subplots() # Create figure and axes explicitly

        if np.std(mono_signal) < 1e-6:
             print(f"Warning: Waveform is near silent, drawing blank spectrogram ({output_file})")
             ax.set_facecolor('black')
             ax.set_xticks([])
             ax.set_yticks([])
             ax.set_title(f'File: {os.path.basename(output_file)} (Silent)')
        else:
             X = librosa.stft(mono_signal)
             Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)
             img = librosa.display.specshow(
                 Xdb,
                 cmap='plasma',
                 sr=sample_rate,
                 x_axis='time',
                 y_axis='linear',
                 ax=ax
             )
             ax.set(title='File: ' + os.path.basename(output_file))
             fig.colorbar(img, ax=ax, format="%+2.f dB")

        if output_file is not None:
            plt.savefig(output_file)
        else:
            plt.show()
        plt.close(fig) # Ensure figure is closed

    except ImportError:
        print("Librosa is required for drawing spectrograms. Please install it.")
    except Exception as e:
        print(f"Error drawing spectrogram for {output_file}: {e}")
        if 'fig' in locals(): plt.close(fig) # Attempt to close figure on error
