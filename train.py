# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'
__version__ = '1.0.4'

import random
import argparse

from torch_log_wmse import LogWMSE
from tqdm.auto import tqdm
import os
import time
import torch
import wandb
import numpy as np
import auraloss
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, RAdam, RMSprop
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR 
from ml_collections import ConfigDict
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Callable, Any, Optional

from dataset import MSSDataset
from utils import get_model_from_config, load_not_compatible_weights 
from valid import valid_multi_gpu, valid

from utils import bind_lora_to_model, load_start_checkpoint
import loralib as lora

try:
    from models.apollo.losses import MultiFrequencyGenLoss, MultiFrequencyDisLoss
except ImportError:
    print("Warning: Could not import Apollo losses. Ensure 'apollo' structure is inside 'models/'.")
    class MultiFrequencyGenLoss(nn.Module):
         def forward(self, *args, **kwargs): return torch.tensor(0.0)
    class MultiFrequencyDisLoss(nn.Module):
         def forward(self, *args, **kwargs): return torch.tensor(0.0)


import warnings
warnings.filterwarnings("ignore")


def parse_args(dict_args: Union[Dict, None]) -> argparse.Namespace:
    """
    Parse command-line arguments for configuring the model, dataset, and training parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='mdx23c',
                        help="One of mdx23c, htdemucs, segm_models, mel_band_roformer, bs_roformer, swin_upernet, bandit, apollo") # Added apollo
    parser.add_argument("--config_path", type=str, help="path to config file")
    parser.add_argument("--start_check_point", type=str, default='', help="Initial checkpoint to start training")
    parser.add_argument("--results_path", type=str,
                        help="path to folder where results will be stored (weights, metadata)")
    parser.add_argument("--data_path", nargs="+", type=str, help="Dataset data paths. You can provide several folders.")
    parser.add_argument("--dataset_type", type=int, default=1,
                        help="Dataset type. Must be one of: 1, 2, 3, 4 (separation) or 5 (enhancement).") # Added 5
    parser.add_argument("--valid_path", nargs="+", type=str,
                        help="validation data paths. You can provide several folders.")
    parser.add_argument("--num_workers", type=int, default=0, help="dataloader num_workers")
    parser.add_argument("--pin_memory", action='store_true', help="dataloader pin_memory")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--device_ids", nargs='+', type=int, default=[0], help='list of gpu ids')
    parser.add_argument("--loss", type=str, nargs='+', choices=['masked_loss', 'mse_loss', 'l1_loss', 'multistft_loss', 'spec_masked_loss', 'log_wmse_loss', 'gan_loss'], # Added gan_loss placeholder
                        default=['masked_loss'], help="List of loss functions to use (gan_loss used implicitly for apollo)")
    parser.add_argument("--masked_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--mse_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--l1_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--log_wmse_loss_coef", type=float, default=1., help="Coef for loss")
    parser.add_argument("--multistft_loss_coef", type=float, default=0.001, help="Coef for loss")
    parser.add_argument("--spec_masked_loss_coef", type=float, default=1, help="Coef for loss")
    parser.add_argument("--wandb_key", type=str, default='', help='wandb API Key')
    parser.add_argument("--pre_valid", action='store_true', help='Run validation before training')
    parser.add_argument("--metrics", nargs='+', type=str, default=["sdr"],
                        choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='List of metrics to use.')
    parser.add_argument("--metric_for_scheduler", default="sdr",
                        choices=['sdr', 'l1_freq', 'si_sdr', 'log_wmse', 'aura_stft', 'aura_mrstft', 'bleedless',
                                 'fullness'], help='Metric which will be used for scheduler.')
    parser.add_argument("--train_lora", action='store_true', help="Train with LoRA")
    parser.add_argument("--lora_checkpoint", type=str, default='', help="Initial checkpoint to LoRA weights")

    if dict_args is not None:
        args = parser.parse_args([])
        args_dict = vars(args)
        args_dict.update(dict_args)
        args = argparse.Namespace(**args_dict)
    else:
        args = parser.parse_args()

    if args.metric_for_scheduler not in args.metrics:
        args.metrics += [args.metric_for_scheduler]

    return args


def manual_seed(seed: int) -> None:
    """ Set the random seed for reproducibility. """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Deterministic needs benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def initialize_environment(seed: int, results_path: str) -> None:
    """ Initialize the environment. """
    manual_seed(seed)
    # Keep benchmark False if deterministic is True
    # torch.backends.cudnn.benchmark = True # Can be faster but less reproducible
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass # Already set or not applicable
    os.makedirs(results_path, exist_ok=True)


def gen_wandb_name(args, config):
    """ Generate a name for wandb run. """
    if config.training.get('is_gan', False):
        instrum = "enhancement" # Task name for Apollo
    else:
        instrum = '-'.join(config.training.get('instruments', ['unknown']))
    time_str = time.strftime("%Y-%m-%d_%H-%M") # Add time for uniqueness
    name = f'{args.model_type}_{instrum}_{time_str}'
    return name


def wandb_init(args: argparse.Namespace, config: ConfigDict, device_ids: List[int], batch_size: int) -> None:
    """ Initialize Weights & Biases. """
    if args.wandb_key is None or args.wandb_key.strip() == '':
        wandb.init(mode='disabled')
        print("WandB disabled.")
    else:
        try:
            wandb.login(key=args.wandb_key)
            # Convert ConfigDict to simple dict for wandb logging
            config_log = config.to_dict() if isinstance(config, ConfigDict) else config
            wandb.init(
                project='msst', # Project name
                name=gen_wandb_name(args, config),
                config={'config': config_log, 'args': vars(args), 'device_ids': device_ids, 'batch_size': batch_size }
            )
            print(f"WandB initialized for run: {wandb.run.name}")
        except Exception as e:
            print(f"WandB initialization failed: {e}. Disabling WandB.")
            wandb.init(mode='disabled')


def prepare_data(config: ConfigDict, args: argparse.Namespace, batch_size: int, dataset_type_override: Optional[int] = None) -> DataLoader:
    """ Prepare the training dataset and data loader. """
    dataset_type_to_use = dataset_type_override if dataset_type_override is not None else args.dataset_type

    trainset = MSSDataset(
        config,
        args.data_path,
        batch_size=batch_size,
        metadata_path=os.path.join(args.results_path, f'metadata_{dataset_type_to_use}.pkl'),
        dataset_type=dataset_type_to_use,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True # Often helpful for distributed/GAN training
    )
    return train_loader


def initialize_model_and_device(model: torch.nn.Module, device_ids: List[int]) -> Tuple[torch.device, torch.nn.Module]:
    """ Initialize the model and assign it to the appropriate device(s). """
    if torch.cuda.is_available():
        if not device_ids: # If empty list, use default device 0
             device_ids = [0]
        device = torch.device(f'cuda:{device_ids[0]}')
        if len(device_ids) > 1:
            print(f"Using DataParallel on devices: {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
        model = model.to(device)
        print(f"Model placed on CUDA device: {device}")
    else:
        device = torch.device('cpu')
        model = model.to(device)
        print("CUDA is not available. Running on CPU.")
    return device, model


def get_optimizer(config: ConfigDict, model: Union[torch.nn.Module, Tuple[torch.nn.Module, torch.nn.Module]]) -> Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]]:
    """ Initializes optimizer(s) based on the configuration. """
    is_gan = config.training.get('is_gan', False)

    def _create_optimizer(opt_name, params, opt_config):
        lr = opt_config.get('lr', 0.001) # Default LR
        optim_params = {k: v for k, v in opt_config.items() if k not in ['name', 'lr']}
        if opt_name == 'adam':
            return Adam(params, lr=lr, **optim_params)
        elif opt_name == 'adamw':
            return AdamW(params, lr=lr, **optim_params)
        elif opt_name == 'radam':
            return RAdam(params, lr=lr, **optim_params)
        elif opt_name == 'rmsprop':
            return RMSprop(params, lr=lr, **optim_params)
        elif opt_name == 'prodigy':
            try:
                from prodigyopt import Prodigy
                # Prodigy often uses lr=1.0 by default, allow override
                lr = opt_config.get('lr', 1.0)
                return Prodigy(params, lr=lr, **optim_params)
            except ImportError:
                print("Prodigy optimizer not found. Install with 'pip install prodigyopt'. Falling back to AdamW.")
                return AdamW(params, lr=lr, **optim_params)
        elif opt_name == 'adamw8bit':
            try:
                import bitsandbytes as bnb
                return bnb.optim.AdamW8bit(params, lr=lr, **optim_params)
            except ImportError:
                print("bitsandbytes not found. Install with 'pip install bitsandbytes'. Falling back to AdamW.")
                return AdamW(params, lr=lr, **optim_params)
        elif opt_name == 'sgd':
            return SGD(params, lr=lr, **optim_params)
        else:
            print(f'Unknown optimizer: {opt_name}. Using AdamW.')
            return AdamW(params, lr=lr, **optim_params)

    if is_gan:
        if not isinstance(model, (tuple, list)) or len(model) != 2:
            raise ValueError("For GAN training, model must be a tuple (generator, discriminator)")
        generator, discriminator = model

        optim_g_config = config.get('optimizer_g', ConfigDict({'name': 'adamw', 'lr': 0.001}))
        optim_d_config = config.get('optimizer_d', ConfigDict({'name': 'adamw', 'lr': 0.0001}))

        name_optimizer_g = optim_g_config.get('name', 'adamw')
        name_optimizer_d = optim_d_config.get('name', 'adamw')

        print(f"Generator Optimizer: {name_optimizer_g}, Config: {optim_g_config.to_dict()}")
        print(f"Discriminator Optimizer: {name_optimizer_d}, Config: {optim_d_config.to_dict()}")

        optimizer_g = _create_optimizer(name_optimizer_g, generator.parameters(), optim_g_config)
        optimizer_d = _create_optimizer(name_optimizer_d, discriminator.parameters(), optim_d_config)

        return {'G': optimizer_g, 'D': optimizer_d}
    else:
        # Use .get for safer access to nested attributes
        optim_name_default = config.training.get('optimizer', 'adamw')
        lr_default = config.training.get('lr', 0.001)
        optim_config = config.get('optimizer', ConfigDict({'name': optim_name_default, 'lr': lr_default}))
        name_optimizer = optim_config.get('name', optim_name_default)

        print(f"Optimizer: {name_optimizer}, Config: {optim_config.to_dict()}")
        optimizer = _create_optimizer(name_optimizer, model.parameters(), optim_config)
        return optimizer


def multistft_loss(y_: torch.Tensor, y: torch.Tensor,
                   loss_multistft: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """ Helper for MultiSTFT loss shaping. """
    if y_.dim() == 4:
        b, n, c, l = y_.shape
        y1_ = y_.reshape(b * n, c, l)
    elif y_.dim() == 3:
        y1_ = y_
    else:
        raise ValueError(f"Invalid shape for predicted array: {y_.shape}. Expected 3 or 4 dimensions.")

    if y.dim() == 4:
        b, n, c, l = y.shape
        y1 = y.reshape(b * n, c, l)
    elif y.dim() == 3:
        y1 = y
    else:
        raise ValueError(f"Invalid shape for target array: {y.shape}. Expected 3 or 4 dimensions.")

    return loss_multistft(y1_, y1)


def masked_loss(y_: torch.Tensor, y: torch.Tensor, q: float, coarse: bool = True) -> torch.Tensor:
    """ Masked MSE loss based on quantile. """
    if y_.shape != y.shape:
         raise ValueError(f"Shape mismatch in masked_loss: Pred {y_.shape}, Target {y.shape}")

    if y_.dim() == 4:
        loss = torch.nn.MSELoss(reduction='none')(y_, y).mean(dim=2)
        if coarse:
            loss = loss.mean(dim=-1)
        loss = loss.reshape(loss.shape[0], -1)
    elif y_.dim() == 3:
        loss = torch.nn.MSELoss(reduction='none')(y_, y).mean(dim=1)
        if coarse:
             loss = loss.mean(dim=-1)
        loss = loss.reshape(loss.shape[0], -1)
    else:
        raise ValueError(f"Unexpected shape in masked_loss: {y_.shape}")

    if loss.numel() == 0: return torch.tensor(0.0, device=y_.device)

    quantile = torch.quantile(loss.detach(), q, interpolation='linear', dim=1, keepdim=True)
    mask = loss < quantile
    masked_loss_val = (loss * mask).sum() / (mask.sum() + 1e-8)
    return masked_loss_val


def spec_masked_loss(estimate, sources, stft_config, q: float = 0.9, coarse: bool = True):
    """ Masked loss in the spectral domain. """
    if estimate.dim() == 4:
        b, n, c, l = estimate.shape
        spec_estimate_flat = estimate.reshape(b * n * c, l)
        spec_sources_flat = sources.reshape(b * n * c, l)
    elif estimate.dim() == 3:
        b, c, l = estimate.shape
        spec_estimate_flat = estimate.reshape(b * c, l)
        spec_sources_flat = sources.reshape(b * c, l)
        n = 1
    else:
         raise ValueError(f"Unexpected shape in spec_masked_loss: {estimate.shape}")

    spec_estimate = torch.stft(spec_estimate_flat, **stft_config, return_complex=True)
    spec_sources = torch.stft(spec_sources_flat, **stft_config, return_complex=True)

    spec_estimate_mag = torch.abs(spec_estimate)
    spec_sources_mag = torch.abs(spec_sources)

    new_shape = (b, n * c) + spec_estimate_mag.shape[-2:]
    spec_estimate_mag = spec_estimate_mag.view(*new_shape)
    spec_sources_mag = spec_sources_mag.view(*new_shape)

    loss = F.mse_loss(spec_estimate_mag, spec_sources_mag, reduction='none')

    if coarse:
        loss = loss.mean(dim=(-2, -1))

    loss = loss.reshape(loss.shape[0], -1)

    if loss.numel() == 0: return torch.tensor(0.0, device=estimate.device)

    quantile = torch.quantile(loss.detach(), q, interpolation='linear', dim=1, keepdim=True)
    mask = loss < quantile
    masked_loss_val = (loss * mask).sum() / (mask.sum() + 1e-8)

    return masked_loss_val


def choice_loss(args: argparse.Namespace, config: ConfigDict) -> Union[Callable[..., torch.Tensor], Dict[str, Callable]]:
    """ Select and return the appropriate loss function(s). """
    is_gan = config.training.get('is_gan', False) and args.model_type == 'apollo'

    if is_gan:
        print("Using GAN losses for Apollo")
        try:
            loss_g = MultiFrequencyGenLoss(**config.get('loss_g', {}))
            loss_d = MultiFrequencyDisLoss(**config.get('loss_d', {}))
            return {'G': loss_g, 'D': loss_d}
        except NameError:
             raise ImportError("Could not find Apollo loss functions. Ensure they are imported correctly.")
    else:
        print(f'Using standard losses: {args.loss}')
        loss_fns = []
        # Use .get for safer access to nested config values
        audio_config = config.get('audio', ConfigDict({'chunk_size': 131072, 'sample_rate': 44100}))
        model_config = config.get('model', ConfigDict({}))
        training_config = config.get('training', ConfigDict({'q': 0.9, 'coarse_loss_clip': True}))

        loss_configs = {
            'masked_loss': (masked_loss, args.masked_loss_coef, {'q': training_config.get('q', 0.9), 'coarse': training_config.get('coarse_loss_clip', True)}),
            'mse_loss': (nn.MSELoss(), args.mse_loss_coef, {}),
            'l1_loss': (F.l1_loss, args.l1_loss_coef, {}),
            'multistft_loss': (multistft_loss, args.multistft_loss_coef, {'loss_multistft': auraloss.freq.MultiResolutionSTFTLoss(**config.get('loss_multistft', {}))}),
            'log_wmse_loss': (LogWMSE(audio_length=audio_config.get('chunk_size', 131072) / audio_config.get('sample_rate', 44100), sample_rate=audio_config.get('sample_rate', 44100), return_as_loss=True, bypass_filter=training_config.get('bypass_filter', False)), args.log_wmse_loss_coef, {}),
            'spec_masked_loss': (spec_masked_loss, args.spec_masked_loss_coef, {'stft_config': {'n_fft': model_config.get('nfft', 4096), 'hop_length': model_config.get('hop_size', 1024),'win_length': model_config.get('win_size', 4096), 'center': True, 'normalized': model_config.get('normalized', True)}, 'q': training_config.get('q', 0.9), 'coarse': training_config.get('coarse_loss_clip', True)})
        }

        for loss_name in args.loss:
            if loss_name == 'gan_loss': continue # Ignore placeholder for standard models
            if loss_name in loss_configs:
                func, coef, params = loss_configs[loss_name]
                loss_fns.append((func, coef, params))
            else:
                print(f"Warning: Unknown loss '{loss_name}' specified.")

        def multi_loss_standard(y_pred: Any, y_true: Any, x: Optional[Any] = None) -> torch.Tensor:
            total_loss = torch.tensor(0.0, device=y_pred.device)
            for func, coef, params in loss_fns:
                 try:
                     # Check if 'x' is an expected argument more robustly
                     takes_x = False
                     if hasattr(func, '__code__'): # For regular functions
                         takes_x = 'x' in func.__code__.co_varnames
                     elif hasattr(func, 'forward') and hasattr(func.forward, '__code__'): # For nn.Module
                         takes_x = 'x' in func.forward.__code__.co_varnames

                     if takes_x:
                          current_loss = func(y_pred, y_true, x=x, **params)
                     else:
                          current_loss = func(y_pred, y_true, **params)
                     total_loss += current_loss * coef
                 except Exception as e:
                      print(f"Error calculating loss {func.__class__.__name__ if isinstance(func, nn.Module) else func.__name__}: {e}")
                      # Decide how to handle error, e.g., skip this loss component
                      # total_loss += 0.0
            return total_loss

        return multi_loss_standard


def normalize_batch(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Normalize a batch based on the mean/std of x. """
    mean = x.mean(dim=[1, 2], keepdim=True)
    std = x.std(dim=[1, 2], keepdim=True) + 1e-8
    x_norm = (x - mean) / std
    y_norm = (y - mean) / std
    return x_norm, y_norm


def train_one_epoch(model: torch.nn.Module, config: ConfigDict, args: argparse.Namespace, optimizer: torch.optim.Optimizer,
                    device: torch.device, device_ids: List[int], epoch: int, use_amp: bool, scaler: torch.cuda.amp.GradScaler,
                    gradient_accumulation_steps: int, train_loader: torch.utils.data.DataLoader,
                    multi_loss: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]) -> None:
    """ Train the standard (non-GAN) model for one epoch. """
    model.train().to(device)
    print(f'Train epoch: {epoch} Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
    loss_val = 0.
    total_items = 0 # Count items instead of batches for accurate average
    grad_clip_val = config.training.get('grad_clip', 0)

    normalize = config.training.get('normalize', False)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    optimizer.zero_grad(set_to_none=True) # Zero grad at the beginning of the epoch Accumulation cycle
    for i, (batch, mixes) in enumerate(pbar):
        x = mixes.to(device)
        y = batch.to(device)
        batch_size = x.size(0)

        if normalize:
            x, y = normalize_batch(x, y)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if 'roformer' in args.model_type:
                loss = model(x, y)
                if isinstance(model, nn.DataParallel):
                    loss = loss.mean()
            else:
                y_ = model(x)
                loss = multi_loss(y_, y, x)

        loss_accum = loss / gradient_accumulation_steps
        scaler.scale(loss_accum).backward()

        if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
            scaler.unscale_(optimizer)
            if grad_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) # Zero grad after step

        li = loss.item()
        loss_val += li * batch_size # Accumulate based on item loss * batch size
        total_items += batch_size
        avg_loss_epoch = loss_val / total_items
        pbar.set_postfix({'loss': f"{li:.4f}", 'avg_loss': f"{avg_loss_epoch:.4f}"})
        wandb.log({'loss': li, 'avg_loss': avg_loss_epoch, 'step': i + epoch * len(train_loader)})
        loss.detach()

    print(f'Training loss: {avg_loss_epoch:.4f}')
    wandb.log({'train_loss_epoch': avg_loss_epoch, 'epoch': epoch, 'learning_rate': optimizer.param_groups[0]['lr']})


def train_gan_one_epoch(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    config: ConfigDict,
    args: argparse.Namespace,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    device_ids: List[int],
    epoch: int,
    use_amp: bool,
    scaler: torch.cuda.amp.GradScaler,
    gradient_accumulation_steps: int,
    train_loader: torch.utils.data.DataLoader,
    loss_funcs: Dict[str, Callable],
) -> None:
    """ Train the GAN (Generator + Discriminator) for one epoch. """
    generator.train().to(device)
    discriminator.train().to(device)
    print(f'Train GAN epoch: {epoch} LR G: {optimizer_g.param_groups[0]["lr"]:.6f} LR D: {optimizer_d.param_groups[0]["lr"]:.6f}')

    loss_g_val = 0.
    loss_d_val = 0.
    total_items = 0 # Count items

    loss_func_g = loss_funcs['G']
    loss_func_d = loss_funcs['D']
    grad_clip_val = config.training.get('grad_clip', 5.0)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} GAN Training")
    # Zero grads once before the loop for accumulation
    optimizer_g.zero_grad(set_to_none=True)
    optimizer_d.zero_grad(set_to_none=True)
    for i, (original_clean, degraded_audio) in enumerate(pbar):
        degraded_audio = degraded_audio.to(device)
        original_clean = original_clean.to(device)
        batch_size = degraded_audio.size(0)
        total_items += batch_size

        # Train Discriminator
        with torch.cuda.amp.autocast(enabled=use_amp):
            generated_audio = generator(degraded_audio).detach()
            target_outputs, _ = discriminator(original_clean)
            est_outputs, _ = discriminator(generated_audio)
            loss_d = loss_func_d(target_outputs, est_outputs)
            loss_d_accum = loss_d / gradient_accumulation_steps

        scaler.scale(loss_d_accum).backward()
        loss_d_item = loss_d.item() # Store before potential step
        loss_d_val += loss_d_item * batch_size

        # Train Generator
        with torch.cuda.amp.autocast(enabled=use_amp):
            generated_audio_for_g = generator(degraded_audio)
            est_outputs_for_g, est_feature_maps = discriminator(generated_audio_for_g)
            _, targets_feature_maps = discriminator(original_clean)
            loss_g = loss_func_g(est_outputs_for_g, est_feature_maps, targets_feature_maps, generated_audio_for_g, original_clean)
            loss_g_accum = loss_g / gradient_accumulation_steps

        scaler.scale(loss_g_accum).backward()
        loss_g_item = loss_g.item() # Store before potential step
        loss_g_val += loss_g_item * batch_size

        # Step optimizers and zero gradients
        if ((i + 1) % gradient_accumulation_steps == 0) or (i == len(train_loader) - 1):
            # Step D
            scaler.unscale_(optimizer_d)
            if grad_clip_val > 0: torch.nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip_val)
            scaler.step(optimizer_d)

            # Step G
            scaler.unscale_(optimizer_g)
            if grad_clip_val > 0: torch.nn.utils.clip_grad_norm_(generator.parameters(), grad_clip_val)
            scaler.step(optimizer_g)

            # Update scaler and zero grads
            scaler.update()
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

        avg_loss_g_epoch = loss_g_val / total_items
        avg_loss_d_epoch = loss_d_val / total_items
        pbar.set_postfix({
            'loss_g': f"{loss_g_item:.4f}", 'loss_d': f"{loss_d_item:.4f}",
            'avg_g': f"{avg_loss_g_epoch:.4f}", 'avg_d': f"{avg_loss_d_epoch:.4f}"
        })
        wandb.log({
            'loss_g': loss_g_item, 'loss_d': loss_d_item,
            'avg_loss_g': avg_loss_g_epoch, 'avg_loss_d': avg_loss_d_epoch,
            'step': i + epoch * len(train_loader)
        })
        loss_d.detach(); loss_g.detach()

    avg_loss_g_epoch = loss_g_val / total_items
    avg_loss_d_epoch = loss_d_val / total_items
    print(f'Training Epoch {epoch} Avg G Loss: {avg_loss_g_epoch:.4f}, Avg D Loss: {avg_loss_d_epoch:.4f}')
    wandb.log({
        'train_loss_g_epoch': avg_loss_g_epoch, 'train_loss_d_epoch': avg_loss_d_epoch,
        'epoch': epoch,
        'learning_rate_g': optimizer_g.param_groups[0]['lr'],
        'learning_rate_d': optimizer_d.param_groups[0]['lr']
    })


def save_weights(store_path, model_or_models, device_ids, train_lora, is_gan=False):
    """ Saves weights for standard or GAN models. """
    if is_gan:
        generator, discriminator = model_or_models
        state_dict_g = generator.state_dict() if not isinstance(generator, nn.DataParallel) else generator.module.state_dict()
        state_dict_d = discriminator.state_dict() if not isinstance(discriminator, nn.DataParallel) else discriminator.module.state_dict()

        if train_lora:
             generator_ref = generator.module if isinstance(generator, nn.DataParallel) else generator
             state_dict_g = lora.lora_state_dict(generator_ref)

        torch.save(
            {'generator_state_dict': state_dict_g,
             'discriminator_state_dict': state_dict_d},
            store_path
        )
    else:
        model = model_or_models
        model_ref = model.module if isinstance(model, nn.DataParallel) else model
        if train_lora:
            torch.save(lora.lora_state_dict(model_ref), store_path)
        else:
            torch.save(model_ref.state_dict(), store_path)


def save_last_weights(args: argparse.Namespace, model_or_models: Union[torch.nn.Module, Tuple[torch.nn.Module, torch.nn.Module]], device_ids: List[int], is_gan=False) -> None:
    """ Saves the last weights for standard or GAN models. """
    store_path = f'{args.results_path}/last_{args.model_type}.ckpt'
    train_lora = args.train_lora
    save_weights(store_path, model_or_models, device_ids, train_lora, is_gan)


def get_scheduler(optimizer, config_sched):
    """ Helper to create scheduler based on config """
    sched_name = config_sched.get('name', 'ReduceLROnPlateau')
    if sched_name == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer,
                                 mode=config_sched.get('mode', 'max'),
                                 patience=config_sched.get('patience', 10),
                                 factor=config_sched.get('factor', 0.5),
                                 verbose=True)
    elif sched_name == 'StepLR':
         return StepLR(optimizer,
                       step_size=config_sched.get('step_size', 2),
                       gamma=config_sched.get('gamma', 0.98))
    else:
        print(f"Warning: Scheduler '{sched_name}' not recognized. Using ReduceLROnPlateau.")
        return ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5, verbose=True)


def compute_epoch_metrics(model_to_validate: torch.nn.Module, args: argparse.Namespace, config: ConfigDict,
                          device: torch.device, device_ids: List[int], best_metric: float,
                          epoch: int, scheduler: Union[torch.optim.lr_scheduler._LRScheduler, Dict[str, torch.optim.lr_scheduler._LRScheduler]],
                          is_gan: bool = False, models_to_save = None) -> float:
    """ Compute validation metrics and step scheduler(s). """
    valid_args_copy = argparse.Namespace(**vars(args))
    valid_args_copy.store_dir = ""

    if torch.cuda.is_available() and len(device_ids) > 1:
        metrics_avg, all_metrics = valid_multi_gpu(model_to_validate, valid_args_copy, config, args.device_ids, verbose=False)
    else:
        metrics_avg, all_metrics = valid(model_to_validate, valid_args_copy, config, device, verbose=False)

    metric_key_for_scheduler = args.metric_for_scheduler
    current_metric_value = -float('inf')

    if metric_key_for_scheduler in metrics_avg:
        current_metric_value = metrics_avg[metric_key_for_scheduler]

    print(f"Epoch {epoch} Validation Metric ({metric_key_for_scheduler}): {current_metric_value:.4f}")

    if current_metric_value > -float('inf'):
        if current_metric_value > best_metric:
            store_path = f'{args.results_path}/model_{args.model_type}_ep_{epoch}_{metric_key_for_scheduler}_{current_metric_value:.4f}.ckpt'
            print(f'Saving best model to: {store_path}')
            if models_to_save is None: models_to_save = model_to_validate
            save_weights(store_path, models_to_save, device_ids, args.train_lora, is_gan)
            best_metric = current_metric_value

        if is_gan:
            sched_g = scheduler['G']
            if isinstance(sched_g, ReduceLROnPlateau): sched_g.step(current_metric_value)
            elif isinstance(sched_g, StepLR): sched_g.step()
            sched_d = scheduler['D']
            if isinstance(sched_d, ReduceLROnPlateau): sched_d.step(current_metric_value)
            elif isinstance(sched_d, StepLR): sched_d.step()
        else:
            if isinstance(scheduler, ReduceLROnPlateau): scheduler.step(current_metric_value)
            elif isinstance(scheduler, StepLR): scheduler.step()
    else:
        print(f"Warning: Metric '{metric_key_for_scheduler}' not found in validation results. Skipping scheduler step and best model check.")

    wandb.log({'val_metric_main': current_metric_value, 'best_val_metric': best_metric, 'epoch': epoch})
    if is_gan:
         # Log metrics under 'enhanced' key if they exist
         for metric_name, value in metrics_avg.items(): # Iterate directly over avg dict
              wandb.log({f'val_metric_{metric_name}': value}) # Log the avg value
    else:
         for metric_name, value in metrics_avg.items():
              wandb.log({f'val_metric_{metric_name}': value})

    return best_metric


def train_model(args: Optional[argparse.Namespace] = None) -> None:
    """ Main training function handling standard and GAN loops. """
    if args is None:
        args = parse_args(None)

    initialize_environment(args.seed, args.results_path)
    model_or_models, config = get_model_from_config(args.model_type, args.config_path)

    is_gan = config.training.get('is_gan', False) and args.model_type == 'apollo'

    if is_gan:
        generator, discriminator = model_or_models
        print("Instantiated Apollo Generator and Discriminator.")
    else:
        model = model_or_models
        print("Instantiated standard model.")

    use_amp = config.training.get('use_amp', True)
    device_ids = args.device_ids
    batch_size = config.training.batch_size * max(1, len(device_ids))

    wandb_init(args, config, device_ids, batch_size)

    current_dataset_type = args.dataset_type
    print(f"Using Dataset Type: {current_dataset_type}") # Added print for confirmation
    train_loader = prepare_data(config, args, batch_size, dataset_type_override=current_dataset_type)

    if args.start_check_point:
        load_start_checkpoint(args, model_or_models, type_='train_gan' if is_gan else 'train')

    if args.train_lora:
        if is_gan:
            print("Applying LoRA to Generator.")
            generator = bind_lora_to_model(config, generator)
            lora.mark_only_lora_as_trainable(generator)
            model_or_models = (generator, discriminator) # Update tuple
        else:
            model = bind_lora_to_model(config, model)
            lora.mark_only_lora_as_trainable(model)
            model_or_models = model # Update single model

    if is_gan:
        device, generator = initialize_model_and_device(generator, args.device_ids)
        _, discriminator = initialize_model_and_device(discriminator, args.device_ids)
        model_or_models = (generator, discriminator)
    else:
        device, model = initialize_model_and_device(model, args.device_ids)
        model_or_models = model

    if args.pre_valid:
        print("Running pre-validation...")
        model_to_validate = generator if is_gan else model
        valid_args_copy = argparse.Namespace(**vars(args))
        valid_args_copy.store_dir = ""
        if torch.cuda.is_available() and len(device_ids) > 1:
            valid_multi_gpu(model_to_validate, valid_args_copy, config, args.device_ids, verbose=True)
        else:
            valid(model_to_validate, valid_args_copy, config, device, verbose=True)

    if is_gan:
        optimizers = get_optimizer(config, model_or_models)
        optimizer_g = optimizers['G']
        optimizer_d = optimizers['D']
        scheduler_g = get_scheduler(optimizer_g, config.get('scheduler_g', {}))
        scheduler_d = get_scheduler(optimizer_d, config.get('scheduler_d', {}))
        schedulers = {'G': scheduler_g, 'D': scheduler_d}
    else:
        optimizer = get_optimizer(config, model_or_models)
        scheduler_config = config.get('scheduler', ConfigDict({'name': 'ReduceLROnPlateau', 'mode': 'max', 'patience': config.training.get('patience', 10), 'factor': config.training.get('reduce_factor', 0.5)}))
        scheduler = get_scheduler(optimizer, scheduler_config)

    gradient_accumulation_steps = int(config.training.get('gradient_accumulation_steps', 1))
    loss_functions = choice_loss(args, config)
    scaler = GradScaler(enabled=use_amp)
    best_metric = -float('inf')

    print(f"Training Mode: {'GAN (Apollo)' if is_gan else 'Standard'}")
    print(f"Batch size: {batch_size}, Grad accum steps: {gradient_accumulation_steps}, Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f'Train for: {config.training.num_epochs} epochs')

    for epoch in range(config.training.num_epochs):
        if is_gan:
            train_gan_one_epoch(
                generator, discriminator, config, args,
                optimizer_g, optimizer_d, device, device_ids, epoch,
                use_amp, scaler, gradient_accumulation_steps, train_loader, loss_functions
            )
            save_last_weights(args, model_or_models, device_ids, is_gan=True)
            best_metric = compute_epoch_metrics(generator, args, config, device, device_ids, best_metric, epoch, schedulers, is_gan=True, models_to_save=model_or_models)
        else:
            train_one_epoch(
                model, config, args, optimizer, device, device_ids, epoch,
                use_amp, scaler, gradient_accumulation_steps, train_loader, loss_functions
            )
            save_last_weights(args, model_or_models, device_ids, is_gan=False)
            best_metric = compute_epoch_metrics(model, args, config, device, device_ids, best_metric, epoch, scheduler, is_gan=False, models_to_save=model_or_models)


if __name__ == "__main__":
    train_model(None)
