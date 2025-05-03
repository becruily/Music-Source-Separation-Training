# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'


import os
import random
import numpy as np
import torch
import soundfile as sf
import pickle
import time
import itertools
import multiprocessing
from tqdm.auto import tqdm
from glob import glob as gf # Import glob with an alias to avoid shadowing
import audiomentations as AU
import pedalboard as PB
import warnings
import torchaudio
from torchaudio.functional import apply_codec
warnings.filterwarnings("ignore")


def load_chunk(path, length, chunk_size, offset=None):
    try:
        if chunk_size <= length:
            if offset is None:
                offset = np.random.randint(length - chunk_size + 1)
            x = sf.read(path, dtype='float32', start=offset, frames=chunk_size)[0]
        else:
            x = sf.read(path, dtype='float32')[0]
            pad_len = chunk_size - length
            if len(x.shape) == 1:
                pad = np.zeros(pad_len, dtype=np.float32)
            else:
                pad = np.zeros([pad_len, x.shape[-1]], dtype=np.float32)
            x = np.concatenate([x, pad], axis=0)

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)
        return x.T # Transpose to [Channels, Time]
    except Exception as e:
        print(f"Error loading chunk from {path}: {e}")
        return np.zeros((2, chunk_size), dtype=np.float32)


def get_track_set_length(params):
    path, instruments, file_types = params
    lengths_arr = []
    found_any = False
    for instr in instruments:
        length = -1
        for extension in file_types:
            path_to_audio_file = os.path.join(path, f'{instr}.{extension}')
            if os.path.isfile(path_to_audio_file):
                try:
                    info = sf.info(path_to_audio_file)
                    length = info.frames
                    found_any = True
                    break
                except Exception as e:
                    # print(f'Warning: Could not read info for {path_to_audio_file}: {e}')
                    length = -1
                    break
        if length != -1:
            lengths_arr.append(length)

    if not found_any:
        return path, 0

    lengths_arr = np.array([l for l in lengths_arr if l != -1])
    if len(lengths_arr) == 0:
         return path, 0

    min_len = lengths_arr.min()
    max_len = lengths_arr.max()
    if min_len != max_len:
        print(f'Warning: Stem lengths differ in {path}. Min: {min_len}, Max: {max_len}. Using min.')

    return path, min_len


def get_track_length(params):
    path = params
    try:
        info = sf.info(path)
        length = info.frames
        return (path, length)
    except Exception as e:
        # print(f"Error getting length for {path}: {e}")
        return (path, None)


def match2(x, d):
    assert x.dim()==2, x.shape
    assert d.dim()==2, d.shape
    minlen = min(x.shape[-1], d.shape[-1])
    x, d = x[:,0:minlen], d[:,0:minlen]
    Fx = torch.fft.rfft(x, dim=-1)
    Fd = torch.fft.rfft(d, dim=-1)
    Phi = Fd*Fx.conj()
    Phi = Phi / (Phi.abs() + 1e-3)
    Phi[:,0] = 0
    tmp = torch.fft.irfft(Phi, dim=-1)
    tau = torch.argmax(tmp.abs(),dim=-1).tolist()
    return tau

def codec_simu(wav, sr=44100, options={'bitrate':'random', 'complexity':'random', 'vbr':'random'}):
    codec_options = options.to_dict().copy() if hasattr(options, 'to_dict') else options.copy()
    target_bitrate = codec_options.get('bitrate', 'random')

    if target_bitrate == 'random':
        target_bitrate_kbps = random.choice([24, 32, 48, 64, 96, 128])
    else:
        target_bitrate_kbps = int(target_bitrate // 1000)
        target_bitrate_kbps = max(8, min(320, target_bitrate_kbps))

    if target_bitrate_kbps >= 256: vbr_quality_setting = 0.0
    elif target_bitrate_kbps >= 192: vbr_quality_setting = 1.0
    elif target_bitrate_kbps >= 160: vbr_quality_setting = 2.0
    elif target_bitrate_kbps >= 128: vbr_quality_setting = 3.0
    elif target_bitrate_kbps >= 112: vbr_quality_setting = 4.0
    elif target_bitrate_kbps >= 96: vbr_quality_setting = 5.0
    elif target_bitrate_kbps >= 80: vbr_quality_setting = 6.0
    elif target_bitrate_kbps >= 64: vbr_quality_setting = 7.0
    elif target_bitrate_kbps >= 48: vbr_quality_setting = 8.0
    else: vbr_quality_setting = 9.0

    try:
        board = PB.Pedalboard([
            PB.MP3Compressor(vbr_quality=float(vbr_quality_setting))
        ])

        original_device = wav.device
        original_dtype = wav.dtype
        wav_numpy = wav.cpu().numpy()
        wav_processed_numpy = board(wav_numpy, sr)
        wav_encdec = torch.from_numpy(wav_processed_numpy).to(original_device).to(original_dtype)

        if wav_encdec.shape[-1] != wav.shape[-1]:
            if wav_encdec.shape[-1] > wav.shape[-1]:
                 wav_encdec = wav_encdec[..., :wav.shape[-1]]
            else:
                 padding = torch.zeros(wav.shape[:-1] + (wav.shape[-1] - wav_encdec.shape[-1],), dtype=original_dtype, device=original_device)
                 wav_encdec = torch.cat([wav_encdec, padding], dim=-1)

    except Exception as e:
        print(f"Pedalboard MP3 simulation failed: {e}. Returning original.")
        wav_encdec = wav.clone()

    return wav_encdec

def find_enhancement_pairs(params):
    folder_path, _, _, file_types = params # Suffixes ignored for fixed name logic
    pairs = []
    dirty_names = [f"dirty.{ext}" for ext in file_types]
    clean_names = [f"clean.{ext}" for ext in file_types]

    dirty_path_found = None
    clean_path_found = None

    try:
        for d_name in dirty_names:
            potential_path = os.path.join(folder_path, d_name)
            if os.path.isfile(potential_path):
                dirty_path_found = potential_path
                break

        for c_name in clean_names:
            potential_path = os.path.join(folder_path, c_name)
            if os.path.isfile(potential_path):
                clean_path_found = potential_path
                break

        if dirty_path_found and clean_path_found:
            try:
                info = sf.info(clean_path_found)
                length = info.frames
                if length > 0:
                    pairs.append((dirty_path_found, clean_path_found, length))
            except Exception as e:
                 print(f"Error getting info for clean file {clean_path_found}: {e}")

    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
    return pairs


class MSSDataset(torch.utils.data.Dataset):
    def __init__(self, config, data_path, metadata_path="metadata.pkl", dataset_type=1, batch_size=None, verbose=True):
        self.verbose = verbose
        self.config = config
        self.dataset_type = dataset_type # 1, 2, 3, 4, 5 or 6
        self.data_path = data_path
        self.instruments = config.training.get('instruments', [])
        if batch_size is None:
            batch_size = config.training.batch_size
        self.batch_size = batch_size
        self.file_types = ['wav', 'flac']
        self.metadata_path = metadata_path

        self.aug = False
        if 'augmentations' in config:
            if config.augmentations.get('enable', False):
                if self.verbose:
                    print('Use augmentation for training')
                self.aug = True
        else:
            if self.verbose:
                print('Augmentations disabled for training.')

        self.codec_options = config.get('datas', {}).get('codec_options', {})
        self.sr = config.get('datas', {}).get('sr', 44100)
        self.enhancement_dirty_suffixes = config.get('datas', {}).get('enhancement_dirty_suffixes', ['_dirty', '_input', '_degraded'])
        self.enhancement_clean_suffixes = config.get('datas', {}).get('enhancement_clean_suffixes', ['_clean', '_orig', '_target', ''])

        metadata = self.get_metadata()

        if self.dataset_type in [1, 4, 5, 6]:
            if metadata and len(metadata) > 0:
                if self.verbose:
                    print(f'Found {len(metadata)} items for dataset type {self.dataset_type}')
            else:
                print(f'Warning: No tracks/pairs found for dataset type {self.dataset_type}. Check paths and naming conventions!')
        else: # Types 2, 3
            for instr in self.instruments:
                 if instr in metadata and len(metadata[instr]) > 0:
                      if self.verbose:
                           print(f'Found {len(metadata[instr])} tracks for {instr} in dataset')
                 else:
                      print(f'Warning: No tracks found for instrument {instr}')

        self.metadata = metadata
        self.chunk_size = config.get('audio', {}).get('chunk_size', 131072)
        self.min_mean_abs = config.get('audio', {}).get('min_mean_abs', 0.001)

    def __len__(self):
        num_steps = self.config.training.get('num_steps', 1000)
        return num_steps * self.batch_size

    def read_from_metadata_cache(self, input_items, instr=None):
        is_list_type = self.dataset_type in [1, 4, 5, 6]
        cached_metadata = [] if is_list_type else {}
        items_to_process = input_items

        if not os.path.isfile(self.metadata_path):
             return items_to_process, cached_metadata

        if self.verbose:
             print('Found metadata cache file: {}'.format(self.metadata_path))
        try:
             loaded_cache = pickle.load(open(self.metadata_path, 'rb'))
        except Exception as e:
             print(f"Error loading cache {self.metadata_path}: {e}. Rebuilding cache.")
             return items_to_process, cached_metadata

        try:
            cached_item_identifiers = set()
            if is_list_type:
                if not isinstance(loaded_cache, list): raise TypeError(f"Expected list cache for type {self.dataset_type}")
                cached_metadata = loaded_cache # Use loaded cache as base
                if self.dataset_type == 6:
                    cached_item_identifiers = {os.path.dirname(item_data[0]) for item_data in loaded_cache if len(item_data) == 3}
                elif self.dataset_type in [1, 4, 5]:
                    cached_item_identifiers = {item_data[0] for item_data in loaded_cache if len(item_data) == 2}
                items_to_process = [item for item in input_items if item not in cached_item_identifiers]

            else: # Types 2, 3
                if not isinstance(loaded_cache, dict): raise TypeError(f"Expected dict cache for type {self.dataset_type}")
                cached_metadata = loaded_cache # Use loaded cache as base
                if instr is not None and instr in loaded_cache:
                     cached_list_for_instr = loaded_cache[instr]
                     if not isinstance(cached_list_for_instr, list): raise TypeError(f"Expected list for cache key '{instr}'")
                     cached_item_identifiers = {item_data[0] for item_data in cached_list_for_instr if len(item_data) == 2}
                items_to_process = [p for p in input_items if p not in cached_item_identifiers]

            if not items_to_process and cached_item_identifiers:
                 cache_info = f" for instrument '{instr}'" if instr else ""
                 print(f"Metadata cache is up-to-date{cache_info}.")

        except (TypeError, KeyError, IndexError, ValueError) as e:
            print(f"Cache structure error or mismatch for type {self.dataset_type}: {e}. Rebuilding cache.")
            items_to_process = input_items
            cached_metadata = [] if is_list_type else {}

        return items_to_process, cached_metadata


    def get_metadata(self):
        read_metadata_procs = self.config.training.get('read_metadata_procs', multiprocessing.cpu_count())

        if self.verbose:
             print(f'Dataset type: {self.dataset_type}, Processes: {read_metadata_procs}')
             print(f'Collecting metadata for: {self.data_path}')

        initial_items = []
        data_paths_list = self.data_path if isinstance(self.data_path, list) else [self.data_path]

        if self.dataset_type in [1, 4, 6]:
            for tp in data_paths_list:
                 try:
                      folders = sorted(gf(os.path.join(tp, '*'))) # Use alias gf
                      initial_items.extend([p for p in folders if os.path.isdir(p) and os.path.basename(p)[0] != '.'])
                 except Exception as e: print(f"Error accessing data path {tp}: {e}")
            initial_items = sorted(list(set(initial_items)))

        elif self.dataset_type == 5:
            for tp in data_paths_list:
                 try:
                      for ext in self.file_types: initial_items.extend(sorted(gf(os.path.join(tp, f'*.{ext}')))) # Use alias gf
                 except Exception as e: print(f"Error accessing data path {tp}: {e}")
            initial_items = [p for p in initial_items if os.path.basename(p)[0] != '.']
            initial_items = sorted(list(set(initial_items)))

        elif self.dataset_type in [2, 3]:
             pass # Handled within instrument loop
        else:
             print(f'Unknown dataset type: {self.dataset_type}.')
             exit()

        if self.dataset_type in [1, 4, 5, 6]:
            items_to_process, cached_metadata = self.read_from_metadata_cache(initial_items, None)
            metadata = cached_metadata

            if items_to_process:
                 print(f"Calculating metadata for {len(items_to_process)} new items (Type {self.dataset_type})...")
                 new_metadata = []
                 pool_func = None
                 pool_params = None

                 if self.dataset_type in [1, 4]:
                      pool_func = get_track_set_length
                      pool_params = zip(items_to_process, itertools.repeat(self.instruments), itertools.repeat(self.file_types))
                 elif self.dataset_type == 5:
                      pool_func = get_track_length
                      pool_params = items_to_process
                 elif self.dataset_type == 6:
                      pool_func = find_enhancement_pairs
                      pool_params = zip(items_to_process, itertools.repeat(self.enhancement_dirty_suffixes), itertools.repeat(self.enhancement_clean_suffixes), itertools.repeat(self.file_types))

                 if read_metadata_procs <= 1:
                      for params_item in tqdm(pool_params, total=len(items_to_process)):
                           result = pool_func(params_item)
                           if result:
                                if self.dataset_type == 6: new_metadata.extend(result)
                                else: new_metadata.append(result)
                 else:
                      p = multiprocessing.Pool(processes=read_metadata_procs)
                      with tqdm(total=len(items_to_process)) as pbar:
                           for result in p.imap_unordered(pool_func, pool_params):
                               if result:
                                    if self.dataset_type == 6: new_metadata.extend(result)
                                    else: new_metadata.append(result)
                               pbar.update()
                      p.close(); p.join()

                 if self.dataset_type in [1, 4, 5]:
                      new_metadata = [m for m in new_metadata if len(m)==2 and m[1] is not None and m[1] > 0]
                 elif self.dataset_type == 6:
                      new_metadata = [m for m in new_metadata if len(m)==3 and m[2] is not None and m[2] > 0]

                 metadata.extend(new_metadata)
                 metadata = sorted(list({item[0]: item for item in metadata}.values()))

        elif self.dataset_type in [2, 3]:
             metadata = {}
             _, cached_metadata_dict = self.read_from_metadata_cache([], None)

             for instr in self.instruments:
                 track_paths_instr = []
                 if self.dataset_type == 2:
                     for tp in data_paths_list:
                         instr_path = os.path.join(tp, instr)
                         if os.path.isdir(instr_path):
                             try:
                                 for ext in self.file_types: track_paths_instr.extend(sorted(gf(os.path.join(instr_path, f'*.{ext}')))) # Use alias gf
                             except Exception as e: print(f"Error accessing data path {instr_path}: {e}")
                 elif self.dataset_type == 3:
                     import pandas as pd
                     all_dfs = []
                     for dp in data_paths_list:
                         try: all_dfs.append(pd.read_csv(dp))
                         except Exception as e: print(f"Error reading CSV {dp}: {e}")
                     if all_dfs:
                         df = pd.concat(all_dfs, ignore_index=True)
                         part = df[df['instrum'] == instr].copy()
                         track_paths_instr = list(part['path'].values)

                 track_paths_instr = [p for p in track_paths_instr if os.path.basename(p)[0] != '.']
                 track_paths_instr = sorted(list(set(track_paths_instr)))

                 items_to_process, _ = self.read_from_metadata_cache(track_paths_instr, instr) # Use the loaded full cache
                 metadata[instr] = cached_metadata_dict.get(instr, [])

                 if items_to_process:
                     print(f"Calculating metadata for {len(items_to_process)} new files ({instr}, Type {self.dataset_type})...")
                     new_metadata_instr = []
                     skipped_count = 0
                     if read_metadata_procs <= 1:
                         for path in tqdm(items_to_process):
                              if not os.path.isfile(path): skipped_count += 1; continue
                              _, track_length = get_track_length(path)
                              if track_length is not None: new_metadata_instr.append((path, track_length))
                              else: skipped_count += 1
                     else:
                         p = multiprocessing.Pool(processes=read_metadata_procs)
                         with tqdm(total=len(items_to_process)) as pbar:
                              for result in p.imap_unordered(get_track_length, items_to_process):
                                  if result and result[1] is not None: new_metadata_instr.append(result)
                                  else: skipped_count += 1
                                  pbar.update()
                         p.close(); p.join()

                     if skipped_count > 0 and self.dataset_type == 3: print(f"Skipped {skipped_count} files for {instr} (not found or error).")
                     metadata[instr].extend(new_metadata_instr)
                     metadata[instr] = sorted(list({item[0]: item for item in metadata[instr]}.values()))

        else:
             print(f'Unknown dataset type: {self.dataset_type}.')
             exit()

        try:
            if isinstance(metadata, list):
                expected_len = 3 if self.dataset_type == 6 else 2
                metadata = [m for m in metadata if len(m) == expected_len and m[-1] is not None and m[-1] > 0]
            elif isinstance(metadata, dict):
                for instr in list(metadata.keys()):
                    metadata[instr] = [m for m in metadata[instr] if len(m) == 2 and m[1] is not None and m[1] > 0]
                    if not metadata[instr]: del metadata[instr]

            pickle.dump(metadata, open(self.metadata_path, 'wb'))
            if self.verbose: print(f"Metadata saved to {self.metadata_path}")
        except Exception as e:
            print(f"Error saving metadata cache: {e}")

        is_empty = False
        if isinstance(metadata, list): is_empty = not metadata
        elif isinstance(metadata, dict): is_empty = not metadata or all(not v for v in metadata.values())
        if is_empty:
             print(f"FATAL: Metadata is empty after processing for dataset type {self.dataset_type}. Cannot proceed.")
             exit()

        return metadata


    def load_source(self, metadata, instr):
        while True:
            source = None
            if self.dataset_type in [1, 4]:
                if not metadata: return torch.zeros((2, self.chunk_size), dtype=torch.float32)
                track_path, track_length = random.choice(metadata)
                path_to_audio_file = None
                for extension in self.file_types:
                    potential_path = os.path.join(track_path, f'{instr}.{extension}')
                    if os.path.isfile(potential_path):
                        path_to_audio_file = potential_path
                        break
                if path_to_audio_file:
                    source = load_chunk(path_to_audio_file, track_length, self.chunk_size)
                else:
                    continue

            elif self.dataset_type in [2, 3]:
                if instr not in metadata or not metadata[instr]:
                     return torch.zeros((2, self.chunk_size), dtype=torch.float32)
                track_path, track_length = random.choice(metadata[instr])
                source = load_chunk(track_path, track_length, self.chunk_size)

            if source is not None and source.shape[-1] == self.chunk_size and np.abs(source).mean() >= self.min_mean_abs:
                break
            elif source is None: pass
            elif source is not None and source.shape[-1] != self.chunk_size: continue

        source_tensor = torch.tensor(source, dtype=torch.float32)
        if self.aug:
            source_np_aug = self.augm_data(source, instr)
            source_tensor = torch.tensor(source_np_aug, dtype=torch.float32)
        return source_tensor

    def load_random_mix(self):
        res = []
        for instr in self.instruments:
            s1 = self.load_source(self.metadata, instr)
            augs_config = self.config.get('augmentations', {})
            if self.aug and 'mixup' in augs_config and augs_config.get('mixup', False):
                 mixup = [s1]
                 mixup_probs = augs_config.get('mixup_probs', [0.3, 0.1])
                 loudness_min = augs_config.get('loudness_min', 0.5)
                 loudness_max = augs_config.get('loudness_max', 1.5)
                 for prob in mixup_probs:
                     if random.uniform(0, 1) < prob:
                         s2 = self.load_source(self.metadata, instr)
                         mixup.append(s2)
                 if len(mixup) > 1:
                      mixup_tensor = torch.stack(mixup, dim=0)
                      loud_values = np.random.uniform(low=loudness_min, high=loudness_max, size=(len(mixup),))
                      loud_values = torch.tensor(loud_values, dtype=torch.float32)
                      mixup_tensor *= loud_values[:, None, None]
                      s1 = mixup_tensor.mean(dim=0, dtype=torch.float32)
            res.append(s1)
        res = torch.stack(res)
        return res

    def load_aligned_data(self):
        attempts = 10
        while attempts:
            if not self.metadata: return torch.zeros((len(self.instruments), 2, self.chunk_size), dtype=torch.float32)
            track_path, track_length = random.choice(self.metadata)
            common_offset = None
            if track_length >= self.chunk_size:
                common_offset = np.random.randint(track_length - self.chunk_size + 1)

            res = []
            silent_chunks = 0
            valid_chunk_loaded = True
            for i in self.instruments:
                 source = None
                 path_to_audio_file = None
                 for extension in self.file_types:
                     potential_path = os.path.join(track_path, f'{i}.{extension}')
                     if os.path.isfile(potential_path):
                         path_to_audio_file = potential_path
                         break
                 if path_to_audio_file:
                     source = load_chunk(path_to_audio_file, track_length, self.chunk_size, offset=common_offset)
                     if source.shape[-1] != self.chunk_size:
                          source = np.zeros((2, self.chunk_size), dtype=np.float32)
                          valid_chunk_loaded = False
                 else:
                      source = np.zeros((2, self.chunk_size), dtype=np.float32)
                      valid_chunk_loaded = False

                 res.append(source)
                 if np.abs(source).mean() < self.min_mean_abs:
                     silent_chunks += 1

            if valid_chunk_loaded and silent_chunks == 0:
                break

            attempts -= 1
            if attempts <= 0:
                if not valid_chunk_loaded:
                     res = [np.zeros((2, self.chunk_size), dtype=np.float32) for _ in self.instruments]
                break
            if common_offset is None and track_length < self.chunk_size:
                 if not valid_chunk_loaded:
                      res = [np.zeros((2, self.chunk_size), dtype=np.float32) for _ in self.instruments]
                 break

        res_np = np.stack(res, axis=0)
        if self.aug:
            res_np_aug = np.copy(res_np)
            for i, instr in enumerate(self.instruments):
                 res_np_aug[i] = self.augm_data(res_np[i], instr)
            res_tensor = torch.tensor(res_np_aug, dtype=torch.float32)
        else:
            res_tensor = torch.tensor(res_np, dtype=torch.float32)
        return res_tensor

    def augm_data(self, source, instr):
        source_shape = source.shape
        source_out = source.copy()
        applied_augs = []

        augs_config = self.config.get('augmentations', {})
        all_augs = augs_config.get('all', {})
        instr_augs = augs_config.get(instr, {})
        augs = {**all_augs, **instr_augs}

        if augs.get('channel_shuffle', 0) > 0:
             if random.uniform(0, 1) < augs['channel_shuffle']:
                 source_out = source_out[::-1].copy(); applied_augs.append('channel_shuffle')
        if augs.get('random_inverse', 0) > 0:
             if random.uniform(0, 1) < augs['random_inverse']:
                 source_out = source_out[:, ::-1].copy(); applied_augs.append('random_inverse')
        if augs.get('random_polarity', 0) > 0:
             if random.uniform(0, 1) < augs['random_polarity']:
                 source_out = -source_out; applied_augs.append('random_polarity')

        augmenter = AU.Compose([])
        if augs.get('pitch_shift', 0) > 0:
             augmenter.add_transform(AU.PitchShift(min_semitones=augs.get('pitch_shift_min_semitones', -2), max_semitones=augs.get('pitch_shift_max_semitones', 2), p=augs['pitch_shift'])); applied_augs.append('pitch_shift')
        if augs.get('seven_band_parametric_eq', 0) > 0:
             augmenter.add_transform(AU.SevenBandParametricEQ(min_gain_db=augs.get('seven_band_parametric_eq_min_gain_db', -6), max_gain_db=augs.get('seven_band_parametric_eq_max_gain_db', 6), p=augs['seven_band_parametric_eq'])); applied_augs.append('seven_band_parametric_eq')
        if augs.get('tanh_distortion', 0) > 0:
             augmenter.add_transform(AU.TanhDistortion(min_distortion=augs.get('tanh_distortion_min', 0.01), max_distortion=augs.get('tanh_distortion_max', 0.5), p=augs['tanh_distortion'])); applied_augs.append('tanh_distortion')
        if augs.get('mp3_compression', 0) > 0:
             augmenter.add_transform(AU.Mp3Compression(min_bitrate=augs.get('mp3_compression_min_bitrate', 32), max_bitrate=augs.get('mp3_compression_max_bitrate', 128), backend=augs.get('mp3_compression_backend', 'lameenc'), p=augs['mp3_compression'])); applied_augs.append('mp3_compression')
        if augs.get('gaussian_noise', 0) > 0:
             augmenter.add_transform(AU.AddGaussianNoise(min_amplitude=augs.get('gaussian_noise_min_amplitude', 0.001), max_amplitude=augs.get('gaussian_noise_max_amplitude', 0.015), p=augs['gaussian_noise'])); applied_augs.append('gaussian_noise')
        if augs.get('time_stretch', 0) > 0:
             augmenter.add_transform(AU.TimeStretch(min_rate=augs.get('time_stretch_min_rate', 0.8), max_rate=augs.get('time_stretch_max_rate', 1.25), leave_length_unchanged=True, p=augs['time_stretch'])); applied_augs.append('time_stretch')

        if len(augmenter.transforms) > 0:
             source_out = augmenter(samples=source_out, sample_rate=self.sr)
             if source_out.shape != source_shape: source_out = source_out[..., :source_shape[-1]]

        board = PB.Pedalboard([])
        if augs.get('pedalboard_reverb', 0) > 0:
             if random.uniform(0, 1) < augs['pedalboard_reverb']:
                 board.append(PB.Reverb(room_size=random.uniform(augs.get('pedalboard_reverb_room_size_min', 0.1), augs.get('pedalboard_reverb_room_size_max', 0.9)), damping=random.uniform(augs.get('pedalboard_reverb_damping_min', 0.1), augs.get('pedalboard_reverb_damping_max', 0.9)), wet_level=random.uniform(augs.get('pedalboard_reverb_wet_level_min', 0.1), augs.get('pedalboard_reverb_wet_level_max', 0.5)), dry_level=random.uniform(augs.get('pedalboard_reverb_dry_level_min', 0.5), augs.get('pedalboard_reverb_dry_level_max', 0.9)), width=random.uniform(augs.get('pedalboard_reverb_width_min', 0.5), augs.get('pedalboard_reverb_width_max', 1.0)))); applied_augs.append('pedalboard_reverb')
        if augs.get('pedalboard_chorus', 0) > 0:
             if random.uniform(0, 1) < augs['pedalboard_chorus']:
                 board.append(PB.Chorus(rate_hz=random.uniform(augs.get('pedalboard_chorus_rate_hz_min', 0.5), augs.get('pedalboard_chorus_rate_hz_max', 2.0)), depth=random.uniform(augs.get('pedalboard_chorus_depth_min', 0.1), augs.get('pedalboard_chorus_depth_max', 0.9)), centre_delay_ms=random.uniform(augs.get('pedalboard_chorus_centre_delay_ms_min', 5), augs.get('pedalboard_chorus_centre_delay_ms_max', 20)), feedback=random.uniform(augs.get('pedalboard_chorus_feedback_min', 0.0), augs.get('pedalboard_chorus_feedback_max', 0.5)), mix=random.uniform(augs.get('pedalboard_chorus_mix_min', 0.1), augs.get('pedalboard_chorus_mix_max', 0.7)))); applied_augs.append('pedalboard_chorus')
        if augs.get('pedalboard_phazer', 0) > 0:
             if random.uniform(0, 1) < augs['pedalboard_phazer']:
                 board.append(PB.Phaser(rate_hz=random.uniform(augs.get('pedalboard_phazer_rate_hz_min', 0.5), augs.get('pedalboard_phazer_rate_hz_max', 5.0)), depth=random.uniform(augs.get('pedalboard_phazer_depth_min', 0.1), augs.get('pedalboard_phazer_depth_max', 0.9)), centre_frequency_hz=random.uniform(augs.get('pedalboard_phazer_centre_frequency_hz_min', 500), augs.get('pedalboard_phazer_centre_frequency_hz_max', 1500)), feedback=random.uniform(augs.get('pedalboard_phazer_feedback_min', 0.0), augs.get('pedalboard_phazer_feedback_max', 0.6)), mix=random.uniform(augs.get('pedalboard_phazer_mix_min', 0.1), augs.get('pedalboard_phazer_mix_max', 0.7)))); applied_augs.append('pedalboard_phazer')
        if augs.get('pedalboard_distortion', 0) > 0:
             if random.uniform(0, 1) < augs['pedalboard_distortion']:
                 board.append(PB.Distortion(drive_db=random.uniform(augs.get('pedalboard_distortion_drive_db_min', 5), augs.get('pedalboard_distortion_drive_db_max', 25)))); applied_augs.append('pedalboard_distortion')
        if augs.get('pedalboard_pitch_shift', 0) > 0:
             if random.uniform(0, 1) < augs['pedalboard_pitch_shift']:
                 board.append(PB.PitchShift(semitones=random.uniform(augs.get('pedalboard_pitch_shift_semitones_min', -2), augs.get('pedalboard_pitch_shift_semitones_max', 2)))); applied_augs.append('pedalboard_pitch_shift')
        if augs.get('pedalboard_resample', 0) > 0:
             if random.uniform(0, 1) < augs['pedalboard_resample']:
                 board.append(PB.Resample(target_sample_rate=random.uniform(augs.get('pedalboard_resample_target_sample_rate_min', 8000), augs.get('pedalboard_resample_target_sample_rate_max', 40000)))); applied_augs.append('pedalboard_resample')
        if augs.get('pedalboard_bitcrash', 0) > 0:
             if random.uniform(0, 1) < augs['pedalboard_bitcrash']:
                 board.append(PB.Bitcrush(bit_depth=random.uniform(augs.get('pedalboard_bitcrash_bit_depth_min', 4), augs.get('pedalboard_bitcrash_bit_depth_max', 12)))); applied_augs.append('pedalboard_bitcrush')
        if augs.get('pedalboard_mp3_compressor', 0) > 0:
             if random.uniform(0, 1) < augs['pedalboard_mp3_compressor']:
                 board.append(PB.MP3Compressor(vbr_quality=random.uniform(augs.get('pedalboard_mp3_compressor_min', 2), augs.get('pedalboard_mp3_compressor_max', 8)))); applied_augs.append('pedalboard_mp3_compressor')

        if len(board.plugins) > 0:
             try:
                 source_out = board(source_out.astype(np.float32), self.sr)
                 if source_out.shape != source_shape: source_out = source_out[..., :source_shape[-1]]
             except Exception as pb_e:
                  print(f"Pedalboard augmentation failed: {pb_e}. Skipping.")

        return source_out


    def __getitem__(self, index):
        if self.dataset_type in [1, 2, 3]:
            target = self.load_random_mix()
            mix = target.sum(0)
        elif self.dataset_type == 4:
            target = self.load_aligned_data()
            mix = target.sum(0)
        elif self.dataset_type == 5:
            while True:
                if not self.metadata:
                     print("Error: Enhancement dataset (Type 5) metadata is empty!")
                     return torch.zeros((2, self.chunk_size), dtype=torch.float32), torch.zeros((2, self.chunk_size), dtype=torch.float32)
                track_path, track_length = random.choice(self.metadata)
                try:
                    original_clean_np = load_chunk(track_path, track_length, self.chunk_size)
                    if original_clean_np.shape[-1] != self.chunk_size: continue
                    if original_clean_np.shape[0] == 1: original_clean_np = np.concatenate([original_clean_np, original_clean_np], axis=0)
                    if np.abs(original_clean_np).mean() >= self.min_mean_abs: break
                except Exception as e: print(f"Error loading chunk from {track_path}: {e}. Trying again.")

            original_clean_tensor = torch.tensor(original_clean_np, dtype=torch.float32)
            degraded_audio_tensor = codec_simu(original_clean_tensor, sr=self.sr, options=self.codec_options)
            max_scale = original_clean_tensor.abs().max()
            if max_scale > 1e-6:
                original_clean_tensor /= max_scale
                degraded_audio_tensor /= max_scale
            mix = degraded_audio_tensor
            target = original_clean_tensor
            return target, mix
        elif self.dataset_type == 6:
             while True:
                 if not self.metadata:
                      print("Error: Stem Enhancement dataset (Type 6) metadata is empty!")
                      return torch.zeros((2, self.chunk_size), dtype=torch.float32), torch.zeros((2, self.chunk_size), dtype=torch.float32)
                 dirty_path, clean_path, track_length = random.choice(self.metadata)
                 try:
                     offset = None
                     if track_length >= self.chunk_size:
                          offset = np.random.randint(track_length - self.chunk_size + 1)

                     dirty_chunk_np = load_chunk(dirty_path, track_length, self.chunk_size, offset=offset)
                     clean_chunk_np = load_chunk(clean_path, track_length, self.chunk_size, offset=offset)

                     if dirty_chunk_np.shape[-1] != self.chunk_size or clean_chunk_np.shape[-1] != self.chunk_size:
                          continue
                     if dirty_chunk_np.shape[0] == 1: dirty_chunk_np = np.concatenate([dirty_chunk_np, dirty_chunk_np], axis=0)
                     if clean_chunk_np.shape[0] == 1: clean_chunk_np = np.concatenate([clean_chunk_np, clean_chunk_np], axis=0)

                     if np.abs(clean_chunk_np).mean() >= self.min_mean_abs:
                          break
                 except Exception as e:
                      print(f"Error loading chunk pair {dirty_path}/{clean_path}: {e}. Trying again.")

             dirty_tensor = torch.tensor(dirty_chunk_np, dtype=torch.float32)
             clean_tensor = torch.tensor(clean_chunk_np, dtype=torch.float32)

             max_scale = clean_tensor.abs().max()
             if max_scale > 1e-6:
                 clean_tensor /= max_scale
                 dirty_tensor /= max_scale

             mix = dirty_tensor
             target = clean_tensor
             return target, mix
        else:
             raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

        augs_config = self.config.get('augmentations', {})
        if self.aug and 'loudness' in augs_config and augs_config.get('loudness', False):
             loud_values = np.random.uniform(low=augs_config.get('loudness_min', 0.5), high=augs_config.get('loudness_max', 1.5), size=(len(target),))
             loud_values = torch.tensor(loud_values, dtype=torch.float32)
             target = target * loud_values[:, None, None]
             mix = target.sum(0)

        if self.aug and 'mp3_compression_on_mixture' in augs_config and augs_config['mp3_compression_on_mixture'] > 0:
             if random.uniform(0, 1) < augs_config['mp3_compression_on_mixture']:
                 apply_aug = AU.Mp3Compression(min_bitrate=augs_config.get('mp3_compression_on_mixture_bitrate_min', 64), max_bitrate=augs_config.get('mp3_compression_on_mixture_bitrate_max', 192), backend=augs_config.get('mp3_compression_on_mixture_backend', 'lameenc'), p=1.0)
                 mix_np = mix.cpu().numpy().astype(np.float32)
                 required_shape = mix_np.shape
                 try:
                     mix_aug = apply_aug(samples=mix_np, sample_rate=self.sr)
                     if mix_aug.shape != required_shape: mix_aug = mix_aug[..., :required_shape[-1]]
                     mix = torch.tensor(mix_aug, dtype=torch.float32)
                 except Exception as mp3_e: print(f"Mixture MP3 augmentation failed: {mp3_e}")

        if self.config.training.get('target_instrument') is not None and self.dataset_type != 5 and self.dataset_type != 6:
            if self.instruments and self.config.training.target_instrument in self.instruments:
                index = self.instruments.index(self.config.training.target_instrument)
                return target[index:index+1], mix
            else:
                 print(f"Warning: Target instrument '{self.config.training.target_instrument}' not found in list {self.instruments}. Returning all stems.")
                 return target, mix

        return target, mix
