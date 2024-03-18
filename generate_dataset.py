from SFXlearner.src.generate_data import (
    slice_guitarset,
    slice_idmt_smt_guitar,
    generate_dataset_sox,
)
import os
import soundfile as sf
import librosa
import numpy as np


guitarset10_path = "dataset/guitarset10_unrendered/"
guitarset10_path_sliced = "dataset/guitarset10_sliced/"
guitarset_path_rendered = "dataset/guitarset10_rendered_small/"
idmt_path = "dataset/idmt-smt-guitar_unrendered/"
idmt_path_sliced = "dataset/idmt-smt-guitar_sliced/"
idmt_path_rendered = "dataset/idmt-smt-guitar_rendered/"


#######
# Uncomment to slice dataset to 5s clips
#######
"""
# slice guitarset into 5s clips
#guitarset_path_sliced = slice_guitarset(
#    guitarset10_path, save_dir=guitarset10_path_sliced, duration=5
#)

#idmt_path_sliced = slice_idmt_smt_guitar(
#    data_home=idmt_path, save_dir=idmt_path_sliced, duration=5
#)
"""
#######
# Uncomment perform generate_sox
#######
"""
guitarset10_path_sliced = guitarset10_path_sliced + "/1/"
idmt_path_sliced = idmt_path_sliced + "/IDMT-SMT-GUITAR_5s/"
generate_dataset_sox([guitarset10_path_sliced], guitarset_path_rendered, methods=[1, 5], valid_split=0.2)
#generate_dataset_sox([idmt_path_sliced], idmt_path_rendered, methods=[1, 5], valid_split=0.2)
#print("hey")
"""
#######
# Uncomment to create subfolders
#######
"""
import os
import shutil

source_dir = guitarset10_path_sliced + "guitarset_5.0s_clean/"

files_per_folder = 20
first_folder_count = 24

wav_files = [f for f in sorted(os.listdir(source_dir)) if f.lower().endswith('.wav')]

def create_subfolders(files, dest_dir, count_per_subfolder, start_index=1):
    subfolder_index = start_index
    while files:
        # Pop a slice of `count_per_subfolder` .wav files from the list
        subset = files[:count_per_subfolder]
        files = files[count_per_subfolder:]

        # Create a subfolder name such as "1", "2", "3", etc.
        subfolder_name = str(subfolder_index)
        subfolder_index += 1
        subfolder_path = os.path.join(dest_dir, subfolder_name)

        # Create the subfolder if it doesn't exist
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Move the subset of .wav files into the newly created subfolder
        for wav_file in subset:
            shutil.move(os.path.join(source_dir, wav_file), os.path.join(subfolder_path, wav_file))

# First, handle the special case folder that has 24 files
create_subfolders(wav_files, guitarset10_path_sliced, first_folder_count, start_index=1)

# Now, use the remaining files and split them into subfolders of 20 files each
create_subfolders(wav_files[first_folder_count:], guitarset10_path_sliced, files_per_folder, start_index=2)
"""
#####
# Uncomment to perform generate_sox on all folder
#####
"""
def resample(folder_path, target_sr=16000):
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            audio, src_sr = librosa.load(filepath, sr=None)
            audio_resampled = librosa.resample(audio, orig_sr=src_sr, target_sr=target_sr)
            sf.write(filepath, audio_resampled, target_sr, subtype='PCM_16')


i = 1
for item in os.listdir(guitarset10_path_sliced):
    item_path = os.path.join(guitarset10_path_sliced, item)
    if os.path.isdir(item_path):
        print(item_path)
        index_str = f"/{i}/"
        sliced_path = guitarset10_path_sliced + index_str
        rendered_path = guitarset_path_rendered + index_str
        #generate_dataset_sox([sliced_path], rendered_path, methods=[1, 5], valid_split=0.2)
        resample(f'{rendered_path}gen_multiFX_03182024/train/audio/')
        resample(f'{rendered_path}gen_multiFX_03182024/valid/audio/')
        i += 1
"""

"""
#####
# Uncomment to merge data
#####
import os
import shutil
import torch
import torchaudio

# Base path for the source dataset
base_dataset_path = guitarset_path_rendered
destination_path = f"{guitarset_path_rendered}data"
parts = ['train', 'valid']
file_rename_index = 0

for part in parts:
    os.makedirs(os.path.join(destination_path, part, 'audio'), exist_ok=True)

for subfolder in sorted(os.listdir(base_dataset_path)):
    subfolder_path = os.path.join(base_dataset_path, subfolder)
    if os.path.isdir(subfolder_path):
        for part in parts:
            audio_source_path = os.path.join(subfolder_path, "gen_multiFX_03182024", part, "audio")
            label_source_path = os.path.join(subfolder_path, "gen_multiFX_03182024", part, "Label_tensor.pt")
            audio_destination_path = os.path.join(destination_path, part, 'audio')

            if os.path.exists(audio_source_path):
                for filename in sorted(os.listdir(audio_source_path)):
                    if filename.lower().endswith('.wav'):
                        new_filename = f"{file_rename_index}.wav"
                        shutil.move(os.path.join(audio_source_path, filename),
                                    os.path.join(audio_destination_path, new_filename))
                        file_rename_index += 1

            if os.path.exists(label_source_path):
                label_destination_path = os.path.join(destination_path, part, 'Label_tensor.pt')
                if os.path.exists(label_destination_path):
                    existing_labels = torch.load(label_destination_path)
                    new_labels = torch.load(label_source_path)
                    combined_labels = torch.cat((existing_labels, new_labels), dim=0)
                    torch.save(combined_labels, label_destination_path)
                else:
                    shutil.move(label_source_path, label_destination_path)
"""
"""
import torchaudio
from datasets import Dataset
from functools import partial
import torch
def gen_inner(path):
    labels = torch.load(os.path.join(path, "Label_tensor.pt"))
    audio_path = os.path.join(path, "audio")
    sorted_filenames = sorted(os.listdir(audio_path), key=lambda filename: int(filename.split('.')[0]))
    for index, filename in enumerate(sorted_filenames):
        if filename.lower().endswith('.wav'):
            full_path = os.path.join(audio_path, filename)
            waveform, sampling_rate = torchaudio.load(full_path)
            waveform = waveform.squeeze().numpy()

            return {
                'audio': {
                    'path': full_path,
                    'array': np.array(waveform),
                    'sampling_rate': sampling_rate},
                'label': labels[index]
            }
def gen(original_path):
    yield {
        'train': partial(gen_inner,f"{original_path}data/train/"),
        'valid': partial(gen_inner,f"{original_path}data/valid/")
    }
gen_partial = partial(gen, path=guitarset_path_rendered)
dataset = Dataset.from_generator(gen_partial)
print(dataset.column_names)
"""

import torchaudio
from datasets import Dataset, load_dataset
from transformers import AutoFeatureExtractor, ASTForAudioClassification
from functools import partial
import torch

def gen(path):
    labels = torch.load(os.path.join(path, "Label_tensor.pt"))
    audio_path = os.path.join(path, "audio")
    sorted_filenames = sorted(os.listdir(audio_path), key=lambda filename: int(filename.split('.')[0]))
    for index, filename in enumerate(sorted_filenames):
        if filename.lower().endswith('.wav'):
            full_path = os.path.join(audio_path, filename)
            waveform, sampling_rate = torchaudio.load(full_path)
            waveform = waveform.squeeze().numpy()

            yield {
                'audio': {
                    'path': full_path,
                    'array': np.array(waveform),
                    'sampling_rate': sampling_rate},
                'label': labels[index]
            }

librispeech = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")

gen_train = partial(gen, path=f"{guitarset_path_rendered}data/train/")
gen_valid = partial(gen, path=f"{guitarset_path_rendered}data/valid/")
dataset_train = Dataset.from_generator(gen_train)
dataset_valid = Dataset.from_generator(gen_valid)

sampling_rate = dataset_train[0]["audio"]["sampling_rate"]
feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
inputs = feature_extractor(dataset_train[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.argmax(logits, dim=-1).item()
predicted_label = model.config.id2label[predicted_class_ids]
print(predicted_label)


"""
print(librispeech.column_names)
print(dataset_train.column_names)
print(librispeech[0]["id"])
print(type(librispeech[0]["audio"]["array"]))
print(type(dataset_train[0]["audio"]["array"]))
print(librispeech[0]["audio"]["array"][:10])
print(dataset_train[0]["audio"]["array"][:10])
"""





"""
import os
import numpy as np
import torchaudio
from transformers import ASTFeatureExtractor

# Initialize the feature extractor
feature_extractor = ASTFeatureExtractor()

source_folder = f'{guitarset_path_rendered}gen_multiFX_03182024/train/audio/'
target_folder = f'{guitarset_path_rendered}gen_multiFX_03182024/train/spectrogram/'

# Ensure target directory exists
if not os.path.isdir(target_folder):
    os.makedirs(target_folder)

# Loop through all files in the source folder
for filename in os.listdir(source_folder):
    if filename.lower().endswith('.wav'):  # Process only .wav files
        filepath = os.path.join(source_folder, filename)

        # Load the waveform and extract features
        waveform, sampling_rate = torchaudio.load(filepath)
        waveform = waveform.squeeze().numpy()

        inputs = feature_extractor(waveform, sampling_rate=sampling_rate, padding="max_length", return_tensors="pt")
        input_values = inputs.input_values

        # Save the "inputs" to a file in the target folder (using the same base filename)
        output_filename = os.path.splitext(filename)[0] + '.npy'
        output_filepath = os.path.join(target_folder, output_filename)

        # Save as a .npy file
        np.save(output_filepath, input_values.numpy())  # Convert to numpy array and save

# Inform user processing complete
print("Feature extraction and saving completed.")
"""
"""

"""

