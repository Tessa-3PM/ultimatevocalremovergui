# Really barebones right now 
# When you definitely know how to spell separate
from separate import (
    SeperateMDX as SeparateMDX, # Model-related, can import the other model classes as well
    save_format, clear_gpu_cache,  # Utility functions
    cuda_available, mps_available, #directml_available,
)
from gui_data.constants import *
from UVR import ModelData, MainWindow

import os
import shutil
import soundfile as sf
import audioread
import librosa

def process_storage_check():
    """Verifies storage requirments"""
    total, used, free = shutil.disk_usage("/") 
    space_details = f'Detected Total Space: {int(total/1.074e+9)} GB\'s\n' +\
                    f'Detected Used Space: {int(used/1.074e+9)} GB\'s\n' +\
                    f'Detected Free Space: {int(free/1.074e+9)} GB\'s\n'     
    appropriate_storage = True     
    if int(free/1.074e+9) <= int(2):
        raise RuntimeError([STORAGE_ERROR[0], f'{STORAGE_ERROR[1]}{space_details}'])
        appropriate_storage = False 
    if int(free/1.074e+9) in [3, 4, 5, 6, 7, 8]:
        raise RuntimeWarning([STORAGE_WARNING[0], f'{STORAGE_WARNING[1]}{space_details}{CONFIRM_WARNING}'])
        appropriate_storage = self.message_box([STORAGE_WARNING[0], f'{STORAGE_WARNING[1]}{space_details}{CONFIRM_WARNING}'])      
    return appropriate_storage


def start_processing(inputPath, exportPath):
    # Starting from the framework of the GUI `process_initialize`
    # There, they will validate inputs. I am going to hard code most of the inputs here (for now). 
    inputPaths = [inputPath]
    if not os.path.isfile(inputPaths[0]):
        raise ValueError(INVALID_INPUT)
    if not os.path.isdir(exportPath):
        raise ValueError(INVALID_EXPORT)
    process_storage_check()
    process(inputPaths, exportPath)

def verify_audio(audio_file):

    if not type(audio_file) is tuple:
        audio_file = [audio_file]

    for i in audio_file: 
        if os.path.isfile(i): 
            # This is a 'check' in that it will error if something is wrong
            # Creates a 30s sample first 
            BASE_PATH = os.path.dirname(os.path.abspath(__file__))
            sample_path = os.path.join(BASE_PATH, 'temp_sample_clips')
            with audioread.audio_open(i) as f:
                track_length = int(f.duration)
            clip_duration = 30
            if track_length >= clip_duration:
                offset_cut = track_length//3
                off_cut = offset_cut + track_length
                if not off_cut >= clip_duration:
                    offset_cut = 0 
                name_apped = f'{clip_duration}_second_'  
            else:
                offset_cut, clip_duration = 0, track_length
                name_apped = ''   
            sample = librosa.load(audio_file, offset=offset_cut, duration=clip_duration, mono=False, sr=44100)[0].T
            audio_sample = os.path.join(sample_path, f'{os.path.splitext(os.path.basename(audio_file))[0]}_{name_apped}sample.wav')
            sf.write(audio_sample, sample, 44100)          
            # Then load the sample 
            librosa.load(audio_sample, duration=3, mono=False, sr=44100)
        else: 
            raise ValueError(f'File {i} does not exist')
    
def process(inputPaths, exportPath):
    chosen_process_method = MDX_ARCH_TYPE
    wav_type = 'PCM_16'
    is_ensemble = False 
    true_model_count = 0
    iteration = 0
    is_verified_audio = True
    inputPath_total_len = len(inputPaths)
    # Would only process 30s
    is_model_sample_mode = False

    model = [ModelData(None, MDX_ARCH_TYPE)]
    # TODO: cached_source_model_list_check(model)
    true_model_4_stem_count = 0 # Would be higher if DEMUCS_ARCH_TYPE 
    true_model_pre_proc_model_count = sum(2 if m.pre_proc_model_activated else 0 for m in model)
    true_model_count = sum(2 if m.is_secondary_model_activated else 1 for m in model) + true_model_4_stem_count + true_model_pre_proc_model_count
    # Skipping a lot of logic here bc I'm only allowing the user to input one file 
    for file_num, audio_file in enumerate(inputPaths, start=1):
        verify_audio(audio_file)
        for current_model_num, current_model in enumerate(model, start=1):
            iteration += 1 
            audio_file_base = f"{file_num}_{os.path.splitext(os.path.basename(audio_file))[0]}"
            process_data = {
                'model_data': current_model, 
                'export_path': exportPath,
                'audio_file_base': audio_file_base,
                'audio_file': audio_file,
                'set_progress_bar': lambda step, inference_iterations=0:print(f"Iterations: {inference_iterations}, step: {step}"),
                'write_to_console': lambda progress_text, base_text='': print(progress_text + "\n" + base_text),
                'process_iteration': iteration + 1,
                'cached_source_callback': None,
                'cached_model_source_holder': None,
                'list_all_models': model,
                'is_ensemble_master': False,
                'is_4_stem_ensemble': False
            }
            seperator = SeparateMDX(current_model, process_data)
            seperator.seperate()
            print("DONE!")