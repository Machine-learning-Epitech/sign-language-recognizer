from genericpath import isfile
import os
import shutil
import sys

DEFAULT_SOURCE = 'asl_alphabet_train/asl_alphabet_train'
target = './filtered_images'
MAX_NUMBER_OF_FILES_PER_LETTER = 300

def get_target_folder(max_number_of_files_per_letter: int):
    return target + str(max_number_of_files_per_letter) + '/' + 'filtered_images'

def get_target_letter_folder(max_number_of_files_per_letter: int, letter: str):
    return get_target_folder(max_number_of_files_per_letter=max_number_of_files_per_letter) +  '/' + letter

def get_target_file_path(letter: str, file_name: str, max_number_of_files_per_letter: int):
    return get_target_letter_folder(max_number_of_files_per_letter=max_number_of_files_per_letter, letter=letter) + '/' + file_name

def filter_letter_folder(letter: str, max_number_of_files_per_letter: int, source: str):
    count = 0
    target_folder = get_target_letter_folder(max_number_of_files_per_letter=max_number_of_files_per_letter, letter=letter)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for file in os.listdir(source + '/' + letter):
        original_file_path = source + '/' + letter + '/' + file
        if count < max_number_of_files_per_letter and isfile(original_file_path):
            shutil.copy(original_file_path, get_target_file_path(file_name=file, max_number_of_files_per_letter=max_number_of_files_per_letter, letter=letter))
            count += 1
        else:
            break
    print("Total files in folder " + letter + ": " + str(count))
    return count

def filter_folders():
    source = DEFAULT_SOURCE
    max_number_of_files_per_letter = MAX_NUMBER_OF_FILES_PER_LETTER
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        max_number_of_files_per_letter = int(sys.argv[1])
    if len(sys.argv) > 2:
        source = sys.argv[2]
    count = 0
    target_folder = get_target_folder(max_number_of_files_per_letter=max_number_of_files_per_letter)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for folder in os.listdir(source):
        count = count + filter_letter_folder(letter=folder, max_number_of_files_per_letter=max_number_of_files_per_letter, source=source)
    print("Total files in all folders: " + str(count))

filter_folders()
