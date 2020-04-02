import os   
import re

def rename(directory, old_string, new_string):
    """Rename files

    This function is used to rename files in a directory by replacing some strings in old nane with new strings.
    For example, we want to rename "MVPA_2019_schizophrenia" to "Machine_Learning_2019_schizophrenia",
    You just need to apply old strings of "MVPA" and new strings of "Machine_Learning" as well as directory containing old files.
    NOTE: old_string and new_string must be standarded regular expression
    """
    old_files_name = os.listdir(directory)
    new_files_name = [re.sub(old_string, new_string, file) for file in old_files_name]
    # print(old_files_name)
    # print(new_files_name)
    [os.rename(os.path.join(os.getcwd(), old_file_name), os.path.join(os.getcwd(), new_file_name)) for (old_file_name, new_file_name) in zip (old_files_name, new_files_name)]


if __name__ == "__main__":
    rename("D:/Papers/DorctorDegree", r"Machine_Learning", "Machine_learning")