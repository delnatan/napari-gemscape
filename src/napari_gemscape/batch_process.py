"""
this file contains functions that carry out 'batch' processing from the widget's
list of files. Progress on the batch processing is monitored via stdout by
printing "PROGRESS:<int>" where the integer has to be a value from 0 to 100.

This is done by parsing the string and checking to see if it's starts with
'PROGRESS:', then taking the second element of the results after splitting it
with ':'

"""

import sys
import time


def process_files(task, file_list):
    total_files = len(file_list)
    if task == "test":
        for i, fn in enumerate(file_list, 1):
            time.sleep(0.5)
            progress = int((i / total_files) * 100)
            print(f"PROGRESS:{progress}")
            sys.stdout.flush()

            print(f"Processing {fn}")
            sys.stdout.flush()


if __name__ == "__main__":

    task = sys.argv[1]
    file_list = sys.argv[2:]

    process_files(task, file_list)
