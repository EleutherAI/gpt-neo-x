# Script counts tokens in a pretokenized dataset from preprocess_data.py
# Necessary for setting batch size, train_iters, etc

import sys
import os

## Necessary for the import
project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", ".."
    )
)
sys.path.insert(0, project_root)

from megatron.data import indexed_dataset
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python dataset_token_count.py /absolute/file/path/to/dataset1 /absolute/file/path/to/dataset2 ...")
    sys.exit(1)

 # Access the command-line arguments
arguments = sys.argv[1:]

 # Your code here - process the list of arguments
print("Command-line arguments:", arguments)
for arg in arguments:
dataset = indexed_dataset.make_dataset("/home/mchorse/data/pile_deduped/pile_0.87_deduped_text_document", "mmap")
size = np.sum(dataset.sizes)
print("Dataset size in tokens is", size)
