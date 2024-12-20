# Add support for LoRA

## Task Progress
1. Establish working gpt-neox environment that can load pretrained pythia model weights
    - Python installation
        ```
        pip install -r requirements/requirements.txt
        pip install -r requirements/requirements-wandb.txt # optional, if logging using WandB
        pip install -r requirements/requirements-flashattention.txt
        python ./megatron/fused_kernels/setup.py install # optional, if using fused kernels
        ```
    - Other software dependencies:
        - git-lfs (for downloading model weights)
            - `wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz`
            - `tar -xvzf git-lfs-linux-amd64-v3.5.1.tar.gz`
            - `export PREFIX=/home/matt; ./git-lfs-3.5.1/install.sh`
            - `echo 'export PATH=$HOME/bin:$PATH' >> ~/.bashrc`
        - nvcc (for apex)
            - `wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run`
            - `bash cuda_12.4.1_550.54.15_linux.run --toolkit --installpath=/home/matt/local/cuda --silent`
            - `export PATH="/home/matt/local/cuda/bin:$PATH"` and add to bashrc
        - apex (for the fused layer norms)
            - **Manual Hack**: `--config-settings "--build-option=--cuda_ext"` did not propagate the `--cuda-ext` flag properly to `setup.py` so I had to manually add the following
                ```python
                sys.argv.append("--cpp_ext")
                sys.argv.append("--cuda_ext")
                print("sys.argv after manuall adding cpp/cuda flags:")
                print(sys.argv)
                ```
                to `setup.py`. [This](https://github.com/NVIDIA/apex/issues/1204#issuecomment-1659884672) would also have worked.
            - `pip install wheel`
            - `cd ..; git clone https://github.com/NVIDIA/apex; cd apex`
            - `pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./`
            - `cd ..`
            - *Check*: `python -c "import torch; import amp_C"` to test we have installed correctly
    - Download pythia models in original neox format from EleutherAI/neox-ckpt-pythia-xxx-deduped-v1
        - `git clone https://huggingface.co/EleutherAI/neox-ckpt-pythia-70m-deduped-v0`
        - `git clone https://huggingface.co/EleutherAI/pythia-70m-deduped` (for the tokenizer)
    - TODO: Instantiate randomly initialised model with pythia config and start training (currently erroring)
        - python ./deepy.py train.py -d configs pythia/70M.yml local_setup.yml finetuning.yml
    - TODO: Load pretrained pythia model and start 1 iteration training run
        - ...
2. Establish baseline: Finetune pretrained Pythia model on easy benchmark
    - Let's go with GLUE to start with as per the LoRA paper

Misc:
- Dead ends:
    - Tokenizer loading was erroring, until I realised that I needed to set this to the pythia tokenizer. Debugging was a bit tricky b/c that library is written in rust.


