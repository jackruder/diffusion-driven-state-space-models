export CC=gcc-11
export CXX=g++-11
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST="8.9" # for RTX 4090, adjust for other
git clone https://github.com/state-spaces/mamba.git
cd mamba
modify setup.py line 193 to “ cc_flag.append("arch=compute_89,code=sm_89") ”
pip install -e .
python -c "from mamba_ssm import Mamba; print('✅ Mamba imported successfully')"
