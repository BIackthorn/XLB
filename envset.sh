python3 -m pip install --upgrade pip
python3 -m pip install pyevtk numpy pillow scipy jupyter jupyter-server matplotlib pandas scikit-fmm
python3 -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python3 -m pip install jmp warp-lang tqdm pyvista trimesh Rtree orbax-checkpoint termcolor
python3 -m pip install git+https://github.com/loliverhennigh/PhantomGaze.git

export PYTHONPATH=.
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_async_all_gather=true --xla_gpu_enable_async_reduce_scatter=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_enable_async_all_reduce=true"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export CUDA_MODULE_LOADING=EAGER