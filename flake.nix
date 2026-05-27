{
  description = "DDSSM — Diffusion-Driven State Space Models (CUDA devshell)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        # CUDA toolkit + cuDNN are unfree-redistributable.
        # cudaSupport=true is deliberately *not* set: we use nix only
        # for the system-side deps (nvcc, gcc13, cudnn headers); the
        # PyTorch wheel bundles its own CUDA runtime and is fetched by
        # uv from PyPI. Flipping cudaSupport=true would trigger CUDA
        # rebuilds for unrelated nixpkgs and is unnecessary here.
        config.allowUnfree = true;
      };

      # RTX 4090. Override per-shell with TORCH_CUDA_ARCH_LIST=...
      # before invoking `nix develop` if running on a different GPU.
      defaultCudaArch = "8.9";

      cudatoolkit = pkgs.cudaPackages.cudatoolkit;
      cudnn = pkgs.cudaPackages.cudnn;
    in {
      devShells.${system}.default = pkgs.mkShell {
        name = "ddssm-dev";

        packages = with pkgs; [
          python313
          uv
          # Host compiler for source-built python ext modules (mamba_ssm
          # etc.). gcc-13 is the newest gcc that CUDA 12.x explicitly
          # supports; build_mamba.sh originally pinned gcc-11 against
          # an older toolkit, but that's been EOL'd in nixpkgs.
          gcc13
          gnumake
          pkg-config
          stdenv.cc.cc.lib

          cudatoolkit
          cudnn

          git
          pre-commit
          # Optional creature comforts; remove freely.
          ripgrep
          jq
        ];

        shellHook = ''
          # Compiler for any source-built python ext (mamba_ssm etc.)
          export CC=${pkgs.gcc13}/bin/gcc
          export CXX=${pkgs.gcc13}/bin/g++

          # CUDA_HOME mirrors a "system" CUDA install for tools that
          # expect a single tree (nvcc, cuda headers, libcudart).
          export CUDA_HOME=${cudatoolkit}
          export CUDA_PATH=${cudatoolkit}

          # PyTorch's setup.py and many third-party CUDA extensions
          # (mamba_ssm in particular) compile only the listed arches.
          # 8.9 = RTX 4090.
          export TORCH_CUDA_ARCH_LIST="''${TORCH_CUDA_ARCH_LIST:-${defaultCudaArch}}"

          # Runtime linker: torch wheels bundle CUDA runtime libs but
          # still dlopen the *userland NVIDIA driver* (libcuda.so.1),
          # which on NixOS lives at /run/opengl-driver/lib — that path
          # has to be searchable or `torch.cuda.is_available()` returns
          # False on this host.
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:${cudatoolkit}/lib:${cudnn}/lib:${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

          # uv: keep its cache and venvs inside the project so the
          # devshell is self-contained and disposable.
          export UV_PYTHON=${pkgs.python313}/bin/python3.13
          export UV_PYTHON_DOWNLOADS=never

          cat <<INFO
ddssm devshell ready
  python : $(python3 --version 2>&1)
  uv     : $(uv --version 2>&1)
  nvcc   : $(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')
  arch   : TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
  first run: \`uv sync\` then \`uv run python -m ddssm.app\`
INFO
        '';
      };
    };
}
