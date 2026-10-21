{
  description = "Python Data Science Environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        
        # Base packages managed by Nix
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          jupyter
          ipykernel   # Required for Jupyter to launch the custom kernel
          pandas
          pip
          numpy       
          matplotlib  
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ 
            pythonEnv 
            pkgs.stdenv.cc.cc.lib 
            pkgs.zlib
          ];

          shellHook = ''
            export VENV_DIR="$PWD/.venv"
            
            # Sandbox Jupyter's configuration to the local project folder
            export JUPYTER_DATA_DIR="$PWD/.jupyter_data"

            # 1. Create a local virtual environment that INHERITS Nix packages
            if [ ! -d "$VENV_DIR" ]; then
              echo "Creating local virtual environment (.venv)..."
              # --system-site-packages allows the venv to see Nix's Jupyter/ipykernel
              ${pythonEnv.interpreter} -m venv --system-site-packages "$VENV_DIR"
            fi

            # 2. Activate the virtual environment
            source "$VENV_DIR/bin/activate"

            # 3. Automatically register the .venv as the default Jupyter kernel
            KERNEL_DIR="$JUPYTER_DATA_DIR/kernels/python3"
            mkdir -p "$KERNEL_DIR"
            
            cat > "$KERNEL_DIR/kernel.json" <<EOF
            {
              "argv": [
                "$VENV_DIR/bin/python",
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}"
              ],
              "display_name": "Python (.venv)",
              "language": "python"
            }
            EOF

            # 4. Fix for dynamically linked libraries when compiling C-extensions via pip
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"

            echo "Environment activated."
            echo "Jupyter is now globally configured to use your .venv!"
            echo "To start, run: jupyter notebook"
          '';
        };
      });
}
