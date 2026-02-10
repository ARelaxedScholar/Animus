{
  description = "Animus - Autonomous YouTube content farm powered by Orichalcum";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";

  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [
            "rust-src"
            "rust-analyzer"
          ];
        };

        # Python environment for MoviePy bridge
        pythonEnv = pkgs.python313.withPackages (
          ps: with ps; [
            moviepy
            pillow
            numpy
            requests
            google-auth-oauthlib
            google-auth-httplib2
            google-api-python-client
            psycopg2
            python-dotenv
          ]
        );

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Rust
            rustToolchain
            cargo-edit
            cargo-watch
            sqlx-cli

            # Build dependencies
            pkg-config
            openssl

            # Python and tools
            pythonEnv
            ffmpeg
            imagemagick
            piper-tts

            # Database
            postgresql_16

            # S3-compatible storage (local dev)
            minio
            minio-client

            # Misc tools
            just
            jq
          ];

          shellHook = ''
            echo "🎬 Animus Development Environment"
            echo "   Rust: $(rustc --version)"
            echo "   Python: $(python --version)"
            echo ""

            # Setup local python venv for packages not in nixpkgs (like dspy-ai)
            if [ ! -d ".venv" ]; then
              echo "Creating Python virtual environment..."
              python -m venv .venv
            fi
            source .venv/bin/activate

            # Install dspy-ai if missing
            if ! python -c "import dspy" &> /dev/null; then
              echo "Installing dspy-ai into virtual environment..."
              pip install dspy-ai
            fi

            # Check for Piper models
            if [ ! -f "models/en_US-lessac-medium.onnx" ]; then
              echo "⚠️  Piper models missing. Run 'just download-models' to fetch them."
            fi

            echo "Commands:"
            echo "   cargo build    - Build the project"
            echo "   cargo run      - Run the daemon"
            echo "   just dev       - Start dev services (DB + MinIO)"
            echo ""
            echo "DSPy Judge Commands:"
            echo "   python farm_ctl.py analytics      - Fetch YouTube performance"
            echo "   python farm_ctl.py compile-judge  - Optimize the DSPy Judge"
          '';

          RUST_BACKTRACE = 1;
          RUST_LOG = "animus=debug,orichalcum=debug";
        };
      }
    );
}
