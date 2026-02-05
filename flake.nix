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
        pythonEnv = pkgs.python312.withPackages (
          ps: with ps; [
            moviepy
            pillow
            numpy
            requests
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

            # Build dependencies
            pkg-config
            openssl

            # Python for video processing
            pythonEnv
            ffmpeg
            imagemagick

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
            echo "Commands:"
            echo "   cargo build    - Build the project"
            echo "   cargo run      - Run the daemon"
            echo "   just dev       - Start dev services (DB + MinIO)"
          '';

          RUST_BACKTRACE = 1;
          RUST_LOG = "animus=debug,orichalcum=debug";
        };
      }
    );
}
