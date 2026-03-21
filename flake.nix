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

        pyproject = builtins.fromTOML (builtins.readFile ./pyproject.toml);
        pyprojectDeps = pyproject.project.dependencies or [ ];

        normalizePythonDepName =
          dep:
          let
            depMatch = builtins.match "^[[:space:]]*([A-Za-z0-9._-]+).*" dep;
            depName = if depMatch == null then dep else builtins.elemAt depMatch 0;
            normalizedName = builtins.replaceStrings [ "-" "." ] [ "_" "_" ] (pkgs.lib.strings.toLower depName);
          in
          if normalizedName == "psycopg2_binary" then "psycopg2" else normalizedName;

        resolvePythonPackageAttr =
          ps: normalizedName:
          let
            candidates = [
              (builtins.replaceStrings [ "_" ] [ "-" ] normalizedName)
              normalizedName
            ];
            resolved = builtins.filter (name: builtins.hasAttr name ps) candidates;
          in
          if resolved == [ ] then null else builtins.elemAt resolved 0;

        # Python environment for MoviePy bridge
        pythonEnv = pkgs.python3.withPackages (
          ps:
          let
            pyprojectRuntimeDeps = builtins.map normalizePythonDepName pyprojectDeps;
            extraRuntimeDeps = builtins.map normalizePythonDepName [
              "moviepy"
              "pillow"
              "numpy"
              "google-auth-oauthlib"
              "google-auth-httplib2"
              "psutil"
            ];
            allRuntimeDepNames = pkgs.lib.lists.unique (pyprojectRuntimeDeps ++ extraRuntimeDeps);
            resolvedRuntimeDepNames = builtins.filter (name: name != null) (
              builtins.map (resolvePythonPackageAttr ps) allRuntimeDepNames
            );
            availableRuntimeDepNames = pkgs.lib.lists.unique resolvedRuntimeDepNames;
          in
          builtins.map (name: builtins.getAttr name ps) availableRuntimeDepNames
        );

        # Animus package
        animus = pkgs.rustPlatform.buildRustPackage {
          pname = "animus";
          version = "0.1.0";
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [
            pkg-config
            makeWrapper
          ];
          buildInputs = with pkgs; [ openssl ];

          SQLX_OFFLINE = "true";

          postInstall = ''
            wrapProgram $out/bin/animus \
              --prefix PATH : ${
                pkgs.lib.makeBinPath [
                  pkgs.ffmpeg
                  pkgs.imagemagick
                  pkgs.ghostscript
                  pkgs.piper-tts
                  pythonEnv
                ]
              }
          '';
        };

        # Application assets and scripts for the Docker image
        appAssets = pkgs.stdenv.mkDerivation {
          name = "animus-app-assets";
          src = ./.;
          installPhase = ''
            mkdir -p $out/app
            cp -r scripts $out/app/
            cp -r src $out/app/
            # Ensure templates directory structure exists
            mkdir -p $out/app/templates/thumbnails
            # Copy any existing template files
            if [ -d templates ]; then
              cp -r templates/. $out/app/templates/ 2>/dev/null || true
            fi
          '';
        };

        # Docker Image
        dockerImage = pkgs.dockerTools.buildLayeredImage {
          name = "animus";
          tag = "latest";
          contents = with pkgs; [
            animus
            appAssets
            bash
            coreutils
            cacert
            ffmpeg
            imagemagick
            ghostscript
            piper-tts
            pythonEnv
          ];

          config = {
            Cmd = [ "/bin/animus" ];
            Env = [
              "PYTHONPATH=${pythonEnv}/${pkgs.python3.sitePackages}"
              "PATH=/bin:/usr/bin"
            ];
            WorkingDir = "/app";
          };
        };

      in
      {
        packages = {
          default = animus;
          animus = animus;
          dockerImage = dockerImage;
        };

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
            uv
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
            export UV_PYTHON=${pythonEnv}/bin/python3
            export UV_PYTHON_DOWNLOADS=never

            echo "🎬 Animus Development Environment"
            echo "   Rust: $(rustc --version)"
            echo "   Python: $(python --version)"
            echo ""

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
