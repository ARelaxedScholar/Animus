{
  description = "Animus service - NixOS module and package";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    animus.url = "path:..";
    animus.inputs.nixpkgs.follows = "nixpkgs";
    animus.inputs.flake-utils.follows = "flake-utils";
    animus.inputs.rust-overlay.follows = "rust-overlay";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay,
      animus,
    }:
    let
      lib = nixpkgs.lib;

      animusModule =
        { config, pkgs, ... }:
        let
          cfg = config.services.animus;
        in
        {
          options.services.animus = {
            enable = lib.mkEnableOption "Animus service";
            package = lib.mkOption {
              type = lib.types.package;
              description = "Animus package to use";
              default = animus.packages.x86_64-linux.animus;
            };
            user = lib.mkOption {
              type = lib.types.str;
              default = "animus";
              description = "User to run animus as";
            };
            group = lib.mkOption {
              type = lib.types.str;
              default = "animus";
              description = "Group to run animus as";
            };
            dataDir = lib.mkOption {
              type = lib.types.path;
              default = "/var/lib/animus";
              description = "Data directory for animus";
            };
            envFile = lib.mkOption {
              type = lib.types.path;
              description = "Path to .env file containing secrets (not tracked in git)";
              example = "/etc/animus/secrets.env";
            };
          };

          config = lib.mkIf cfg.enable {
            users.users.${cfg.user} = {
              isSystemUser = true;
              group = cfg.group;
              description = "Animus service user";
              home = cfg.dataDir;
              createHome = true;
            };

            users.groups.${cfg.group} = { };

            systemd.services.animus = {
              description = "Animus - Autonomous YouTube content farm";
              wantedBy = [ "multi-user.target" ];
              after = [
                "network.target"
                "postgresql.service"
              ];
              serviceConfig = {
                Type = "simple";
                User = cfg.user;
                Group = cfg.group;
                WorkingDirectory = cfg.dataDir;
                Restart = "on-failure";
                RestartSec = "5s";
                EnvironmentFile = cfg.envFile;

                Environment = [
                  "S3_ENDPOINT=http://localhost:9000"
                  "S3_BUCKET=animus-assets"
                  "S3_REGION=us-east-1"
                  "TTS_PROVIDER=piper"
                  "QWEN3_API_URL=http://localhost:8000/v1"
                  "QWEN3_VOICE=default"
                  "OPENAI_TTS_VOICE=onyx"
                  "OPENAI_TTS_MODEL=tts-1-hd"
                  "TTS_SPEED=1.0"
                  "ASSET_MIN_CLIPS_PER_SECTION=5"
                  "CHANNEL_NAME=The Light Of Orion"
                  "CHANNEL_TAGLINE=Wisdom for the journey upward, enlightenment from a higher realm."
                  "TARGET_DURATION_MIN=15"
                  "TARGET_DURATION_MAX=20"
                  "SCRIPT_IMPROVEMENT_ENABLED=true"
                  "SCRIPT_IMPROVEMENT_MAX_ITERATIONS=10"
                  "SCRIPT_IMPROVEMENT_STAGNATION_THRESHOLD=10"
                  "SCRIPT_IMPROVEMENT_THRESHOLD=9.0"
                  "SCRIPT_IMPROVEMENT_CANDIDATES=3"
                  "VIDEOS_PER_WEEK=3"
                  "CONTROL_API_PORT=8080"
                  "RUST_LOG=animus=info,orichalcum=info"
                ];
              };
            };

            systemd.tmpfiles.rules = [
              "d ${cfg.dataDir} 0755 ${cfg.user} ${cfg.group} -"
            ];
          };
        };
    in
    {
      packages.x86_64-linux = {
        inherit (animus.packages.x86_64-linux) animus dockerImage;
      };

      nixosModules.animus = animusModule;
    };
}
