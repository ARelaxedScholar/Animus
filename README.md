# Animus - Autonomous YouTube Content Farm

**Powered by [Orichalcum](https://github.com/ARelaxedScholar/Orichalcum)**

Animus is an autonomous content production system for YouTube, built on the Orichalcum agent orchestration framework. It produces long-form motivation/self-help videos from classic wisdom sources (Stoicism, Biblical Proverbs, Psychological classics, etc.).

## Features

- **Adaptive Content Strategy** - Learns from YouTube analytics to improve over time.
- **Full Pipeline Automation** - Script → TTS → Video → Thumbnail → SEO → Publish.
- **Classic Wisdom Sources** - Bible, Stoicism, Philosophy, Biographies.
- **High-Fidelity Thumbnails** - Integrated with Google's **Imagen (Nano Banana)** via Gemini for cinematic, eye-catching thumbnails.
- **Multi-Account Fleet Management** - Manage multiple YouTube channels and niches from a single dashboard.
- **Intelligent Assembly** - Uses a MoviePy + FFmpeg bridge for high-performance timeline rendering.
- **Concurrent TUI Dashboard** - Non-blocking terminal interface to monitor production, logs, and queue status.
- **Graceful Control** - HTTP API for pause/resume/terminate.

## Channel & Niches

Animus supports multiple niches (Stoicism, Philosophy, etc.) by mapping them to different YouTube accounts stored in the database.

## Quick Start (Nix-based)

This project uses **Nix Flakes** to manage its entire toolchain (Rust, Python, FFmpeg, ImageMagick).

```bash
# 1. Enter the development environment
nix develop

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (DeepSeek, Gemini, Pexels, etc.)

# 3. Start development services (PostgreSQL + MinIO)
# Requires Docker to be installed and running
just dev

# 4. Run database migrations
just migrate

# 5. Add your first YouTube Account
# Rust OAuth (may have issues):
#   just auth-account "Excelsior Academy" "stoicism" --client-id <ID> --client-secret <SECRET>
# Python OAuth (recommended):
just add-account "Excelsior Academy" "stoicism"

# 6. Start the full dashboard (Daemon + TUI)
just dashboard
```

## Commands

Managed via `just`:

- `just dev` - Start infrastructure (Postgres, MinIO).
- `just migrate` - Apply DB migrations.
- `just auth-account <name> <niche>` - Rust OAuth (may have issues)
- `just add-account <name> <niche>` - Python OAuth (recommended)
- `just list-accounts` - List all YouTube accounts
- `just test-account` - Test YouTube credentials
- `just run` - Run the production daemon.
- `just tui` - Run the monitoring dashboard.
- `just dashboard` - Start daemon (background) and TUI (foreground).
- `just clean` - Wipe all local data and build artifacts.

## Deployment to New Machine (Arch/EndeavourOS)

1. **Install Nix**: Use the Determinate Systems installer:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
   ```
2. **Install Docker**: `sudo pacman -S docker docker-compose`.
3. **Clone & Develop**: `git clone <repo> && cd Animus && nix develop`.
4. **Copy Secrets**: Manually copy your `.env` and `credentials.json` (not tracked by git).

## Architecture

```
Scheduler → Strategy → ScriptWriter → TTS → AssetCollector
                                              ↓
                                        VideoAssembler
                                              ↓
                              Thumbnail (Imagen) → SEO → Publisher
```

## License

MIT

