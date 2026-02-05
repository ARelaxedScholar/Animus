# Animus - Autonomous YouTube Content Farm

**Powered by [Orichalcum](https://github.com/ARelaxedScholar/Orichalcum)**

Animus is an autonomous content production system for YouTube, built on the Orichalcum agent orchestration framework. It produces long-form motivation/self-help videos from classic wisdom sources.

## Features

- **Adaptive Content Strategy** - Learns from YouTube analytics to improve over time
- **Full Pipeline Automation** - Script → TTS → Video → Thumbnail → SEO → Publish
- **Classic Wisdom Sources** - Bible, Stoicism, Philosophy, Biographies
- **Configurable Voice Personas** - ElevenLabs TTS with customizable voices
- **Graceful Control** - HTTP API for pause/resume/terminate

## Channel

**Excelsior Academy** - *Wisdom for the journey upward*

## Quick Start

```bash
# Enter development environment
nix develop

# Copy and configure environment
cp .env.example .env

# Start dev services (PostgreSQL + MinIO)
just dev

# Run migrations
just migrate

# Start the daemon
cargo run
```

## Architecture

```
Scheduler → Strategy → ScriptWriter → TTS → AssetCollector
                                              ↓
                                        VideoAssembler
                                              ↓
                              Thumbnail → SEO → Publisher
```

## Configuration

See `.env.example` for required environment variables.

## License

MIT
