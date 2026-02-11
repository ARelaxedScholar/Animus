# Dev commands
set dotenv-load

# Start development services (PostgreSQL + MinIO)
dev:
    docker-compose up -d
    @echo "Services started. PostgreSQL on :5432, MinIO on :9000"

# Stop development services
dev-down:
    docker-compose down

# Run database migrations
migrate:
    sqlx migrate run

# Download TTS models (Piper)
download-models:
    @mkdir -p models
    @echo "Downloading en_US-lessac-medium.onnx..."
    curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx -o models/en_US-lessac-medium.onnx
    @echo "Downloading en_US-lessac-medium.onnx.json..."
    curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json -o models/en_US-lessac-medium.onnx.json
    @echo "Models downloaded successfully."

# Authenticate a new YouTube account
auth-account name niche *args:
    cargo run --bin auth-helper -- --name "{{name}}" --niche "{{niche}}" {{args}}

# Create a new migration
migration name:
    sqlx migrate add {{name}}

# Build the Docker image using Nix
docker-build:
    nix build .#dockerImage
    docker load < result
    @echo "Docker image 'animus:latest' built and loaded."

# Build the project
build:
    cargo build

# Run the daemon
run:
    cargo run

# Run with hot reload
watch:
    cargo watch -x run

# Run tests
test:
    cargo test

# Format code
fmt:
    cargo fmt

# Lint
lint:
    cargo clippy -- -D warnings

# Generate SQLx offline data
sqlx-prepare:
    cargo sqlx prepare

# Clean build artifacts
clean:
    cargo clean
    rm -rf data/

# Run the TUI dashboard
tui:
    cargo run --bin animus-tui

# Run the daemon in the background and open TUI
dashboard:
    @echo "Starting daemon in background..."
    cargo run --bin animus &
    sleep 2
    cargo run --bin animus-tui
