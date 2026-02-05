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

# Create a new migration
migration name:
    sqlx migrate add {{name}}

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
