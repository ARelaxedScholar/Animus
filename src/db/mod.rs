//! Database operations with SQLx

pub mod models;
pub mod queries;

pub use models::{Video, VideoStatus};
pub use queries::*;

use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

/// Create a database connection pool
pub async fn create_pool(database_url: &str) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        .max_connections(5)
        .connect(database_url)
        .await
}

/// Run database migrations
pub async fn run_migrations(pool: &PgPool) -> Result<(), sqlx::migrate::MigrateError> {
    sqlx::migrate!("./migrations").run(pool).await
}
