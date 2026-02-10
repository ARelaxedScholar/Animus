use sqlx::PgPool;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct YouTubeAccount {
    pub id: i32,
    pub name: String,
    pub niche: Option<String>,
    pub client_id: String,
    pub client_secret: String,
    pub refresh_token: String,
    pub channel_id: Option<String>,
    pub is_active: bool,
}

pub async fn get_account(pool: &PgPool, id: i32) -> Result<Option<YouTubeAccount>, sqlx::Error> {
    sqlx::query_as!(
        YouTubeAccount,
        "SELECT id, name, niche, client_id, client_secret, refresh_token, channel_id, is_active AS \"is_active!\" FROM youtube_accounts WHERE id = $1",
        id
    )
    .fetch_optional(pool)
    .await
}

pub async fn get_account_by_name(pool: &PgPool, name: &str) -> Result<Option<YouTubeAccount>, sqlx::Error> {
    sqlx::query_as!(
        YouTubeAccount,
        "SELECT id, name, niche, client_id, client_secret, refresh_token, channel_id, is_active AS \"is_active!\" FROM youtube_accounts WHERE name = $1",
        name
    )
    .fetch_optional(pool)
    .await
}

pub async fn list_active_accounts(pool: &PgPool) -> Result<Vec<YouTubeAccount>, sqlx::Error> {
    sqlx::query_as!(
        YouTubeAccount,
        "SELECT id, name, niche, client_id, client_secret, refresh_token, channel_id, is_active AS \"is_active!\" FROM youtube_accounts WHERE is_active = true"
    )
    .fetch_all(pool)
    .await
}

pub async fn upsert_account(
    pool: &PgPool,
    name: &str,
    niche: Option<&str>,
    client_id: &str,
    client_secret: &str,
    refresh_token: &str,
) -> Result<i32, sqlx::Error> {
    let rec = sqlx::query!(
        r#"
        INSERT INTO youtube_accounts (name, niche, client_id, client_secret, refresh_token)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (name) DO UPDATE SET
            niche = EXCLUDED.niche,
            client_id = EXCLUDED.client_id,
            client_secret = EXCLUDED.client_secret,
            refresh_token = EXCLUDED.refresh_token,
            updated_at = NOW()
        RETURNING id
        "#,
        name,
        niche,
        client_id,
        client_secret,
        refresh_token
    )
    .fetch_one(pool)
    .await?;

    Ok(rec.id)
}
