use clap::Parser;
use axum::{
    extract::{Query, State},
    response::Html,
    routing::get,
    Router,
};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::info;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Account name (e.g. "Excelsior Academy")
    #[arg(short, long)]
    name: String,

    /// Niche name (e.g. "stoicism")
    #[arg(short, long)]
    niche: Option<String>,

    /// Google OAuth Client ID
    #[arg(long)]
    client_id: String,

    /// Google OAuth Client Secret
    #[arg(long)]
    client_secret: String,

    /// Local port for redirect URI (default: 8085)
    #[arg(short, long, default_value_t = 8085)]
    port: u16,
}

#[derive(Deserialize)]
struct AuthCallback {
    code: String,
}

struct AppState {
    args: Args,
    tx: mpsc::Sender<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    dotenvy::dotenv().ok();
    
    let args = Args::parse();
    let (tx, mut rx) = mpsc::channel(1);
    let state = Arc::new(AppState { args: args.clone(), tx });

    let app = Router::new()
        .route("/callback", get(callback))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], args.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    let auth_url = format!(
        "https://accounts.google.com/o/oauth2/v2/auth?client_id={}&redirect_uri=http://localhost:{}/callback&response_type=code&scope=https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly&access_type=offline&prompt=consent",
        args.client_id, args.port
    );

    println!("\n🚀 Animus Multi-Account Auth Helper");
    println!("=================================");
    println!("Registering account: {}", args.name);
    println!("Niche: {}", args.niche.as_deref().unwrap_or("none"));
    println!("\n1. Open this URL in your browser:\n\n{}\n", auth_url);
    println!("2. Authenticate with the desired Google account.");
    println!("3. Waiting for redirect...");

    // Run server in background
    let server_handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Wait for code from callback
    if let Some(code) = rx.recv().await {
        println!("Received authorization code. Exchanging for tokens...");
        
        let client = reqwest::Client::new();
        let params = [
            ("code", code),
            ("client_id", args.client_id.clone()),
            ("client_secret", args.client_secret.clone()),
            ("redirect_uri", format!("http://localhost:{}/callback", args.port)),
            ("grant_type", "authorization_code".to_string()),
        ];

        let res = client.post("https://oauth2.googleapis.com/token")
            .form(&params)
            .send()
            .await?;

        if !res.status().is_success() {
            eprintln!("Error exchanging code: {}", res.text().await?);
            return Ok(());
        }

        let token_data: serde_json::Value = res.json().await?;
        let refresh_token = token_data["refresh_token"].as_str().ok_or("No refresh token returned")?;

        // Save to DB
        let database_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
        let pool = sqlx::PgPool::connect(&database_url).await?;
        
        animus::db::accounts::upsert_account(
            &pool,
            &args.name,
            args.niche.as_deref(),
            &args.client_id,
            &args.client_secret,
            refresh_token
        ).await?;

        println!("\n✅ Success! Account '{}' has been added to the database.", args.name);
        println!("You can now use this account in Animus for niche production.");
    }

    server_handle.abort();
    Ok(())
}

async fn callback(
    Query(params): Query<AuthCallback>,
    State(state): State<Arc<AppState>>,
) -> Html<&'static str> {
    let _ = state.tx.send(params.code).await;
    Html("<h1>Authentication Successful!</h1><p>You can close this window and return to the terminal.</p>")
}
