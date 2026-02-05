//! S3-compatible storage client for assets and videos

use aws_config::BehaviorVersion;
use aws_sdk_s3::{
    config::{Credentials, Region},
    primitives::ByteStream,
    Client,
};
use std::path::Path;

/// S3-compatible storage client
#[derive(Clone)]
pub struct S3Client {
    client: Client,
    bucket: String,
}

impl S3Client {
    /// Create a new S3 client from configuration
    pub async fn new(
        endpoint: &str,
        access_key: &str,
        secret_key: &str,
        bucket: &str,
        region: &str,
    ) -> Result<Self, String> {
        let creds = Credentials::new(access_key, secret_key, None, None, "animus");

        let config = aws_sdk_s3::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .credentials_provider(creds)
            .region(Region::new(region.to_string()))
            .endpoint_url(endpoint)
            .force_path_style(true) // Required for MinIO
            .build();

        let client = Client::from_conf(config);

        Ok(Self {
            client,
            bucket: bucket.to_string(),
        })
    }

    /// Upload bytes to S3
    pub async fn upload_bytes(
        &self,
        key: &str,
        data: Vec<u8>,
        content_type: &str,
    ) -> Result<(), String> {
        let body = ByteStream::from(data);
        self.upload_stream(key, body, content_type).await
    }

    /// Upload a stream to S3
    pub async fn upload_stream(
        &self,
        key: &str,
        stream: ByteStream,
        content_type: &str,
    ) -> Result<(), String> {
        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(stream)
            .content_type(content_type)
            .send()
            .await
            .map_err(|e| format!("S3 upload failed: {}", e))?;

        Ok(())
    }

    /// Upload a local file to S3
    pub async fn upload_file(
        &self,
        local_path: &str,
        key: &str,
        content_type: &str,
    ) -> Result<(), String> {
        let data = tokio::fs::read(local_path)
            .await
            .map_err(|e| format!("Failed to read file: {}", e))?;

        self.upload_bytes(key, data, content_type).await
    }

    /// Download bytes from S3
    pub async fn download_bytes(&self, key: &str) -> Result<Vec<u8>, String> {
        let response = self.client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| format!("S3 download failed: {}", e))?;

        let data = response.body
            .collect()
            .await
            .map_err(|e| format!("Failed to read S3 body: {}", e))?
            .into_bytes()
            .to_vec();

        Ok(data)
    }

    /// Download from S3 to a local file
    pub async fn download_to_file(&self, key: &str, local_path: &str) -> Result<(), String> {
        let data = self.download_bytes(key).await?;

        // Create parent directories if needed
        if let Some(parent) = Path::new(local_path).parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| format!("Failed to create directories: {}", e))?;
        }

        tokio::fs::write(local_path, data)
            .await
            .map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(())
    }

    /// Delete an object from S3
    pub async fn delete(&self, key: &str) -> Result<(), String> {
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| format!("S3 delete failed: {}", e))?;

        Ok(())
    }

    /// Check if an object exists
    pub async fn exists(&self, key: &str) -> bool {
        self.client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .is_ok()
    }

    /// List objects with a prefix
    pub async fn list(&self, prefix: &str) -> Result<Vec<String>, String> {
        let response = self.client
            .list_objects_v2()
            .bucket(&self.bucket)
            .prefix(prefix)
            .send()
            .await
            .map_err(|e| format!("S3 list failed: {}", e))?;

        let keys = response
            .contents()
            .iter()
            .filter_map(|obj| obj.key().map(|s| s.to_string()))
            .collect();

        Ok(keys)
    }
}
