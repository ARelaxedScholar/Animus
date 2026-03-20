use crate::nodes::{Claim, VerificationStatus, Script};
use crate::state_keys;
use async_trait::async_trait;
use orichalcum::llm::{Client as LlmClient, Enabled, Providers};
use orichalcum::{AsyncNodeLogic, NodeValue};
use serde_json::{json, Value};
use sqlx::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

pub use crate::config::FactCheckerConfig;

pub struct FactCheckerLogic {
    config: FactCheckerConfig,
    http_client: reqwest::Client,
    llm_client: Arc<LlmClient<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
    db_pool: Arc<PgPool>,
}

impl FactCheckerLogic {
    pub fn new(
        config: FactCheckerConfig,
        http_client: reqwest::Client,
        llm_client: Arc<LlmClient<Providers<orichalcum::llm::Disabled, Enabled, Enabled>>>,
        db_pool: Arc<PgPool>,
    ) -> Self {
        Self {
            config,
            http_client,
            llm_client,
            db_pool,
        }
    }

    async fn extract_claims(&self, script_text: &str) -> Result<Vec<Claim>, String> {
        let url = "https://api.groq.com/openai/v1/chat/completions";
        let body = json!({
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a claim extraction system. Given a video script, identify all factual claims that could be verified. For each claim, return: { \"id\": u32, \"sentence\": \"original sentence\", \"claim_text\": \"atomic claim\" }. Output as a JSON array."},
                {"role": "user", "content": script_text}
            ]
        });

        let resp = self.http_client.post(url)
            .header("Authorization", format!("Bearer {}", self.config.groq_api_key))
            .json(&body)
            .send().await.map_err(|e| e.to_string())?;
        
        if !resp.status().is_success() {
            return Err(format!("Groq API error: {}", resp.status()));
        }

        let json_resp: Value = resp.json().await.map_err(|e| e.to_string())?;
        let content = json_resp["choices"][0]["message"]["content"].as_str().ok_or("No content")?;
        
        let claims: Vec<Claim> = serde_json::from_str(content).map_err(|e| e.to_string())?;
        Ok(claims)
    }

    async fn search_evidence(&self, claim_text: &str) -> Result<Vec<String>, String> {
        let url = format!("https://duckduckgo.com/html/?q={}", urlencoding::encode(claim_text));
        
        let resp = self.http_client.get(url)
            .header("User-Agent", "Mozilla/5.0")
            .send().await.map_err(|e| e.to_string())?;
        
        if !resp.status().is_success() {
            return Err(format!("DDG search error: {}", resp.status()));
        }
        
        let html = resp.text().await.map_err(|e| e.to_string())?;
        // Basic extraction of snippets (in a real app, use a proper HTML parser)
        let snippets: Vec<String> = html.split("result__snippet").skip(1).take(3)
            .map(|s| s.split('>').nth(1).and_then(|s| s.split('<').next()).unwrap_or("").to_string())
            .collect();
            
        Ok(snippets)
    }

    async fn verify_claim(&self, claim: &Claim, evidence: &[String]) -> Result<VerificationStatus, String> {
        let system_prompt = "Given the following claim and search results, classify the claim: SUPPORTED, REFUTED, or NOT_VERIFIABLE. Respond with JSON: {\"status\": \"...\", \"reason\": \"...\"}";
        let user_prompt = format!("Claim: {}\nEvidence: {:?}", claim.claim_text, evidence);

        let response = self.llm_client.gemini_complete()
            .model("gemini-2.0-flash") // Assuming the model name is correct for the free tier
            .system(system_prompt)
            .user(&user_prompt)
            .await.map_err(|e| e.to_string())?;

        let json_resp: Value = serde_json::from_str(&response).map_err(|e| e.to_string())?;
        let status = json_resp["status"].as_str().unwrap_or("NOT_VERIFIABLE");
        
        match status {
            "SUPPORTED" => Ok(VerificationStatus::Supported),
            "REFUTED" => Ok(VerificationStatus::Refuted),
            _ => Ok(VerificationStatus::NotVerifiable),
        }
    }
}

#[async_trait]
impl AsyncNodeLogic for FactCheckerLogic {
    async fn prep(&self, _params: &HashMap<String, NodeValue>, shared: &HashMap<String, NodeValue>) -> NodeValue {
        let script = shared
            .get(state_keys::SCRIPT)
            .cloned()
            .unwrap_or(json!(null));
        json!({ "script": script })
    }

    async fn exec(&self, input: NodeValue) -> NodeValue {
        if !self.config.enabled {
            return input;
        }

        let mut script: Script = match serde_json::from_value(input["script"].clone()) {
            Ok(s) => s,
            Err(_) => return input,
        };

        info!("FactChecker: Verifying script {}", script.video_id);

        let claims_res = self.extract_claims(&script.full_text).await;
        
        if let Ok(claims) = claims_res {
            let mut refuted_sentences = Vec::new();
            
            for claim in claims {
                let evidence = self.search_evidence(&claim.claim_text).await.unwrap_or_default();
                let status: VerificationStatus = self.verify_claim(&claim, &evidence).await.unwrap_or(VerificationStatus::NotVerifiable);

                if status == VerificationStatus::Refuted || (status == VerificationStatus::NotVerifiable && !self.config.fail_open) {
                    refuted_sentences.push(claim.sentence.clone());
                }
            }
            
            // Surgical Cut: Remove refuted sentences
            let mut new_text = script.full_text.clone();
            for sentence in refuted_sentences {
                new_text = new_text.replace(&sentence, "");
            }
            script.full_text = new_text;
        } else if !self.config.fail_open {
             warn!("FactChecker: Failed to extract claims, halting production");
             return json!({ "error": "Fact-checking failed" });
        }
        
        json!({ "script": script })
    }

    async fn post(&self, shared: &mut HashMap<String, NodeValue>, _prep_res: NodeValue, exec_res: NodeValue) -> Option<String> {
        if let Some(script) = exec_res.get("script") {
            shared.insert(state_keys::SCRIPT.to_string(), script.clone());
        }
        Some("default".to_string())
    }

    fn clone_box(&self) -> Box<dyn AsyncNodeLogic> {
        Box::new(Self {
            config: self.config.clone(),
            http_client: self.http_client.clone(),
            llm_client: self.llm_client.clone(),
            db_pool: self.db_pool.clone(),
        })
    }
}
