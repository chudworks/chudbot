use std::path::Path;

use crate::VibeError;

/// Browser SDK served from every Vibe site host.
pub const SDK_V1_JAVASCRIPT: &str = include_str!("browser/vibe.js");

/// Canonical TypeScript declarations installed into every Vibe workspace.
pub const SDK_V1_TYPESCRIPT: &str = include_str!("browser/vibe.d.ts");

/// Install the declarations owned by this Chudbot binary into a site workspace.
pub async fn install_typescript_bindings(workspace: &Path) -> Result<(), VibeError> {
    let source = workspace.join("src");
    tokio::fs::create_dir_all(&source).await?;
    tokio::fs::write(source.join("vibe.d.ts"), SDK_V1_TYPESCRIPT).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn installing_bindings_replaces_a_stale_site_copy() {
        let workspace = std::env::temp_dir().join(format!(
            "vibe-browser-bindings-test-{}",
            uuid::Uuid::new_v4()
        ));
        tokio::fs::create_dir_all(workspace.join("src"))
            .await
            .unwrap();
        tokio::fs::write(workspace.join("src/vibe.d.ts"), "stale")
            .await
            .unwrap();
        install_typescript_bindings(&workspace).await.unwrap();
        assert_eq!(
            tokio::fs::read_to_string(workspace.join("src/vibe.d.ts"))
                .await
                .unwrap(),
            SDK_V1_TYPESCRIPT
        );
        let _ = tokio::fs::remove_dir_all(workspace).await;
    }
}
