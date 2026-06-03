//! Local filesystem media storage for chudbot.
//!
//! This backend owns `file://images/...`, `file://videos/...`, and
//! `file://avatars/...` URIs. The URI is stable and model-facing; the public
//! URL is deployment-facing and can point at the Axum static routes today.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use chudbot_api::{
    BoxedMediaRef, CreateMedia, LoadedMedia, MediaCategory, MediaError, MediaFuture, MediaMetadata,
    MediaRef, MediaStore, MediaUri, PublicMediaUrl,
};
use uuid::Uuid;

const FILE_SCHEME: &str = "file://";

/// Local filesystem media store.
#[derive(Debug, Clone)]
pub struct LocalMediaStore {
    inner: Arc<LocalMediaStoreInner>,
}

#[derive(Debug)]
struct LocalMediaStoreInner {
    images_dir: PathBuf,
    videos_dir: PathBuf,
    avatars_dir: PathBuf,
    public_base_url: Option<String>,
}

impl LocalMediaStore {
    /// Construct a local media store.
    pub fn new(
        images_dir: impl Into<PathBuf>,
        videos_dir: impl Into<PathBuf>,
        avatars_dir: impl Into<PathBuf>,
        public_base_url: Option<String>,
    ) -> Self {
        let images_dir = images_dir.into();
        let videos_dir = videos_dir.into();
        let avatars_dir = avatars_dir.into();
        tracing::info!(
            images_dir = %images_dir.display(),
            videos_dir = %videos_dir.display(),
            avatars_dir = %avatars_dir.display(),
            has_public_base_url = public_base_url.is_some(),
            "created local media store"
        );
        Self {
            inner: Arc::new(LocalMediaStoreInner {
                images_dir,
                videos_dir,
                avatars_dir,
                public_base_url,
            }),
        }
    }

    /// Directory for image media.
    pub fn images_dir(&self) -> &Path {
        &self.inner.images_dir
    }

    /// Directory for video media.
    pub fn videos_dir(&self) -> &Path {
        &self.inner.videos_dir
    }

    /// Directory for avatar media.
    pub fn avatars_dir(&self) -> &Path {
        &self.inner.avatars_dir
    }

    /// Convert a resolved local media reference to an on-disk path.
    pub fn local_path(&self, media: &dyn MediaRef) -> Result<PathBuf, MediaError> {
        let path = self.path_for_name(media.category(), media.name())?;
        tracing::trace!(
            category = ?media.category(),
            name = media.name(),
            path = %path.display(),
            "resolved local media path"
        );
        Ok(path)
    }

    /// Convert a local file media URI to an on-disk path.
    pub fn local_path_from_uri(&self, uri: &MediaUri) -> Result<PathBuf, MediaError> {
        let ParsedLocalUri { category, name } = parse_local_uri(uri)?;
        let path = self.path_for_name(&category, &name)?;
        tracing::trace!(
            uri = %uri,
            category = ?category,
            name = %name,
            path = %path.display(),
            "resolved local media URI path"
        );
        Ok(path)
    }

    fn media_ref(
        &self,
        category: MediaCategory,
        name: String,
        mime_type: String,
        size_bytes: u64,
    ) -> BoxedMediaRef {
        let uri = media_uri(&category, &name);
        Box::new(LocalMediaRef {
            store: self.clone(),
            metadata: MediaMetadata {
                category,
                name,
                uri,
                mime_type,
                size_bytes,
            },
        })
    }

    fn path_for_name(&self, category: &MediaCategory, name: &str) -> Result<PathBuf, MediaError> {
        validate_media_name(name)?;
        Ok(self.base_dir(category)?.join(name))
    }

    fn base_dir(&self, category: &MediaCategory) -> Result<&Path, MediaError> {
        match category {
            MediaCategory::Image => Ok(&self.inner.images_dir),
            MediaCategory::Video => Ok(&self.inner.videos_dir),
            MediaCategory::Avatar => Ok(&self.inner.avatars_dir),
            MediaCategory::Other(prefix) => Err(MediaError::UnsupportedCategory(prefix.clone())),
        }
    }

    #[tracing::instrument(
        name = "local_media.load",
        skip_all,
        fields(uri = %metadata.uri, category = ?metadata.category, name = %metadata.name)
    )]
    async fn load_local_media(&self, metadata: &MediaMetadata) -> Result<LoadedMedia, MediaError> {
        let canonical = self.media_from_uri(&metadata.uri).await?;
        let path = self.local_path(canonical.as_ref())?;
        let bytes = tokio::fs::read(&path).await?;
        tracing::debug!(
            path = %path.display(),
            bytes = bytes.len(),
            mime_type = canonical.mime_type(),
            "loaded local media"
        );
        Ok(LoadedMedia {
            media: canonical,
            bytes,
        })
    }

    #[tracing::instrument(name = "local_media.public_url", skip_all, fields(uri = %uri))]
    async fn public_url_for_uri(&self, uri: &MediaUri) -> Result<PublicMediaUrl, MediaError> {
        let Some(public_url) = public_url_from_base(self.inner.public_base_url.as_deref(), uri)
        else {
            tracing::debug!("no public URL configured for local media URI");
            return Err(MediaError::NoPublicUrl { uri: uri.clone() });
        };
        tracing::trace!(public_url = %public_url, "resolved local media public URL");
        Ok(public_url)
    }
}

impl MediaStore for LocalMediaStore {
    #[tracing::instrument(
        name = "local_media.create",
        skip_all,
        fields(category = ?input.category, requested_name = ?input.name, bytes = input.bytes.len())
    )]
    async fn create_media(&self, input: CreateMedia) -> Result<BoxedMediaRef, MediaError> {
        let mime_type = detect_mime_type(&input);
        let name = match input.name {
            Some(name) => {
                validate_media_name(&name)?;
                name
            }
            None => generated_name(input.extension.as_deref(), &mime_type),
        };
        let base_dir = self.base_dir(&input.category)?;
        tokio::fs::create_dir_all(base_dir).await?;
        let path = base_dir.join(&name);
        tokio::fs::write(&path, &input.bytes).await?;
        tracing::info!(
            category = ?input.category,
            name = %name,
            path = %path.display(),
            mime_type = %mime_type,
            bytes = input.bytes.len(),
            "created local media"
        );

        Ok(self.media_ref(input.category, name, mime_type, input.bytes.len() as u64))
    }

    #[tracing::instrument(name = "local_media.from_uri", skip_all, fields(uri = %uri))]
    async fn media_from_uri(&self, uri: &MediaUri) -> Result<BoxedMediaRef, MediaError> {
        let ParsedLocalUri { category, name } = parse_local_uri(uri)?;
        tracing::trace!(category = ?category, name = %name, "parsed local media URI");
        self.media_from_name(category, &name).await
    }

    #[tracing::instrument(
        name = "local_media.from_name",
        skip_all,
        fields(category = ?category, name = %name)
    )]
    async fn media_from_name(
        &self,
        category: MediaCategory,
        name: &str,
    ) -> Result<BoxedMediaRef, MediaError> {
        let path = self.path_for_name(&category, name)?;
        let metadata = tokio::fs::metadata(&path).await?;
        let mime_type = mime_for_extension(path.extension().and_then(|s| s.to_str())).to_string();
        tracing::debug!(
            path = %path.display(),
            mime_type = %mime_type,
            bytes = metadata.len(),
            "loaded local media metadata"
        );
        Ok(self.media_ref(category, name.to_string(), mime_type, metadata.len()))
    }
}

/// Local filesystem media handle.
#[derive(Debug, Clone)]
struct LocalMediaRef {
    store: LocalMediaStore,
    metadata: MediaMetadata,
}

impl MediaRef for LocalMediaRef {
    fn metadata(&self) -> &MediaMetadata {
        &self.metadata
    }

    fn clone_box(&self) -> BoxedMediaRef {
        Box::new(self.clone())
    }

    fn public_url(&self) -> MediaFuture<'_, PublicMediaUrl> {
        Box::pin(async move { self.store.public_url_for_uri(&self.metadata.uri).await })
    }

    fn load(&self) -> MediaFuture<'_, LoadedMedia> {
        Box::pin(async move { self.store.load_local_media(&self.metadata).await })
    }
}

#[derive(Debug)]
struct ParsedLocalUri {
    category: MediaCategory,
    name: String,
}

fn parse_local_uri(uri: &MediaUri) -> Result<ParsedLocalUri, MediaError> {
    let path = uri
        .as_str()
        .strip_prefix(FILE_SCHEME)
        .ok_or_else(|| MediaError::UnsupportedUri(uri.to_string()))?;
    let (prefix, name) = path
        .split_once('/')
        .ok_or_else(|| MediaError::UnsupportedUri(uri.to_string()))?;
    validate_media_name(name).map_err(|_| MediaError::UnsupportedUri(uri.to_string()))?;
    let category = match prefix {
        "images" => MediaCategory::Image,
        "videos" => MediaCategory::Video,
        "avatars" => MediaCategory::Avatar,
        _ => return Err(MediaError::UnsupportedUri(uri.to_string())),
    };
    Ok(ParsedLocalUri {
        category,
        name: name.to_string(),
    })
}

fn media_uri(category: &MediaCategory, name: &str) -> MediaUri {
    MediaUri::new(format!("{FILE_SCHEME}{}/{name}", category.prefix()))
}

fn public_url_from_base(base: Option<&str>, uri: &MediaUri) -> Option<PublicMediaUrl> {
    let path = uri.as_str().strip_prefix(FILE_SCHEME)?;
    if !is_supported_prefix(path) {
        return None;
    }
    let base = base?.trim_end_matches('/');
    Some(PublicMediaUrl::new(format!("{base}/{path}")))
}

fn is_supported_prefix(path: &str) -> bool {
    path.starts_with("images/") || path.starts_with("videos/") || path.starts_with("avatars/")
}

fn generated_name(extension: Option<&str>, mime_type: &str) -> String {
    let extension = extension
        .map(normalize_extension)
        .filter(|extension| !extension.is_empty())
        .unwrap_or_else(|| extension_for_mime(mime_type).to_string());
    format!("{}.{}", Uuid::new_v4().simple(), extension)
}

fn detect_mime_type(input: &CreateMedia) -> String {
    if let Some(mime_type) = input.mime_type.as_deref().and_then(normalize_mime_type) {
        return mime_type;
    }

    let extension = input
        .extension
        .as_deref()
        .or_else(|| input.name.as_deref().and_then(extension_from_name));
    let extension_mime = mime_for_extension(extension);
    if extension_mime != "application/octet-stream" {
        return extension_mime.to_string();
    }

    detect_mime_from_bytes(&input.bytes)
        .unwrap_or("application/octet-stream")
        .to_string()
}

fn normalize_mime_type(mime_type: &str) -> Option<String> {
    let mime_type = mime_type
        .split(';')
        .next()
        .unwrap_or(mime_type)
        .trim()
        .to_ascii_lowercase();
    (!mime_type.is_empty()).then_some(mime_type)
}

fn extension_from_name(name: &str) -> Option<&str> {
    name.rsplit_once('.')
        .and_then(|(_, extension)| (!extension.is_empty()).then_some(extension))
}

fn detect_mime_from_bytes(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Some("image/png");
    }
    if bytes.starts_with(b"\xff\xd8\xff") {
        return Some("image/jpeg");
    }
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return Some("image/gif");
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WEBP" {
        return Some("image/webp");
    }
    if bytes.starts_with(b"\x1a\x45\xdf\xa3") {
        return Some("video/webm");
    }
    if bytes.len() >= 12 && &bytes[4..8] == b"ftyp" {
        return Some(mime_for_iso_base_media_brand(&bytes[8..12]));
    }
    None
}

fn mime_for_iso_base_media_brand(brand: &[u8]) -> &'static str {
    match brand {
        b"heic" | b"heix" | b"hevc" | b"hevx" | b"mif1" | b"msf1" => "image/heic",
        b"qt  " => "video/quicktime",
        _ => "video/mp4",
    }
}

fn validate_media_name(name: &str) -> Result<(), MediaError> {
    if name.is_empty()
        || name.contains('/')
        || name.contains('\\')
        || name == "."
        || name == ".."
        || name.chars().any(char::is_control)
    {
        return Err(MediaError::UnsafeName(name.to_string()));
    }
    Ok(())
}

fn normalize_extension(extension: &str) -> String {
    extension
        .trim_start_matches('.')
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase()
}

/// Map a MIME type to a file extension.
pub fn extension_for_mime(mime: &str) -> &'static str {
    match mime
        .split(';')
        .next()
        .unwrap_or(mime)
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/gif" => "gif",
        "image/webp" => "webp",
        "image/heic" | "image/heif" => "heic",
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        "video/quicktime" => "mov",
        _ => "bin",
    }
}

/// Guess a MIME type from a filename extension.
pub fn mime_for_extension(extension: Option<&str>) -> &'static str {
    match extension.unwrap_or("").to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "heic" | "heif" => "image/heic",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn public_url_maps_file_uri_to_served_path() {
        let url = public_url_from_base(
            Some("https://chudbot.example.com/"),
            &MediaUri::new("file://images/abc.png"),
        )
        .unwrap();
        assert_eq!(url.as_str(), "https://chudbot.example.com/images/abc.png");
    }

    #[test]
    fn rejects_path_traversal() {
        let uri = MediaUri::new("file://images/../secret.png");
        assert!(matches!(
            parse_local_uri(&uri),
            Err(MediaError::UnsupportedUri(_))
        ));
    }

    #[tokio::test]
    async fn creates_resolves_and_loads_media() {
        let root = std::env::temp_dir().join(format!(
            "chudbot-local-media-test-{}",
            Uuid::new_v4().simple()
        ));
        let store = LocalMediaStore::new(
            root.join("images"),
            root.join("videos"),
            root.join("avatars"),
            Some("https://chudbot.example.com".to_string()),
        );

        let media = store
            .create_media(CreateMedia {
                category: MediaCategory::Image,
                bytes: b"image bytes".to_vec(),
                mime_type: None,
                name: Some("sample.png".to_string()),
                extension: None,
            })
            .await
            .unwrap();

        assert_eq!(media.category(), &MediaCategory::Image);
        assert_eq!(media.name(), "sample.png");
        assert_eq!(media.mime_type(), "image/png");
        assert_eq!(media.size_bytes(), 11);
        assert_eq!(media.uri().as_str(), "file://images/sample.png");
        assert_eq!(
            media.public_url().await.unwrap().as_str(),
            "https://chudbot.example.com/images/sample.png"
        );

        let by_uri = store.media_from_uri(media.uri()).await.unwrap();
        assert_eq!(by_uri.name(), "sample.png");

        let by_name = store
            .media_from_name(MediaCategory::Image, "sample.png")
            .await
            .unwrap();
        assert_eq!(by_name.uri().as_str(), "file://images/sample.png");

        let loaded = media.load().await.unwrap();
        assert_eq!(loaded.bytes, b"image bytes");
        assert_eq!(loaded.media.uri().as_str(), "file://images/sample.png");

        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn public_url_errors_when_not_configured() {
        let store = LocalMediaStore::new("images", "videos", "avatars", None);
        let media = store
            .media_from_name(MediaCategory::Image, "abc.png")
            .await
            .unwrap_err();
        assert!(matches!(media, MediaError::Io(_)));

        let media = LocalMediaRef {
            store,
            metadata: MediaMetadata {
                category: MediaCategory::Image,
                name: "abc.png".to_string(),
                uri: MediaUri::new("file://images/abc.png"),
                mime_type: "image/png".to_string(),
                size_bytes: 10,
            },
        };
        assert!(matches!(
            media.public_url().await,
            Err(MediaError::NoPublicUrl { .. })
        ));
    }

    #[tokio::test]
    async fn detects_mime_type_from_bytes_when_name_has_no_extension() {
        let root = std::env::temp_dir().join(format!(
            "chudbot-local-media-test-{}",
            Uuid::new_v4().simple()
        ));
        let store = LocalMediaStore::new(
            root.join("images"),
            root.join("videos"),
            root.join("avatars"),
            None,
        );

        let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
        bytes.extend_from_slice(b"png payload");
        let media = store
            .create_media(CreateMedia {
                category: MediaCategory::Image,
                bytes,
                mime_type: None,
                name: Some("sample".to_string()),
                extension: None,
            })
            .await
            .unwrap();

        assert_eq!(media.mime_type(), "image/png");

        fs::remove_dir_all(root).ok();
    }

    #[tokio::test]
    async fn explicit_mime_type_is_used_as_override() {
        let root = std::env::temp_dir().join(format!(
            "chudbot-local-media-test-{}",
            Uuid::new_v4().simple()
        ));
        let store = LocalMediaStore::new(
            root.join("images"),
            root.join("videos"),
            root.join("avatars"),
            None,
        );

        let media = store
            .create_media(CreateMedia {
                category: MediaCategory::Image,
                bytes: b"not actually jpeg".to_vec(),
                mime_type: Some("image/jpeg; charset=binary".to_string()),
                name: Some("sample.bin".to_string()),
                extension: None,
            })
            .await
            .unwrap();

        assert_eq!(media.mime_type(), "image/jpeg");

        fs::remove_dir_all(root).ok();
    }
}
