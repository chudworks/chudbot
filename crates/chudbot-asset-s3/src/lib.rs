//! S3-compatible media storage for chudbot.
//!
//! This backend preserves chudbot's existing model-facing `file://...` media
//! handles while storing bytes in an S3 bucket. The AWS SDK client is built
//! lazily, so credentials and region are resolved through the standard AWS
//! environment/profile chains only when the backend is first used.

use std::fmt;
use std::sync::Arc;

use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::primitives::ByteStream;
use chudbot_api::{
    BoxedMediaRef, CreateMedia, LoadedMedia, MediaCategory, MediaError, MediaMetadata, MediaRef,
    MediaStore, MediaUri, PublicMediaUrl,
};
use tokio::sync::OnceCell;
use uuid::Uuid;

const FILE_SCHEME: &str = "file://";

/// S3-compatible media store.
#[derive(Debug, Clone)]
pub struct S3MediaStore {
    inner: Arc<S3MediaStoreInner>,
}

#[derive(Debug)]
struct S3MediaStoreInner {
    bucket: String,
    region: Option<String>,
    endpoint_url: Option<String>,
    force_path_style: bool,
    public_base_url: Option<String>,
    client: OnceCell<Client>,
}

impl S3MediaStore {
    /// Construct an S3 media store.
    ///
    /// `region` is optional so the AWS SDK can use `AWS_REGION` or the selected
    /// shared config profile. `endpoint_url` overrides the service endpoint for
    /// S3-compatible APIs such as MinIO, R2, or LocalStack.
    pub fn new(
        bucket: impl Into<String>,
        region: Option<String>,
        endpoint_url: Option<String>,
        force_path_style: bool,
        public_base_url: Option<String>,
    ) -> Self {
        let bucket = bucket.into();
        tracing::info!(
            bucket = %bucket,
            region = ?region,
            endpoint_url_override = endpoint_url.is_some(),
            force_path_style,
            has_public_base_url = public_base_url.is_some(),
            "created S3 media store"
        );
        Self {
            inner: Arc::new(S3MediaStoreInner {
                bucket,
                region,
                endpoint_url,
                force_path_style,
                public_base_url,
                client: OnceCell::new(),
            }),
        }
    }

    /// Bucket configured for media objects.
    pub fn bucket(&self) -> &str {
        &self.inner.bucket
    }

    #[tracing::instrument(name = "s3_media.client", skip_all)]
    async fn client(&self) -> Result<&Client, MediaError> {
        self.inner
            .client
            .get_or_try_init(|| async { self.build_client().await })
            .await
    }

    async fn build_client(&self) -> Result<Client, MediaError> {
        let mut loader = aws_config::defaults(BehaviorVersion::latest());
        if let Some(region) = self.inner.region.clone() {
            loader = loader.region(Region::new(region));
        }
        let sdk_config = loader.load().await;
        let mut builder = aws_sdk_s3::config::Builder::from(&sdk_config)
            .force_path_style(self.inner.force_path_style);
        if let Some(endpoint_url) = &self.inner.endpoint_url {
            builder = builder.endpoint_url(endpoint_url.clone());
        }
        tracing::debug!(
            bucket = %self.inner.bucket,
            region = ?sdk_config.region().map(|region| region.as_ref()),
            endpoint_url_override = self.inner.endpoint_url.is_some(),
            force_path_style = self.inner.force_path_style,
            "built S3 media client"
        );
        Ok(Client::from_conf(builder.build()))
    }

    fn media_ref(
        &self,
        category: MediaCategory,
        name: String,
        mime_type: String,
        size_bytes: u64,
    ) -> BoxedMediaRef {
        let uri = media_uri(&category, &name);
        Box::new(S3MediaRef {
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

    fn object_key(&self, category: &MediaCategory, name: &str) -> Result<String, MediaError> {
        validate_media_name(name)?;
        match category {
            MediaCategory::Image
            | MediaCategory::Video
            | MediaCategory::Audio
            | MediaCategory::Avatar
            | MediaCategory::GuildIcon => Ok(format!("{}/{name}", category.prefix())),
            MediaCategory::Other(prefix) => Err(MediaError::UnsupportedCategory(prefix.clone())),
        }
    }

    #[tracing::instrument(
        name = "s3_media.load",
        skip_all,
        fields(uri = %metadata.uri, category = ?metadata.category, name = %metadata.name)
    )]
    async fn load_s3_media(&self, metadata: &MediaMetadata) -> Result<LoadedMedia, MediaError> {
        let canonical = self.media_from_uri(&metadata.uri).await?;
        let key = self.object_key(canonical.category(), canonical.name())?;
        let client = self.client().await?;
        let object = client
            .get_object()
            .bucket(&self.inner.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|error| s3_error("get object", error))?;
        let bytes = object
            .body
            .collect()
            .await
            .map_err(|error| s3_error("read object body", error))?
            .into_bytes()
            .to_vec();
        tracing::debug!(
            bucket = %self.inner.bucket,
            key = %key,
            bytes = bytes.len(),
            mime_type = canonical.mime_type(),
            "loaded S3 media"
        );
        Ok(LoadedMedia {
            media: canonical,
            bytes,
        })
    }

    #[tracing::instrument(name = "s3_media.public_url", skip_all, fields(uri = %uri))]
    async fn public_url_for_uri(&self, uri: &MediaUri) -> Result<PublicMediaUrl, MediaError> {
        let Some(public_url) = public_url_from_base(self.inner.public_base_url.as_deref(), uri)
        else {
            tracing::debug!("no public URL configured for S3 media URI");
            return Err(MediaError::NoPublicUrl { uri: uri.clone() });
        };
        tracing::trace!(public_url = %public_url, "resolved S3 media public URL");
        Ok(public_url)
    }
}

impl MediaStore for S3MediaStore {
    #[tracing::instrument(
        name = "s3_media.create",
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
        let key = self.object_key(&input.category, &name)?;
        let size_bytes = input.bytes.len() as u64;
        let client = self.client().await?;
        client
            .put_object()
            .bucket(&self.inner.bucket)
            .key(&key)
            .content_type(mime_type.clone())
            .body(ByteStream::from(input.bytes))
            .send()
            .await
            .map_err(|error| s3_error("put object", error))?;
        tracing::info!(
            bucket = %self.inner.bucket,
            key = %key,
            category = ?input.category,
            name = %name,
            mime_type = %mime_type,
            bytes = size_bytes,
            "created S3 media"
        );

        Ok(self.media_ref(input.category, name, mime_type, size_bytes))
    }

    #[tracing::instrument(name = "s3_media.from_uri", skip_all, fields(uri = %uri))]
    async fn media_from_uri(&self, uri: &MediaUri) -> Result<BoxedMediaRef, MediaError> {
        let ParsedStoredUri { category, name } = parse_stored_uri(uri)?;
        tracing::trace!(category = ?category, name = %name, "parsed S3 media URI");
        self.media_from_name(category, &name).await
    }

    #[tracing::instrument(
        name = "s3_media.from_name",
        skip_all,
        fields(category = ?category, name = %name)
    )]
    async fn media_from_name(
        &self,
        category: MediaCategory,
        name: &str,
    ) -> Result<BoxedMediaRef, MediaError> {
        let key = self.object_key(&category, name)?;
        let client = self.client().await?;
        let metadata = client
            .head_object()
            .bucket(&self.inner.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|error| s3_error("head object", error))?;
        let mime_type = metadata
            .content_type()
            .and_then(normalize_mime_type)
            .unwrap_or_else(|| {
                mime_for_category_extension(&category, extension_from_name(name)).to_string()
            });
        let size_bytes = metadata
            .content_length()
            .and_then(|bytes| u64::try_from(bytes).ok())
            .unwrap_or(0);
        tracing::debug!(
            bucket = %self.inner.bucket,
            key = %key,
            mime_type = %mime_type,
            bytes = size_bytes,
            "loaded S3 media metadata"
        );
        Ok(self.media_ref(category, name.to_string(), mime_type, size_bytes))
    }
}

/// S3 media handle.
#[derive(Debug, Clone)]
struct S3MediaRef {
    store: S3MediaStore,
    metadata: MediaMetadata,
}

#[async_trait::async_trait]
impl MediaRef for S3MediaRef {
    fn metadata(&self) -> &MediaMetadata {
        &self.metadata
    }

    fn clone_box(&self) -> BoxedMediaRef {
        Box::new(self.clone())
    }

    async fn public_url(&self) -> Result<PublicMediaUrl, MediaError> {
        self.store.public_url_for_uri(&self.metadata.uri).await
    }

    async fn load(&self) -> Result<LoadedMedia, MediaError> {
        self.store.load_s3_media(&self.metadata).await
    }
}

#[derive(Debug)]
struct ParsedStoredUri {
    category: MediaCategory,
    name: String,
}

fn parse_stored_uri(uri: &MediaUri) -> Result<ParsedStoredUri, MediaError> {
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
        "audio" => MediaCategory::Audio,
        "avatars" => MediaCategory::Avatar,
        "guild-icons" => MediaCategory::GuildIcon,
        _ => return Err(MediaError::UnsupportedUri(uri.to_string())),
    };
    Ok(ParsedStoredUri {
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
    path.starts_with("images/")
        || path.starts_with("videos/")
        || path.starts_with("audio/")
        || path.starts_with("avatars/")
        || path.starts_with("guild-icons/")
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
    let extension_mime = mime_for_category_extension(&input.category, extension);
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
    if bytes.starts_with(b"OggS") {
        return Some("audio/ogg");
    }
    if bytes.starts_with(b"ID3")
        || bytes
            .get(0..2)
            .is_some_and(|prefix| prefix[0] == 0xff && prefix[1] & 0xe0 == 0xe0)
    {
        return Some("audio/mpeg");
    }
    if bytes.len() >= 12 && bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE" {
        return Some("audio/wav");
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

fn extension_for_mime(mime: &str) -> &'static str {
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
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/ogg" => "ogg",
        "audio/opus" => "opus",
        "audio/webm" => "webm",
        "audio/mp4" | "audio/m4a" => "m4a",
        "audio/aac" => "aac",
        "audio/flac" => "flac",
        _ => "bin",
    }
}

fn mime_for_extension(extension: Option<&str>) -> &'static str {
    match extension.unwrap_or("").to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "heic" | "heif" => "image/heic",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "opus" => "audio/opus",
        "m4a" => "audio/m4a",
        "aac" => "audio/aac",
        "flac" => "audio/flac",
        _ => "application/octet-stream",
    }
}

fn mime_for_category_extension(category: &MediaCategory, extension: Option<&str>) -> &'static str {
    let normalized = extension.unwrap_or("").to_ascii_lowercase();
    if matches!(category, MediaCategory::Audio) {
        return match normalized.as_str() {
            "mp4" => "audio/mp4",
            "webm" => "audio/webm",
            "mkv" => "audio/x-matroska",
            _ => mime_for_extension(Some(&normalized)),
        };
    }
    mime_for_extension(Some(&normalized))
}

fn s3_error(operation: &'static str, error: impl fmt::Display) -> MediaError {
    MediaError::Io(format!("s3 {operation} failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_url_maps_file_uri_to_served_path() {
        let url = public_url_from_base(
            Some("https://assets.chudbot.example.com/"),
            &MediaUri::new("file://images/abc.png"),
        )
        .unwrap();
        assert_eq!(
            url.as_str(),
            "https://assets.chudbot.example.com/images/abc.png"
        );
    }

    #[test]
    fn rejects_path_traversal() {
        let uri = MediaUri::new("file://images/../secret.png");
        assert!(matches!(
            parse_stored_uri(&uri),
            Err(MediaError::UnsupportedUri(_))
        ));
    }

    #[test]
    fn object_key_uses_category_prefix() {
        let store = S3MediaStore::new(
            "assets",
            Some("us-east-1".to_string()),
            Some("https://s3.example.test".to_string()),
            true,
            Some("https://assets.example.test".to_string()),
        );
        assert_eq!(
            store.object_key(&MediaCategory::Image, "abc.png").unwrap(),
            "images/abc.png"
        );
        assert_eq!(
            store
                .object_key(&MediaCategory::Avatar, "user.webp")
                .unwrap(),
            "avatars/user.webp"
        );
    }

    #[test]
    fn explicit_mime_type_is_normalized() {
        let input = CreateMedia {
            category: MediaCategory::Image,
            bytes: b"not actually jpeg".to_vec(),
            mime_type: Some("image/jpeg; charset=binary".to_string()),
            name: Some("sample.bin".to_string()),
            extension: None,
        };

        assert_eq!(detect_mime_type(&input), "image/jpeg");
    }
}
