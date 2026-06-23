//! Span-aware config diagnostics for `check-config`.
//!
//! Syntax, type, and missing-field failures are emitted by the TOML
//! parser/deserializer before a `RuntimeConfig` exists. Once deserialization
//! succeeds, this module aggregates semantic failures and stale/unknown config
//! keys into one compiler-style report. Keeping unknown-key checks here instead
//! of `#[serde(deny_unknown_fields)]` preserves the richer diagnostic path:
//! users can see all related config problems, with source labels, in one run.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error as _;
use std::fmt;
use std::fmt::Write as _;
use std::io::IsTerminal;
use std::net::SocketAddr;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use chudbot_api::{ProviderName, SamplingNumber};
use chudbot_bot::{GenerationBinding, TranscriptionBinding, VideoGenerationRateLimit};
use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::config::{
    AudioProviderConfig, ImageProviderConfig, LlmProviderConfig, MessagePlatformConfig,
    RuntimeConfig, StorageConfig, VideoProviderConfig,
};

const DATETIME_FIELD: &str = "$__toml_private_datetime";
const MAX_SOURCE_LINE_CHARS: usize = 160;
const SOURCE_WINDOW_CHARS: usize = 120;
const ANSI_RESET: &str = "\x1b[0m";
const ANSI_BOLD_RED: &str = "\x1b[1;31m";
const ANSI_BLUE: &str = "\x1b[34m";
const ANSI_GREEN: &str = "\x1b[32m";
const ANSI_BOLD: &str = "\x1b[1m";

/// Original TOML plus a shape-only, spanned tree used for diagnostic labels.
///
/// Runtime code reads the typed `RuntimeConfig`; validators use this companion
/// tree only to find related source ranges for keys, tables, arrays, and array
/// items.
#[derive(Debug, Clone)]
pub(crate) struct ConfigSource {
    path: PathBuf,
    input: String,
    root: Option<toml::Spanned<SourceNode>>,
    lines: LineIndex,
}

impl ConfigSource {
    pub(crate) fn new(path: PathBuf, input: String) -> Self {
        // Best effort: parse/type errors are reported through the normal TOML
        // path. If this tree is unavailable, semantic validation can still fall
        // back to broad source ranges instead of changing config behavior.
        let root = toml::from_str::<toml::Spanned<SourceNode>>(&input).ok();
        let lines = LineIndex::new(&input);
        Self {
            path,
            input,
            root,
            lines,
        }
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn input(&self) -> &str {
        &self.input
    }

    pub(crate) fn source_for_keys(&self, keys: &[&str]) -> Option<&str> {
        let path = keys
            .iter()
            .map(|part| PathPart::Key(*part))
            .collect::<Vec<_>>();
        let span = self.span_for(&path)?;
        self.input.get(span)
    }

    // Walk a logical config path through the spanned source tree. Quoted TOML
    // keys that contain dots remain one `PathPart::Key`.
    fn node_for(&self, path: &[PathPart<'_>]) -> Option<&toml::Spanned<SourceNode>> {
        let mut value = self.root.as_ref()?;
        for part in path {
            value = match (value.get_ref(), part) {
                (SourceNode::Table(entries), PathPart::Key(key)) => entries.get(*key)?,
                (SourceNode::Array(items), PathPart::Index(index)) => items.get(*index)?,
                _ => return None,
            };
        }
        Some(value)
    }

    fn table_at(
        &self,
        path: &[PathPart<'_>],
    ) -> Option<&BTreeMap<String, toml::Spanned<SourceNode>>> {
        self.node_for(path)?.get_ref().as_table()
    }

    fn array_at(&self, path: &[PathPart<'_>]) -> Option<&[toml::Spanned<SourceNode>]> {
        self.node_for(path)?.get_ref().as_array()
    }

    fn span_for(&self, path: &[PathPart<'_>]) -> Option<Range<usize>> {
        self.node_for(path).map(toml::Spanned::span)
    }

    // TOML and serde do not always expose the exact key token that caused a
    // semantic error. Prefer the requested value span, then walk outward until a
    // related table/array span is available.
    fn nearest_span_for(&self, path: &[PathPart<'_>]) -> Option<Range<usize>> {
        for len in (0..=path.len()).rev() {
            if let Some(span) = self.span_for(&path[..len]) {
                return Some(span);
            }
        }
        self.root.as_ref().map(toml::Spanned::span)
    }

    fn primary_label(
        &self,
        path: &[PathPart<'_>],
        fallback_path: &[PathPart<'_>],
        message: impl Into<String>,
    ) -> DiagnosticLabel {
        // Most validators know the ideal field path plus a containing table
        // that is still useful when the field is absent or represented through
        // an inline table.
        let span = self
            .span_for(path)
            .or_else(|| self.nearest_span_for(fallback_path))
            .unwrap_or(0..self.input.len().min(1));
        DiagnosticLabel {
            span,
            message: message.into(),
        }
    }
}

/// Minimal TOML shape preserved solely for source-span lookup.
///
/// Scalars collapse to one variant because validation uses the typed
/// `RuntimeConfig` for values. The tree keeps only container boundaries and
/// child names so diagnostics can point back into the user's file.
#[derive(Debug, Clone)]
enum SourceNode {
    Scalar,
    Array(Vec<toml::Spanned<SourceNode>>),
    Table(BTreeMap<String, toml::Spanned<SourceNode>>),
}

impl SourceNode {
    fn as_table(&self) -> Option<&BTreeMap<String, toml::Spanned<SourceNode>>> {
        match self {
            Self::Table(entries) => Some(entries),
            Self::Scalar | Self::Array(_) => None,
        }
    }

    fn as_array(&self) -> Option<&[toml::Spanned<SourceNode>]> {
        match self {
            Self::Array(items) => Some(items),
            Self::Scalar | Self::Table(_) => None,
        }
    }

    fn is_table(&self) -> bool {
        matches!(self, Self::Table(_))
    }
}

impl<'de> Deserialize<'de> for SourceNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(SourceNodeVisitor)
    }
}

struct SourceNodeVisitor;

impl<'de> Visitor<'de> for SourceNodeVisitor {
    type Value = SourceNode;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("any TOML value")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(SourceNode::Scalar)
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(SourceNode::Scalar)
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(SourceNode::Scalar)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(SourceNode::Scalar)
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(SourceNode::Scalar)
    }

    fn visit_borrowed_str<E>(self, _value: &'de str) -> Result<Self::Value, E> {
        Ok(SourceNode::Scalar)
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E> {
        Ok(SourceNode::Scalar)
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element::<toml::Spanned<SourceNode>>()? {
            items.push(item);
        }
        Ok(SourceNode::Array(items))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut entries = BTreeMap::new();
        while let Some(key) = map.next_key::<String>()? {
            if key == DATETIME_FIELD {
                // `toml` exposes datetime values to `deserialize_any` as a
                // private one-field map. Treat that implementation detail as a
                // scalar so datetime-like settings do not look like tables.
                let _ = map.next_value::<IgnoredAny>()?;
                return Ok(SourceNode::Scalar);
            }
            let value = map.next_value::<toml::Spanned<SourceNode>>()?;
            entries.insert(key, value);
        }
        Ok(SourceNode::Table(entries))
    }
}

/// One segment in a diagnostic path through `ConfigSource`.
#[derive(Debug, Clone, Copy)]
enum PathPart<'a> {
    Key(&'a str),
    Index(usize),
}

fn key(value: &str) -> PathPart<'_> {
    PathPart::Key(value)
}

fn index(value: usize) -> PathPart<'static> {
    PathPart::Index(value)
}

/// Aggregated semantic config errors rendered in a compiler-style format.
#[derive(Debug, Clone)]
pub(crate) struct ConfigValidationReport {
    path: PathBuf,
    input: String,
    lines: LineIndex,
    diagnostics: Vec<ConfigDiagnostic>,
    ansi: bool,
}

impl ConfigValidationReport {
    fn new(source: &ConfigSource, diagnostics: Vec<ConfigDiagnostic>, ansi: bool) -> Self {
        Self {
            path: source.path.clone(),
            input: source.input.clone(),
            lines: source.lines.clone(),
            diagnostics,
            ansi,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.diagnostics.len()
    }

    #[cfg(test)]
    fn render(&self) -> String {
        self.render_with_style(DiagnosticStyle::plain())
    }

    pub(crate) fn render_for_stderr(&self) -> String {
        self.render_with_style(DiagnosticStyle::new(self.ansi && stderr_supports_color()))
    }

    fn render_with_style(&self, style: DiagnosticStyle) -> String {
        // Every collected error gets its own `error: ...` block. The trailing
        // summary mirrors rustc-style multi-error output and makes automation
        // logs easier to scan.
        let mut out = String::new();
        for (index, diagnostic) in self.diagnostics.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            diagnostic.render(&self.path, &self.input, &self.lines, style, &mut out);
        }
        if self.diagnostics.len() > 1 {
            let _ = writeln!(
                out,
                "\n{}: aborting due to {} config errors",
                style.error("error"),
                self.diagnostics.len()
            );
        }
        out
    }
}

impl fmt::Display for ConfigValidationReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "config validation failed with {} error{}",
            self.len(),
            if self.len() == 1 { "" } else { "s" }
        )
    }
}

impl std::error::Error for ConfigValidationReport {}

/// One rendered config error, with optional notes and a single help message.
#[derive(Debug, Clone)]
struct ConfigDiagnostic {
    message: String,
    labels: Vec<DiagnosticLabel>,
    notes: Vec<String>,
    help: Option<String>,
}

impl ConfigDiagnostic {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            labels: Vec::new(),
            notes: Vec::new(),
            help: None,
        }
    }

    fn with_label(mut self, label: DiagnosticLabel) -> Self {
        self.labels.push(label);
        self
    }

    fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    fn with_optional_note(mut self, note: Option<String>) -> Self {
        if let Some(note) = note {
            self.notes.push(note);
        }
        self
    }

    fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }

    fn with_optional_help(mut self, help: Option<String>) -> Self {
        self.help = help;
        self
    }

    fn render(
        &self,
        path: &Path,
        input: &str,
        lines: &LineIndex,
        style: DiagnosticStyle,
        out: &mut String,
    ) {
        let _ = writeln!(
            out,
            "{}: {}",
            style.error("error"),
            style.bold(&self.message)
        );
        if let Some(primary) = self.labels.first() {
            let (line, column) = lines.line_col(input, primary.span.start);
            let _ = writeln!(
                out,
                "  {} {}:{}:{}",
                style.location_arrow("-->"),
                path.display(),
                line + 1,
                column + 1
            );
            write_gutter_separator(out, line + 1, style);
            for label in &self.labels {
                render_label(input, lines, label, style, out);
            }
        }
        for note in &self.notes {
            let _ = writeln!(out, "  = {}: {note}", style.note("note"));
        }
        if let Some(help) = &self.help {
            let _ = writeln!(out, "  = {}: {help}", style.help("help"));
        }
    }
}

/// A source range and label text for one highlighted diagnostic span.
#[derive(Debug, Clone)]
struct DiagnosticLabel {
    span: Range<usize>,
    message: String,
}

#[derive(Debug, Clone, Copy)]
struct DiagnosticStyle {
    color: bool,
}

impl DiagnosticStyle {
    fn new(color: bool) -> Self {
        Self { color }
    }

    #[cfg(test)]
    fn plain() -> Self {
        Self { color: false }
    }

    fn error(self, text: &str) -> String {
        self.paint(ANSI_BOLD_RED, text)
    }

    fn caret(self, text: &str) -> String {
        self.paint(ANSI_BOLD_RED, text)
    }

    fn gutter(self, text: &str) -> String {
        self.paint(ANSI_BLUE, text)
    }

    fn location_arrow(self, text: &str) -> String {
        self.paint(ANSI_BLUE, text)
    }

    fn note(self, text: &str) -> String {
        self.paint(ANSI_GREEN, text)
    }

    fn help(self, text: &str) -> String {
        self.paint(ANSI_GREEN, text)
    }

    fn bold(self, text: &str) -> String {
        self.paint(ANSI_BOLD, text)
    }

    fn paint(self, code: &str, text: &str) -> String {
        if self.color {
            format!("{code}{text}{ANSI_RESET}")
        } else {
            text.to_string()
        }
    }
}

/// Whether stderr should receive ANSI color for diagnostic rendering.
pub(crate) fn stderr_supports_color() -> bool {
    std::io::stderr().is_terminal()
        && env::var_os("NO_COLOR").is_none()
        && env::var("TERM").map_or(true, |term| term != "dumb")
}

/// Byte line starts used to translate TOML byte spans into display positions.
#[derive(Debug, Clone)]
struct LineIndex {
    starts: Vec<usize>,
}

impl LineIndex {
    fn new(input: &str) -> Self {
        let mut starts = vec![0];
        for (index, byte) in input.bytes().enumerate() {
            if byte == b'\n' {
                starts.push(index + 1);
            }
        }
        Self { starts }
    }

    fn line_col(&self, input: &str, byte: usize) -> (usize, usize) {
        let byte = byte.min(input.len());
        let line = self
            .starts
            .partition_point(|start| *start <= byte)
            .saturating_sub(1);
        let line_start = self.starts.get(line).copied().unwrap_or(0);
        let prefix = input.get(line_start..byte).unwrap_or_default();
        (line, prefix.chars().count())
    }

    fn line_range(&self, input: &str, line: usize) -> Range<usize> {
        let start = self.starts.get(line).copied().unwrap_or(input.len());
        let end = self
            .starts
            .get(line + 1)
            .copied()
            .map(|next| next.saturating_sub(1))
            .unwrap_or(input.len());
        start..end
    }
}

fn write_gutter_separator(out: &mut String, max_line_number: usize, style: DiagnosticStyle) {
    let width = max_line_number.to_string().len();
    let _ = writeln!(out, "{:width$} {}", "", style.gutter("|"), width = width);
}

fn render_label(
    input: &str,
    lines: &LineIndex,
    label: &DiagnosticLabel,
    style: DiagnosticStyle,
    out: &mut String,
) {
    let start = label.span.start.min(input.len());
    let mut end = label.span.end.min(input.len()).max(start);
    if end == start {
        end = next_char_boundary(input, start);
    }
    let (start_line, _) = lines.line_col(input, start);
    let (end_line, _) = lines.line_col(input, end.saturating_sub(1));
    let max_line_number = end_line + 1;
    let width = max_line_number.to_string().len();
    let rendered_lines = rendered_line_indexes(start_line, end_line);
    for line in rendered_lines {
        match line {
            RenderedLine::Line(line) => {
                let line_range = lines.line_range(input, line);
                let line_start = line_range.start;
                let line_len = line_range.len();
                let line_text = input.get(line_range).unwrap_or_default();
                let highlight_start_byte = if line == start_line {
                    start.saturating_sub(line_start)
                } else {
                    0
                };
                let highlight_end_byte = if line == end_line {
                    end.saturating_sub(line_start)
                } else {
                    line_len
                };
                // Spans are byte offsets, but gutters and carets are rendered
                // in character columns so UTF-8 input stays aligned.
                let start_col = byte_to_char_col(line_text, highlight_start_byte);
                let end_col = byte_to_char_col(line_text, highlight_end_byte).max(start_col + 1);
                let window = SourceLineWindow::new(line_text, start_col, end_col);
                let _ = writeln!(
                    out,
                    "{:>width$} {} {}",
                    line + 1,
                    style.gutter("|"),
                    window.text,
                    width = width
                );
                let caret_len = window.end_col.saturating_sub(window.start_col).max(1);
                let _ = writeln!(
                    out,
                    "{:width$} {} {}{} {}",
                    "",
                    style.gutter("|"),
                    " ".repeat(window.start_col),
                    style.caret(&"^".repeat(caret_len)),
                    style.error(&label.message),
                    width = width
                );
            }
            RenderedLine::Ellipsis => {
                let _ = writeln!(
                    out,
                    "{:width$} {} ...",
                    "",
                    style.gutter("|"),
                    width = width
                );
            }
        }
    }
}

fn rendered_line_indexes(start_line: usize, end_line: usize) -> Vec<RenderedLine> {
    if end_line <= start_line + 3 {
        return (start_line..=end_line).map(RenderedLine::Line).collect();
    }
    vec![
        RenderedLine::Line(start_line),
        RenderedLine::Ellipsis,
        RenderedLine::Line(end_line),
    ]
}

enum RenderedLine {
    Line(usize),
    Ellipsis,
}

fn next_char_boundary(input: &str, start: usize) -> usize {
    input[start..]
        .char_indices()
        .nth(1)
        .map(|(offset, _)| start + offset)
        .unwrap_or_else(|| input.len().min(start + 1))
}

fn byte_to_char_col(line: &str, byte: usize) -> usize {
    line.get(..byte.min(line.len()))
        .unwrap_or_default()
        .chars()
        .count()
}

struct SourceLineWindow {
    text: String,
    start_col: usize,
    end_col: usize,
}

impl SourceLineWindow {
    fn new(line: &str, start_col: usize, end_col: usize) -> Self {
        let chars = line.chars().collect::<Vec<_>>();
        if chars.len() <= MAX_SOURCE_LINE_CHARS {
            return Self {
                text: line.to_string(),
                start_col,
                end_col: end_col.min(chars.len()).max(start_col + 1),
            };
        }

        let highlight_end = end_col.min(chars.len()).max(start_col + 1);
        // Long inline arrays/tables can be much wider than a terminal. Keep the
        // highlighted region visible and add ellipses only around the clipped
        // source text.
        let mut window_start = start_col.saturating_sub(48);
        let mut window_end = highlight_end.saturating_add(72).min(chars.len());
        if window_end.saturating_sub(window_start) < SOURCE_WINDOW_CHARS {
            window_start = window_end.saturating_sub(SOURCE_WINDOW_CHARS);
        }
        if window_end.saturating_sub(window_start) > SOURCE_WINDOW_CHARS {
            window_end = window_start + SOURCE_WINDOW_CHARS;
        }

        let prefix = if window_start > 0 { "... " } else { "" };
        let suffix = if window_end < chars.len() { " ..." } else { "" };
        let text = format!(
            "{prefix}{}{suffix}",
            chars[window_start..window_end].iter().collect::<String>()
        );
        let prefix_chars = prefix.chars().count();
        Self {
            text,
            start_col: prefix_chars + start_col.saturating_sub(window_start),
            end_col: prefix_chars + highlight_end.saturating_sub(window_start),
        }
    }
}

// Allowed-key schemas for config-owned TOML surfaces. Dynamic maps stay open at
// the map-key level; validators apply the relevant entry schema after serde has
// identified each configured agent, provider, platform, or pricing model.
const ROOT_KEYS: &[&str] = &[
    "database",
    "logging",
    "bot",
    "memory",
    "llm",
    "image",
    "video",
    "audio",
    "platforms",
    "web",
    "storage",
];
const DATABASE_KEYS: &[&str] = &["url"];
const LOGGING_KEYS: &[&str] = &["filter", "format", "ansi"];
const WEB_KEYS: &[&str] = &[
    "listen",
    "title_prefix",
    "frontend_dir",
    "favicon_path",
    "public_base_url",
    "og_image_path",
    "trust_forwarded_for",
];
const STORAGE_LOCAL_KEYS: &[&str] = &[
    "kind",
    "images_dir",
    "videos_dir",
    "audio_dir",
    "avatars_dir",
    "guild_icons_dir",
    "public_base_url",
];
const STORAGE_S3_KEYS: &[&str] = &[
    "kind",
    "bucket",
    "region",
    "endpoint_url",
    "force_path_style",
    "public_base_url",
];
const MEMORY_KEYS: &[&str] = &[
    "enabled",
    "poll_interval_seconds",
    "compaction_interval",
    "diary_backfill_window",
    "diary_interval",
    "lease_seconds",
    "max_jobs_per_tick",
    "max_concurrent_jobs",
    "max_transcript_turns_per_diary_job",
    "retry_backoff_seconds",
    "max_job_attempts",
];
const USER_REF_KEYS: &[&str] = &["platform", "guild_id", "user_id"];
const BOT_KEYS: &[&str] = &[
    "web_base_url",
    "default_agent",
    "agents",
    "admins",
    "platforms",
    "extra_system_prompt",
    "version",
    "limits",
    "thread_threshold_chars",
    "thread_threshold_lines",
];
const LIMIT_KEYS: &[&str] = &["max_iterations"];
const PLATFORM_BINDING_KEYS: &[&str] = &["agent"];
const AGENT_KEYS: &[&str] = &[
    "provider",
    "system_prompt",
    "model",
    "server_tools",
    "client_tools",
    "limits",
    "image_generation",
    "video_generation",
    "audio_transcription",
    "memory",
    "subagents",
];
const MODEL_KEYS: &[&str] = &["id", "server_tools", "sampling", "provider_options"];
const SAMPLING_KEYS: &[&str] = &["max_output_tokens", "temperature", "top_p"];
const PROVIDER_OPTIONS_KEYS: &[&str] = &["value"];
const GENERATION_BINDING_KEYS: &[&str] = &["provider", "model", "rate_limit"];
const TRANSCRIPTION_BINDING_KEYS: &[&str] = &["provider", "model", "wake_word"];
const RATE_LIMIT_KEYS: &[&str] = &["limit", "interval", "bypass_scopes"];
const PLATFORM_SCOPE_BYPASS_KEYS: &[&str] = &["platform", "scope_id"];
const SUBAGENT_BINDING_KEYS: &[&str] = &["agent", "description"];
const LLM_XAI_KEYS: &[&str] = &["kind", "api_key", "base_url", "dump_dir", "model_info"];
const LLM_OPENAI_KEYS: &[&str] = &["kind", "api_key", "base_url", "pricing", "model_info"];
const LLM_ANTHROPIC_KEYS: &[&str] = &["kind", "api_key", "base_url", "pricing", "model_info"];
const LLM_OPENAI_COMPAT_KEYS: &[&str] = &["kind", "base_url", "api_key", "model_info"];
const LLM_GEMINI_KEYS: &[&str] = &["kind", "api_key", "base_url", "model_info"];
const IMAGE_OPENAI_KEYS: &[&str] = &["kind", "api_key", "base_url", "pricing"];
const IMAGE_XAI_KEYS: &[&str] = &["kind", "api_key", "base_url"];
const IMAGE_GEMINI_KEYS: &[&str] = &["kind", "api_key", "base_url"];
const VIDEO_PROVIDER_KEYS: &[&str] = &["kind", "api_key", "base_url"];
const AUDIO_PROVIDER_KEYS: &[&str] = &["kind", "api_key", "base_url"];
const DISCORD_PLATFORM_KEYS: &[&str] = &["kind", "token", "dev_guild_id"];
const OPENAI_TOKEN_PRICING_KEYS: &[&str] = &[
    "input_usd_per_million_tokens",
    "cached_input_usd_per_million_tokens",
    "output_usd_per_million_tokens",
];
const OPENAI_IMAGE_PRICING_KEYS: &[&str] = &[
    "text_input_usd_per_million_tokens",
    "cached_text_input_usd_per_million_tokens",
    "image_input_usd_per_million_tokens",
    "cached_image_input_usd_per_million_tokens",
    "image_output_usd_per_million_tokens",
    "text_output_usd_per_million_tokens",
];
const ANTHROPIC_TOKEN_PRICING_KEYS: &[&str] = &[
    "input_usd_per_million_tokens",
    "cache_creation_5m_usd_per_million_tokens",
    "cache_creation_1h_usd_per_million_tokens",
    "cache_read_usd_per_million_tokens",
    "output_usd_per_million_tokens",
];
const LLM_MODEL_INFO_KEYS: &[&str] = &["context_window_tokens", "max_output_tokens"];

/// Runs post-deserialization validation and returns every semantic/stale-key
/// problem found in the config.
///
/// This is intentionally separate from serde's unknown-field handling. Serde
/// can stop at the first stale key and often lacks the path-specific rendering
/// context this layer has.
pub(crate) fn validate_runtime_config(
    config: &RuntimeConfig,
    source: &ConfigSource,
) -> Result<(), ConfigValidationReport> {
    let mut diagnostics = Vec::new();
    // Report stale keys first so renamed/removed settings are visible before
    // follow-on semantic errors caused by the surviving config values.
    validate_unexpected_keys(config, source, &mut diagnostics);
    // The remaining passes validate relationships that only make sense after
    // TOML has deserialized into the typed runtime config.
    validate_database(config, source, &mut diagnostics);
    validate_logging(config, source, &mut diagnostics);
    validate_storage(config, source, &mut diagnostics);
    validate_bot_config(config, source, &mut diagnostics);
    validate_sampling_number_literals(config, source, &mut diagnostics);
    validate_memory_durations(config, source, &mut diagnostics);
    validate_runtime_references(config, source, &mut diagnostics);
    validate_web(config, source, &mut diagnostics);

    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(ConfigValidationReport::new(
            source,
            diagnostics,
            config.logging.ansi,
        ))
    }
}

fn validate_unexpected_keys(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    // Variant-specific sections use the deserialized config to choose the
    // correct allowed-key set. That keeps one diagnostics path for stale keys
    // without making unrelated provider/agent names closed.
    validate_known_keys(source, diagnostics, &[], ROOT_KEYS);
    validate_known_keys(source, diagnostics, &[key("database")], DATABASE_KEYS);
    validate_known_keys(source, diagnostics, &[key("logging")], LOGGING_KEYS);
    validate_known_keys(source, diagnostics, &[key("web")], WEB_KEYS);
    match &config.storage {
        StorageConfig::Local(_) => {
            validate_known_keys(source, diagnostics, &[key("storage")], STORAGE_LOCAL_KEYS);
        }
        StorageConfig::S3(_) => {
            validate_known_keys(source, diagnostics, &[key("storage")], STORAGE_S3_KEYS);
        }
    }
    validate_known_keys(source, diagnostics, &[key("memory")], MEMORY_KEYS);
    validate_bot_unexpected_keys(config, source, diagnostics);
    validate_runtime_provider_unexpected_keys(config, source, diagnostics);
}

fn validate_bot_unexpected_keys(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let bot_path = [key("bot")];
    validate_known_keys(source, diagnostics, &bot_path, BOT_KEYS);
    validate_known_keys(
        source,
        diagnostics,
        &child_path(&bot_path, "limits"),
        LIMIT_KEYS,
    );
    validate_array_item_keys(
        source,
        diagnostics,
        &child_path(&bot_path, "admins"),
        USER_REF_KEYS,
    );
    validate_map_entry_keys(
        source,
        diagnostics,
        &child_path(&bot_path, "platforms"),
        PLATFORM_BINDING_KEYS,
    );

    let agents_path = child_path(&bot_path, "agents");
    for agent_name in config.bot.agents.keys() {
        validate_agent_unexpected_keys(source, diagnostics, &agents_path, agent_name);
    }
}

fn validate_agent_unexpected_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    agents_path: &[PathPart<'_>],
    agent_name: &str,
) {
    let agent_path = child_path(agents_path, agent_name);
    validate_known_keys(source, diagnostics, &agent_path, AGENT_KEYS);
    validate_known_keys(
        source,
        diagnostics,
        &child_path(&agent_path, "limits"),
        LIMIT_KEYS,
    );
    let model_path = child_path(&agent_path, "model");
    validate_known_keys(source, diagnostics, &model_path, MODEL_KEYS);
    validate_known_keys(
        source,
        diagnostics,
        &child_path(&model_path, "sampling"),
        SAMPLING_KEYS,
    );
    validate_known_keys(
        source,
        diagnostics,
        &child_path(&model_path, "provider_options"),
        PROVIDER_OPTIONS_KEYS,
    );
    // `provider_options.value` is provider-owned and intentionally opaque here;
    // this layer only catches stale keys in the config-owned envelope.
    validate_generation_binding_unexpected_keys(
        source,
        diagnostics,
        &child_path(&agent_path, "image_generation"),
    );
    validate_generation_binding_unexpected_keys(
        source,
        diagnostics,
        &child_path(&agent_path, "video_generation"),
    );
    validate_transcription_binding_unexpected_keys(
        source,
        diagnostics,
        &child_path(&agent_path, "audio_transcription"),
    );
    validate_map_entry_keys(
        source,
        diagnostics,
        &child_path(&agent_path, "subagents"),
        SUBAGENT_BINDING_KEYS,
    );
}

fn validate_generation_binding_unexpected_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    binding_path: &[PathPart<'_>],
) {
    validate_known_keys(source, diagnostics, binding_path, GENERATION_BINDING_KEYS);
    let rate_limit_path = child_path(binding_path, "rate_limit");
    validate_known_keys(source, diagnostics, &rate_limit_path, RATE_LIMIT_KEYS);
    validate_array_item_keys(
        source,
        diagnostics,
        &child_path(&rate_limit_path, "bypass_scopes"),
        PLATFORM_SCOPE_BYPASS_KEYS,
    );
}

fn validate_transcription_binding_unexpected_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    binding_path: &[PathPart<'_>],
) {
    validate_known_keys(
        source,
        diagnostics,
        binding_path,
        TRANSCRIPTION_BINDING_KEYS,
    );
}

fn validate_runtime_provider_unexpected_keys(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    // Provider and platform names are user-defined map keys. Iterate the typed
    // config so each provider variant gets the correct inner schema.
    for (provider_name, provider) in &config.llm {
        validate_llm_provider_unexpected_keys(
            source,
            diagnostics,
            provider_name.as_str(),
            provider,
        );
    }
    for (provider_name, provider) in &config.image {
        validate_image_provider_unexpected_keys(
            source,
            diagnostics,
            provider_name.as_str(),
            provider,
        );
    }
    for (provider_name, provider) in &config.video {
        validate_video_provider_unexpected_keys(
            source,
            diagnostics,
            provider_name.as_str(),
            provider,
        );
    }
    for (provider_name, provider) in &config.audio {
        validate_audio_provider_unexpected_keys(
            source,
            diagnostics,
            provider_name.as_str(),
            provider,
        );
    }
    for (platform_name, platform) in &config.platforms {
        validate_platform_unexpected_keys(source, diagnostics, platform_name.as_str(), platform);
    }
}

fn validate_llm_provider_unexpected_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    provider_name: &str,
    provider: &LlmProviderConfig,
) {
    let path = [key("llm"), key(provider_name)];
    validate_map_entry_keys(
        source,
        diagnostics,
        &child_path(&path, "model_info"),
        LLM_MODEL_INFO_KEYS,
    );
    match provider {
        LlmProviderConfig::Xai { .. } => {
            validate_known_keys(source, diagnostics, &path, LLM_XAI_KEYS);
        }
        LlmProviderConfig::OpenAi { .. } => {
            validate_known_keys(source, diagnostics, &path, LLM_OPENAI_KEYS);
            validate_map_entry_keys(
                source,
                diagnostics,
                &child_path(&path, "pricing"),
                OPENAI_TOKEN_PRICING_KEYS,
            );
        }
        LlmProviderConfig::Anthropic { .. } => {
            validate_known_keys(source, diagnostics, &path, LLM_ANTHROPIC_KEYS);
            validate_map_entry_keys(
                source,
                diagnostics,
                &child_path(&path, "pricing"),
                ANTHROPIC_TOKEN_PRICING_KEYS,
            );
        }
        LlmProviderConfig::OpenAiCompat { .. } => {
            validate_known_keys(source, diagnostics, &path, LLM_OPENAI_COMPAT_KEYS);
        }
        LlmProviderConfig::Gemini { .. } => {
            validate_known_keys(source, diagnostics, &path, LLM_GEMINI_KEYS);
        }
    }
}

fn validate_image_provider_unexpected_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    provider_name: &str,
    provider: &ImageProviderConfig,
) {
    let path = [key("image"), key(provider_name)];
    match provider {
        ImageProviderConfig::OpenAi { .. } => {
            validate_known_keys(source, diagnostics, &path, IMAGE_OPENAI_KEYS);
            validate_map_entry_keys(
                source,
                diagnostics,
                &child_path(&path, "pricing"),
                OPENAI_IMAGE_PRICING_KEYS,
            );
        }
        ImageProviderConfig::Xai { .. } => {
            validate_known_keys(source, diagnostics, &path, IMAGE_XAI_KEYS);
        }
        ImageProviderConfig::Gemini { .. } => {
            validate_known_keys(source, diagnostics, &path, IMAGE_GEMINI_KEYS);
        }
    }
}

fn validate_video_provider_unexpected_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    provider_name: &str,
    provider: &VideoProviderConfig,
) {
    let path = [key("video"), key(provider_name)];
    match provider {
        VideoProviderConfig::Xai { .. } | VideoProviderConfig::Gemini { .. } => {
            validate_known_keys(source, diagnostics, &path, VIDEO_PROVIDER_KEYS);
        }
    }
}

fn validate_audio_provider_unexpected_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    provider_name: &str,
    provider: &AudioProviderConfig,
) {
    let path = [key("audio"), key(provider_name)];
    match provider {
        AudioProviderConfig::Xai { .. } => {
            validate_known_keys(source, diagnostics, &path, AUDIO_PROVIDER_KEYS);
        }
    }
}

fn validate_platform_unexpected_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    platform_name: &str,
    platform: &MessagePlatformConfig,
) {
    let path = [key("platforms"), key(platform_name)];
    match platform {
        MessagePlatformConfig::Discord { .. } => {
            validate_known_keys(source, diagnostics, &path, DISCORD_PLATFORM_KEYS);
        }
    }
}

fn validate_known_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    path: &[PathPart<'_>],
    allowed: &[&str],
) {
    // Core stale-key check. If the path is absent or is not a table, serde's
    // syntax/type path owns that failure and this pass stays silent.
    let Some(entries) = source.table_at(path) else {
        return;
    };
    for (entry_name, entry) in entries {
        if !allowed.contains(&entry_name.as_str()) {
            diagnostics.push(unexpected_key_diagnostic(
                source, path, entry_name, entry, allowed,
            ));
        }
    }
}

fn validate_map_entry_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    map_path: &[PathPart<'_>],
    entry_allowed: &[&str],
) {
    // The table's direct keys are names such as agents, providers, or models;
    // each named child gets checked against the schema for one entry.
    let Some(entries) = source.table_at(map_path) else {
        return;
    };
    for entry_name in entries.keys() {
        validate_known_keys(
            source,
            diagnostics,
            &child_path(map_path, entry_name),
            entry_allowed,
        );
    }
}

fn validate_array_item_keys(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    array_path: &[PathPart<'_>],
    item_allowed: &[&str],
) {
    // Arrays are closed at the item shape, not at the array length.
    let Some(items) = source.array_at(array_path) else {
        return;
    };
    for index in 0..items.len() {
        validate_known_keys(
            source,
            diagnostics,
            &indexed_path(array_path, index),
            item_allowed,
        );
    }
}

fn child_path<'a>(path: &[PathPart<'a>], child: &'a str) -> Vec<PathPart<'a>> {
    let mut out = path.to_vec();
    out.push(key(child));
    out
}

fn indexed_path<'a>(path: &[PathPart<'a>], child: usize) -> Vec<PathPart<'a>> {
    let mut out = path.to_vec();
    out.push(index(child));
    out
}

fn unexpected_key_diagnostic(
    source: &ConfigSource,
    parent_path: &[PathPart<'_>],
    entry_name: &str,
    entry: &toml::Spanned<SourceNode>,
    allowed: &[&str],
) -> ConfigDiagnostic {
    let mut full_path = parent_path.to_vec();
    full_path.push(key(entry_name));
    let entry_kind = if entry.get_ref().is_table() {
        "section"
    } else {
        "key"
    };
    // Prefer a direct path lookup so inline tables and arrays get consistent
    // path formatting; otherwise use the span already carried by the entry.
    let span = source.span_for(&full_path).unwrap_or_else(|| entry.span());
    ConfigDiagnostic::new(format!(
        "unexpected config {entry_kind} `{}`",
        config_path(&full_path)
    ))
    .with_label(DiagnosticLabel {
        span,
        message: format!("unexpected {entry_kind} here"),
    })
    .with_optional_help(unexpected_key_help(entry_name, allowed))
}

fn unexpected_key_help(entry_name: &str, allowed: &[&str]) -> Option<String> {
    if allowed.is_empty() {
        return None;
    }
    let allowed_names = allowed
        .iter()
        .map(|name| (*name).to_string())
        .collect::<BTreeSet<_>>();
    let expected = allowed
        .iter()
        .map(|name| format!("`{}`", toml_key(name)))
        .collect::<Vec<_>>()
        .join(", ");
    if let Some(suggestion) = closest_name(entry_name, &allowed_names) {
        Some(format!(
            "did you mean `{}`?; expected keys here: {expected}",
            toml_key(suggestion)
        ))
    } else {
        Some(format!("expected keys here: {expected}"))
    }
}

// Semantic validators below operate on a successfully deserialized
// `RuntimeConfig` and use `ConfigSource` only to attach useful source labels.
fn validate_database(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if config.database.url.trim().is_empty() {
        diagnostics.push(
            ConfigDiagnostic::new("database.url must not be empty").with_label(
                source.primary_label(
                    &[key("database"), key("url")],
                    &[key("database")],
                    "empty database URL",
                ),
            ),
        );
    }
}

fn validate_logging(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if let Err(error) = config.logging.filter() {
        let detail = error.source().map(|source| source.to_string());
        diagnostics.push(
            ConfigDiagnostic::new(format!(
                "invalid logging filter `{}`",
                config.logging.filter
            ))
            .with_label(source.primary_label(
                &[key("logging"), key("filter")],
                &[key("logging")],
                "invalid tracing filter",
            ))
            .with_optional_note(detail)
            .with_help("use tracing-subscriber EnvFilter syntax, for example `info,chudbot=debug`"),
        );
    }
}

fn validate_storage(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let StorageConfig::S3(storage) = &config.storage else {
        return;
    };

    let bucket = storage.bucket.trim();
    if bucket.is_empty() {
        diagnostics.push(
            ConfigDiagnostic::new("storage.bucket must not be empty when storage.kind is `s3`")
                .with_label(source.primary_label(
                    &[key("storage"), key("bucket")],
                    &[key("storage")],
                    "missing S3 bucket name",
                ))
                .with_help("set `bucket` to the S3 bucket that will store media objects"),
        );
    } else if bucket.contains('/') || bucket.contains("://") {
        diagnostics.push(
            ConfigDiagnostic::new("storage.bucket must be an S3 bucket name, not a URL or path")
                .with_label(source.primary_label(
                    &[key("storage"), key("bucket")],
                    &[key("storage")],
                    "bucket names cannot contain `/` or `://`",
                ))
                .with_help(
                    "put the bucket host override in `endpoint_url`; keep `bucket` as the bucket name",
                ),
        );
    }

    if storage
        .region
        .as_deref()
        .is_some_and(|region| region.trim().is_empty())
    {
        diagnostics.push(invalid_storage_string_diagnostic(
            source,
            "region",
            "storage.region must not be empty when set",
        ));
    }
    if storage
        .endpoint_url
        .as_deref()
        .is_some_and(|endpoint_url| endpoint_url.trim().is_empty())
    {
        diagnostics.push(invalid_storage_string_diagnostic(
            source,
            "endpoint_url",
            "storage.endpoint_url must not be empty when set",
        ));
    }
    if storage
        .public_base_url
        .as_deref()
        .is_some_and(|public_base_url| public_base_url.trim().is_empty())
    {
        diagnostics.push(invalid_storage_string_diagnostic(
            source,
            "public_base_url",
            "storage.public_base_url must not be empty when set",
        ));
    }
}

fn invalid_storage_string_diagnostic(
    source: &ConfigSource,
    field: &'static str,
    message: &'static str,
) -> ConfigDiagnostic {
    ConfigDiagnostic::new(message).with_label(source.primary_label(
        &[key("storage"), key(field)],
        &[key("storage")],
        "empty storage setting",
    ))
}

fn validate_memory_durations(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if let Err(error) = config.memory.compaction_interval_seconds() {
        diagnostics.push(memory_duration_diagnostic(
            source,
            "compaction_interval",
            &error.to_string(),
        ));
    }
    if let Err(error) = config.memory.diary_backfill_window_seconds() {
        diagnostics.push(memory_duration_diagnostic(
            source,
            "diary_backfill_window",
            &error.to_string(),
        ));
    }
    if let Err(error) = config.memory.diary_interval_seconds() {
        diagnostics.push(memory_duration_diagnostic(
            source,
            "diary_interval",
            &error.to_string(),
        ));
    }
}

fn memory_duration_diagnostic(
    source: &ConfigSource,
    field: &'static str,
    error: &str,
) -> ConfigDiagnostic {
    ConfigDiagnostic::new(format!("invalid memory duration in `memory.{field}`"))
        .with_label(source.primary_label(
            &[key("memory"), key(field)],
            &[key("memory")],
            "invalid duration",
        ))
        .with_note(error)
        .with_help("use digits followed by `s`, `m`, `h`, or `d`; the value must be non-zero")
}

fn validate_bot_config(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    // Agent names form a local registry. Validate the registry itself first,
    // then every place that refers back into it.
    let agents = config.bot.agents.keys().cloned().collect::<BTreeSet<_>>();
    if !config.bot.agents.contains_key(&config.bot.default_agent) {
        diagnostics.push(
            ConfigDiagnostic::new(format!(
                "default agent `{}` is not configured",
                config.bot.default_agent
            ))
            .with_label(source.primary_label(
                &[key("bot"), key("default_agent")],
                &[key("bot")],
                format!(
                    "no {} table exists",
                    table_path(&["bot", "agents", &config.bot.default_agent])
                ),
            ))
            .with_optional_help(missing_name_help(
                &config.bot.default_agent,
                "agent",
                &agents,
                Some(format!(
                    "add {}",
                    table_path(&["bot", "agents", &config.bot.default_agent])
                )),
            )),
        );
    }

    for (platform, binding) in &config.bot.platforms {
        if !config.bot.agents.contains_key(&binding.agent) {
            diagnostics.push(
                ConfigDiagnostic::new(format!(
                    "platform `{platform}` uses missing default agent `{}`",
                    binding.agent
                ))
                .with_label(source.primary_label(
                    &[
                        key("bot"),
                        key("platforms"),
                        key(platform.as_str()),
                        key("agent"),
                    ],
                    &[key("bot"), key("platforms"), key(platform.as_str())],
                    "missing agent referenced here",
                ))
                .with_optional_help(missing_name_help(
                    &binding.agent,
                    "agent",
                    &agents,
                    Some(format!(
                        "add {}",
                        table_path(&["bot", "agents", &binding.agent])
                    )),
                )),
            );
        }
    }

    for (agent_name, agent) in &config.bot.agents {
        if let Some(binding) = &agent.image_generation {
            validate_generation_binding(
                source,
                diagnostics,
                agent_name,
                "image_generation",
                binding,
            );
        }
        if let Some(binding) = &agent.video_generation {
            validate_generation_binding(
                source,
                diagnostics,
                agent_name,
                "video_generation",
                binding,
            );
        }
        if let Some(binding) = &agent.audio_transcription {
            validate_transcription_binding(
                source,
                diagnostics,
                agent_name,
                "audio_transcription",
                binding,
            );
        }
        for (tool_name, binding) in &agent.subagents {
            if !config.bot.agents.contains_key(&binding.agent) {
                diagnostics.push(
                    ConfigDiagnostic::new(format!(
                        "agent `{agent_name}` references missing subagent `{}`",
                        binding.agent
                    ))
                    .with_label(source.primary_label(
                        &[
                            key("bot"),
                            key("agents"),
                            key(agent_name),
                            key("subagents"),
                            key(tool_name.as_str()),
                            key("agent"),
                        ],
                        &[
                            key("bot"),
                            key("agents"),
                            key(agent_name),
                            key("subagents"),
                            key(tool_name.as_str()),
                        ],
                        "missing agent referenced here",
                    ))
                    .with_optional_help(missing_name_help(
                        &binding.agent,
                        "agent",
                        &agents,
                        Some(format!(
                            "add {}",
                            table_path(&["bot", "agents", &binding.agent])
                        )),
                    )),
                );
            }
        }
    }
}

fn validate_sampling_number_literals(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    for (agent_name, agent) in &config.bot.agents {
        let sampling = &agent.model.sampling;
        if sampling.temperature.is_some() {
            validate_sampling_number_literal(source, diagnostics, agent_name, "temperature");
        }
        if sampling.top_p.is_some() {
            validate_sampling_number_literal(source, diagnostics, agent_name, "top_p");
        }
    }
}

fn validate_sampling_number_literal(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    agent_name: &str,
    field: &str,
) {
    let path = [
        key("bot"),
        key("agents"),
        key(agent_name),
        key("model"),
        key("sampling"),
        key(field),
    ];
    let Some(raw) =
        source.source_for_keys(&["bot", "agents", agent_name, "model", "sampling", field])
    else {
        return;
    };
    if SamplingNumber::from_json_number_literal(raw).is_ok() {
        return;
    }
    diagnostics.push(
        ConfigDiagnostic::new(format!(
            "`{}` must be a JSON-compatible number literal",
            config_path(&path)
        ))
        .with_label(source.primary_label(
            &path,
            &path[..path.len() - 1],
            "cannot be copied into JSON exactly as written",
        ))
        .with_help("use JSON number syntax such as `1.3`; avoid TOML-only forms like `+1.3`, `1_000`, `inf`, or `nan`"),
    );
}

fn validate_generation_binding(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    agent_name: &str,
    field: &'static str,
    binding: &GenerationBinding,
) {
    let base = [key("bot"), key("agents"), key(agent_name), key(field)];
    if binding.provider.as_str().trim().is_empty() {
        diagnostics.push(invalid_binding_diagnostic(
            source,
            agent_name,
            field,
            &[
                key("bot"),
                key("agents"),
                key(agent_name),
                key(field),
                key("provider"),
            ],
            &base,
            "provider is empty",
        ));
    }
    if binding.model.as_str().trim().is_empty() {
        diagnostics.push(invalid_binding_diagnostic(
            source,
            agent_name,
            field,
            &[
                key("bot"),
                key("agents"),
                key(agent_name),
                key(field),
                key("model"),
            ],
            &base,
            "model is empty",
        ));
    }
    if let Some(rate_limit) = &binding.rate_limit {
        if field != "video_generation" {
            diagnostics.push(invalid_binding_diagnostic(
                source,
                agent_name,
                field,
                &[
                    key("bot"),
                    key("agents"),
                    key(agent_name),
                    key(field),
                    key("rate_limit"),
                ],
                &base,
                "rate_limit is only supported on video_generation",
            ));
        } else {
            validate_video_rate_limit(source, diagnostics, agent_name, field, rate_limit);
        }
    }
}

fn validate_video_rate_limit(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    agent_name: &str,
    field: &'static str,
    rate_limit: &VideoGenerationRateLimit,
) {
    let base = [
        key("bot"),
        key("agents"),
        key(agent_name),
        key(field),
        key("rate_limit"),
    ];
    if rate_limit.limit == 0 {
        diagnostics.push(invalid_binding_diagnostic(
            source,
            agent_name,
            field,
            &[
                key("bot"),
                key("agents"),
                key(agent_name),
                key(field),
                key("rate_limit"),
                key("limit"),
            ],
            &base,
            "rate_limit.limit must be greater than zero",
        ));
    }
    if let Err(message) = rate_limit.interval_seconds() {
        diagnostics.push(invalid_binding_diagnostic(
            source,
            agent_name,
            field,
            &[
                key("bot"),
                key("agents"),
                key(agent_name),
                key(field),
                key("rate_limit"),
                key("interval"),
            ],
            &base,
            &message,
        ));
    }
    for (scope_index, scope) in rate_limit.bypass_scopes.iter().enumerate() {
        if scope.platform.as_str().trim().is_empty() {
            diagnostics.push(invalid_binding_diagnostic(
                source,
                agent_name,
                field,
                &[
                    key("bot"),
                    key("agents"),
                    key(agent_name),
                    key(field),
                    key("rate_limit"),
                    key("bypass_scopes"),
                    index(scope_index),
                    key("platform"),
                ],
                &base,
                "rate_limit.bypass_scopes platform must not be empty",
            ));
        }
        if scope.scope_id.as_str().trim().is_empty() {
            diagnostics.push(invalid_binding_diagnostic(
                source,
                agent_name,
                field,
                &[
                    key("bot"),
                    key("agents"),
                    key(agent_name),
                    key(field),
                    key("rate_limit"),
                    key("bypass_scopes"),
                    index(scope_index),
                    key("scope_id"),
                ],
                &base,
                "rate_limit.bypass_scopes scope_id must not be empty",
            ));
        }
    }
}

fn validate_transcription_binding(
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
    agent_name: &str,
    field: &'static str,
    binding: &TranscriptionBinding,
) {
    let base = [key("bot"), key("agents"), key(agent_name), key(field)];
    if binding.provider.as_str().trim().is_empty() {
        diagnostics.push(invalid_binding_diagnostic(
            source,
            agent_name,
            field,
            &[
                key("bot"),
                key("agents"),
                key(agent_name),
                key(field),
                key("provider"),
            ],
            &base,
            "provider is empty",
        ));
    }
    if let Some(model) = &binding.model
        && model.as_str().trim().is_empty()
    {
        diagnostics.push(invalid_binding_diagnostic(
            source,
            agent_name,
            field,
            &[
                key("bot"),
                key("agents"),
                key(agent_name),
                key(field),
                key("model"),
            ],
            &base,
            "model is empty",
        ));
    }
    if let Some(wake_word) = &binding.wake_word
        && wake_word.trim().is_empty()
    {
        diagnostics.push(invalid_binding_diagnostic(
            source,
            agent_name,
            field,
            &[
                key("bot"),
                key("agents"),
                key(agent_name),
                key(field),
                key("wake_word"),
            ],
            &base,
            "wake_word is empty",
        ));
    }
}

fn invalid_binding_diagnostic(
    source: &ConfigSource,
    agent_name: &str,
    field: &'static str,
    path: &[PathPart<'_>],
    fallback_path: &[PathPart<'_>],
    message: &str,
) -> ConfigDiagnostic {
    ConfigDiagnostic::new(format!(
        "agent `{agent_name}` has invalid `{field}` binding: {message}"
    ))
    .with_label(source.primary_label(path, fallback_path, message))
}

fn validate_runtime_references(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    // Provider/platform references cross top-level registries. Snapshot the
    // available names once so every diagnostic can suggest the same choices.
    let llm_names = provider_names(&config.llm);
    let image_names = provider_names(&config.image);
    let video_names = provider_names(&config.video);
    let audio_names = provider_names(&config.audio);
    let platform_names = config
        .platforms
        .keys()
        .map(ToString::to_string)
        .collect::<BTreeSet<_>>();

    if config.memory.enabled {
        for (agent, provider) in config
            .memory
            .resolved_agent_providers(&config.bot.agents, config.bot.limits)
        {
            if config.bot.agents.contains_key(&agent) || llm_names.contains(provider.as_str()) {
                continue;
            }
            diagnostics.push(
                missing_provider_diagnostic(
                    source,
                    MissingProviderDiagnostic {
                        subject: "memory agent",
                        name: &agent,
                        provider: provider.as_str(),
                        table: "llm",
                        available: &llm_names,
                    },
                    &[key("memory"), key("enabled")],
                    &[key("memory")],
                )
                .with_note(format!(
                    "`{agent}` uses the built-in memory-agent default because {} is omitted",
                    table_path(&["bot", "agents", &agent])
                )),
            );
        }
    }

    for (agent_name, agent) in &config.bot.agents {
        if !agent.provider.as_str().trim().is_empty()
            && !llm_names.contains(agent.provider.as_str())
        {
            diagnostics.push(missing_provider_diagnostic(
                source,
                MissingProviderDiagnostic {
                    subject: "agent",
                    name: agent_name,
                    provider: agent.provider.as_str(),
                    table: "llm",
                    available: &llm_names,
                },
                &[key("bot"), key("agents"), key(agent_name), key("provider")],
                &[key("bot"), key("agents"), key(agent_name)],
            ));
        }
        if let Some(binding) = &agent.image_generation
            && !binding.provider.as_str().trim().is_empty()
            && !image_names.contains(binding.provider.as_str())
        {
            diagnostics.push(missing_provider_diagnostic(
                source,
                MissingProviderDiagnostic {
                    subject: "agent image",
                    name: agent_name,
                    provider: binding.provider.as_str(),
                    table: "image",
                    available: &image_names,
                },
                &[
                    key("bot"),
                    key("agents"),
                    key(agent_name),
                    key("image_generation"),
                    key("provider"),
                ],
                &[
                    key("bot"),
                    key("agents"),
                    key(agent_name),
                    key("image_generation"),
                ],
            ));
        }
        if let Some(binding) = &agent.video_generation
            && !binding.provider.as_str().trim().is_empty()
            && !video_names.contains(binding.provider.as_str())
        {
            diagnostics.push(missing_provider_diagnostic(
                source,
                MissingProviderDiagnostic {
                    subject: "agent video",
                    name: agent_name,
                    provider: binding.provider.as_str(),
                    table: "video",
                    available: &video_names,
                },
                &[
                    key("bot"),
                    key("agents"),
                    key(agent_name),
                    key("video_generation"),
                    key("provider"),
                ],
                &[
                    key("bot"),
                    key("agents"),
                    key(agent_name),
                    key("video_generation"),
                ],
            ));
        }
        if let Some(binding) = &agent.audio_transcription
            && !binding.provider.as_str().trim().is_empty()
            && !audio_names.contains(binding.provider.as_str())
        {
            diagnostics.push(missing_provider_diagnostic(
                source,
                MissingProviderDiagnostic {
                    subject: "agent audio",
                    name: agent_name,
                    provider: binding.provider.as_str(),
                    table: "audio",
                    available: &audio_names,
                },
                &[
                    key("bot"),
                    key("agents"),
                    key(agent_name),
                    key("audio_transcription"),
                    key("provider"),
                ],
                &[
                    key("bot"),
                    key("agents"),
                    key(agent_name),
                    key("audio_transcription"),
                ],
            ));
        }
    }

    for platform in config.bot.platforms.keys() {
        if !config.platforms.contains_key(platform) {
            diagnostics.push(
                ConfigDiagnostic::new(format!(
                    "platform `{platform}` is bound in [bot.platforms] but has no [platforms] entry"
                ))
                .with_label(source.primary_label(
                    &[key("bot"), key("platforms"), key(platform.as_str())],
                    &[key("bot"), key("platforms")],
                    "platform binding has no matching platform config",
                ))
                .with_optional_help(missing_name_help(
                    platform.as_str(),
                    "platform config",
                    &platform_names,
                    Some(format!(
                        "add {}",
                        table_path(&["platforms", platform.as_str()])
                    )),
                )),
            );
        }
    }
}

fn provider_names<T>(providers: &BTreeMap<ProviderName, T>) -> BTreeSet<String> {
    providers.keys().map(ToString::to_string).collect()
}

struct MissingProviderDiagnostic<'a> {
    subject: &'a str,
    name: &'a str,
    provider: &'a str,
    table: &'a str,
    available: &'a BTreeSet<String>,
}

fn missing_provider_diagnostic(
    source: &ConfigSource,
    details: MissingProviderDiagnostic<'_>,
    path: &[PathPart<'_>],
    fallback_path: &[PathPart<'_>],
) -> ConfigDiagnostic {
    let message = match details.subject {
        "agent" => {
            format!(
                "agent `{}` uses provider `{}` but no matching [llm] entry exists",
                details.name, details.provider
            )
        }
        "memory agent" => format!(
            "memory agent `{}` uses provider `{}` but no matching [llm] entry exists",
            details.name, details.provider
        ),
        "agent image" => format!(
            "agent `{}` uses image provider `{}` but no matching [image] entry exists",
            details.name, details.provider
        ),
        "agent video" => format!(
            "agent `{}` uses video provider `{}` but no matching [video] entry exists",
            details.name, details.provider
        ),
        "agent audio" => format!(
            "agent `{}` uses audio provider `{}` but no matching [audio] entry exists",
            details.name, details.provider
        ),
        _ => format!(
            "{} `{}` uses provider `{}` but no matching [{}] entry exists",
            details.subject, details.name, details.provider, details.table
        ),
    };
    ConfigDiagnostic::new(message)
        .with_label(source.primary_label(path, fallback_path, "missing provider referenced here"))
        .with_optional_help(missing_name_help(
            details.provider,
            &format!("{} provider", details.table),
            details.available,
            Some(format!(
                "add {}",
                table_path(&[details.table, details.provider])
            )),
        ))
}

fn validate_web(
    config: &RuntimeConfig,
    source: &ConfigSource,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if let Err(error) = SocketAddr::from_str(&config.web.listen) {
        diagnostics.push(
            ConfigDiagnostic::new(format!(
                "invalid web listen address `{}`",
                config.web.listen
            ))
            .with_label(source.primary_label(
                &[key("web"), key("listen")],
                &[key("web")],
                "invalid socket address",
            ))
            .with_note(error.to_string())
            .with_help("use `IP:PORT`, for example `127.0.0.1:1860`"),
        );
    }
}

fn missing_name_help(
    name: &str,
    noun: &str,
    available: &BTreeSet<String>,
    fallback: Option<String>,
) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(candidate) = closest_name(name, available) {
        parts.push(format!("did you mean `{candidate}`?"));
    }
    if !available.is_empty() {
        parts.push(format!(
            "available {noun}s: {}",
            available
                .iter()
                .map(|name| format!("`{name}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(fallback) = fallback {
        parts.push(fallback);
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join("; "))
    }
}

fn closest_name<'a>(needle: &str, candidates: &'a BTreeSet<String>) -> Option<&'a str> {
    if needle.is_empty() {
        return None;
    }
    if let Some(prefix_match) = candidates.iter().find(|candidate| {
        let shared_prefix = needle.starts_with(candidate.as_str()) || candidate.starts_with(needle);
        shared_prefix && needle.chars().count().min(candidate.chars().count()) >= 3
    }) {
        return Some(prefix_match);
    }
    let max_distance = (needle.chars().count() / 3).max(2);
    candidates
        .iter()
        .map(|candidate| (edit_distance(needle, candidate), candidate.as_str()))
        .filter(|(distance, _)| *distance <= max_distance)
        .min_by_key(|(distance, candidate)| (*distance, candidate.len()))
        .map(|(_, candidate)| candidate)
}

fn edit_distance(left: &str, right: &str) -> usize {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    let mut costs = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_char) in left.iter().enumerate() {
        let mut previous = costs[0];
        costs[0] = left_index + 1;
        for (right_index, right_char) in right.iter().enumerate() {
            let insertion = costs[right_index + 1] + 1;
            let deletion = costs[right_index] + 1;
            let substitution = previous + usize::from(left_char != right_char);
            previous = costs[right_index + 1];
            costs[right_index + 1] = insertion.min(deletion).min(substitution);
        }
    }
    costs[right.len()]
}

fn table_path(parts: &[&str]) -> String {
    format!(
        "[{}]",
        parts
            .iter()
            .map(|part| toml_key(part))
            .collect::<Vec<_>>()
            .join(".")
    )
}

fn config_path(parts: &[PathPart<'_>]) -> String {
    let mut out = String::new();
    for part in parts {
        match part {
            PathPart::Key(key) => {
                if !out.is_empty() {
                    out.push('.');
                }
                out.push_str(&toml_key(key));
            }
            PathPart::Index(index) => {
                let _ = write!(out, "[{index}]");
            }
        }
    }
    if out.is_empty() {
        "top level".to_string()
    } else {
        out
    }
}

fn toml_key(value: &str) -> String {
    if !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
    {
        return value.to_string();
    }
    let mut quoted = String::from("\"");
    for ch in value.chars() {
        match ch {
            '\\' => quoted.push_str("\\\\"),
            '"' => quoted.push_str("\\\""),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            ch => quoted.push(ch),
        }
    }
    quoted.push('"');
    quoted
}

/// Renders TOML parser/deserializer failures that occur before semantic
/// validation can run.
pub(crate) fn render_toml_error_for_stderr(
    path: &Path,
    input: &str,
    error: &toml::de::Error,
) -> String {
    render_toml_error_with_style(
        path,
        input,
        error,
        DiagnosticStyle::new(stderr_supports_color()),
    )
}

fn render_toml_error_with_style(
    path: &Path,
    input: &str,
    error: &toml::de::Error,
    style: DiagnosticStyle,
) -> String {
    let source = ConfigSource::new(path.to_path_buf(), input.to_string());
    let mut diagnostic =
        ConfigDiagnostic::new(format!("could not parse config file `{}`", path.display()))
            .with_note(error.message().to_string());
    if let Some(span) = error.span() {
        // Parser errors already carry their own span, so reuse the same
        // diagnostic renderer instead of special-casing TOML errors elsewhere.
        diagnostic.labels.push(DiagnosticLabel {
            span,
            message: "TOML could not be decoded here".to_string(),
        });
    } else {
        diagnostic = diagnostic.with_note(error.to_string());
    }
    let mut out = String::new();
    diagnostic.render(
        source.path(),
        source.input(),
        &source.lines,
        style,
        &mut out,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invalid_config() -> (&'static str, RuntimeConfig, ConfigSource) {
        let input = r#"
[database]
url = ""

[logging]
filter = "info,]"

[web]
listen = "not-a-socket"
title_prefix = "Chudbot"
frontend_dir = "frontend-build"

[memory]
enabled = true
compaction_interval = "forever"
diary_backfill_window = "0s"
diary_interval = "24h"

[bot]
web_base_url = "http://localhost:1860"
default_agent = "missing-default"
admins = []
limits = { max_iterations = 8 }

[bot.platforms.discord]
agent = "missing-platform-agent"

[bot.agents.default]
provider = "missing-llm"
system_prompt = "hi"
model = { id = "gpt-test", server_tools = [], sampling = {} }

[bot.agents.default.image_generation]
provider = "missing-image"
model = ""
rate_limit = { limit = 1 }

[bot.agents.default.video_generation]
provider = "missing-video"
model = "veo"
rate_limit = { limit = 0, interval = "never", bypass_scopes = [{ platform = "", scope_id = "" }] }

[bot.agents.default.audio_transcription]
provider = "missing-audio"
model = ""
wake_word = " "

[bot.agents.default.subagents.ask]
agent = "missing-subagent"
description = "Ask another agent"

[platforms.discord]
kind = "discord"
token = "token"
"#;
        let config = toml::from_str::<RuntimeConfig>(input).unwrap();
        let source = ConfigSource::new(PathBuf::from("config.test.toml"), input.to_string());
        (input, config, source)
    }

    #[test]
    fn aggregated_validation_reports_multiple_spanned_errors() {
        let (_input, config, source) = invalid_config();
        let report = validate_runtime_config(&config, &source).unwrap_err();

        assert!(report.len() >= 16, "got {} diagnostics", report.len());
        let rendered = report.render();
        assert!(rendered.contains("database.url must not be empty"));
        assert!(rendered.contains("invalid logging filter"));
        assert!(rendered.contains("agent `default` uses provider `missing-llm`"));
        assert!(rendered.contains("provider = \"missing-llm\""));
        assert!(rendered.contains("rate_limit.limit must be greater than zero"));
        assert!(rendered.contains("aborting due to"));
        assert!(!rendered.contains("\x1b["));
    }

    #[test]
    fn colored_render_includes_ansi_sequences_when_enabled() {
        let (_input, config, source) = invalid_config();
        let report = validate_runtime_config(&config, &source).unwrap_err();
        let rendered = report.render_with_style(DiagnosticStyle::new(true));

        assert!(rendered.contains("\x1b[1;31merror\x1b[0m"));
        assert!(rendered.contains("\x1b[34m-->\x1b[0m"));
        assert!(rendered.contains("\x1b[34m|\x1b[0m"));
        assert!(rendered.contains("\x1b[32mhelp\x1b[0m"));
    }

    #[test]
    fn sampling_numbers_reject_toml_only_number_syntax() {
        let input = r#"
[database]
url = "postgres://localhost/chudbot"

[web]
title_prefix = "Chudbot"
frontend_dir = "frontend-build"

[bot]
web_base_url = "http://localhost:1860"
default_agent = "default"

[bot.agents.default]
provider = "grok"
system_prompt = "hi"

[bot.agents.default.model]
id = "grok-test"

[bot.agents.default.model.sampling]
temperature = +1.3
top_p = 1_000

[llm.grok]
kind = "xai"
api_key = "key"
"#;
        let config = toml::from_str::<RuntimeConfig>(input).unwrap();
        let source = ConfigSource::new(PathBuf::from("config.test.toml"), input.to_string());
        let report = validate_runtime_config(&config, &source).unwrap_err();
        let rendered = report.render();

        assert!(rendered.contains(
            "`bot.agents.default.model.sampling.temperature` must be a JSON-compatible number literal"
        ));
        assert!(rendered.contains("temperature = +1.3"));
        assert!(rendered.contains(
            "`bot.agents.default.model.sampling.top_p` must be a JSON-compatible number literal"
        ));
        assert!(rendered.contains("top_p = 1_000"));
        assert!(rendered.contains("avoid TOML-only forms"));
    }

    #[test]
    fn unexpected_keys_are_reported_with_local_suggestions() {
        let input = r#"
[database]
url = "postgres://localhost/chudbot"
pool_size = 5

[web]
listen = "127.0.0.1:1860"
title_prefix = "Chudbot"
frontend_dir = "frontend-build"
frontned_dir = "typo"

[memory]
enabled = true
provider = "stale"
max_diary_output_tokens = 1024
max_profile_output_tokens = 2048

[bot]
web_base_url = "http://localhost:1860"
default_agent = "default"
admins = [{ platform = "discord", user_id = "123", nickname = "Chud" }]

[bot.agents.default]
provider = "openai"
system_prompt = "hi"
persona = "old"

[bot.agents.default.model]
id = "gpt-test"
model_typo = true

[bot.agents.default.model.provider_options]
value = { reasoning_effort = "high", nested = { arbitrary = true } }
unexpected_envelope = true

[bot.agents.default.video_generation]
provider = "grok_video"
model = "grok-imagine-video"

[bot.agents.default.video_generation.rate_limit]
limit = 1
bypass_scopes = [{ platform = "discord", scope_id = "123", guild = "456" }]

[bot.agents.default.subagents.ask]
agent = "default"
description = "Ask self"
extra = true

[llm.openai]
kind = "openai"
api_key = "key"
organization = "old"

[llm.openai.pricing.gpt-test]
input_usd_per_million_tokens = 1.0
output_usd_per_million_tokens = 2.0
typo = 3.0

[video.grok_video]
kind = "xai"
api_key = "key"

[personas.default]
provider = "openai"
"#;
        let config = toml::from_str::<RuntimeConfig>(input).unwrap();
        let source = ConfigSource::new(PathBuf::from("config.test.toml"), input.to_string());
        let report = validate_runtime_config(&config, &source).unwrap_err();
        let rendered = report.render();

        assert!(rendered.contains("unexpected config key `database.pool_size`"));
        assert!(rendered.contains("unexpected config key `web.frontned_dir`"));
        assert!(rendered.contains("did you mean `frontend_dir`?"));
        assert!(rendered.contains("unexpected config key `memory.provider`"));
        assert!(rendered.contains("unexpected config key `memory.max_diary_output_tokens`"));
        assert!(rendered.contains("unexpected config key `memory.max_profile_output_tokens`"));
        assert!(rendered.contains("unexpected config key `bot.admins[0].nickname`"));
        assert!(rendered.contains("unexpected config key `bot.agents.default.persona`"));
        assert!(rendered.contains("unexpected config key `bot.agents.default.model.model_typo`"));
        assert!(rendered.contains(
            "unexpected config key `bot.agents.default.model.provider_options.unexpected_envelope`"
        ));
        assert!(!rendered.contains("reasoning_effort"));
        assert!(rendered.contains(
            "unexpected config key `bot.agents.default.video_generation.rate_limit.bypass_scopes[0].guild`"
        ));
        assert!(
            rendered.contains("unexpected config key `bot.agents.default.subagents.ask.extra`")
        );
        assert!(rendered.contains("unexpected config key `llm.openai.organization`"));
        assert!(rendered.contains("unexpected config key `llm.openai.pricing.gpt-test.typo`"));
        assert!(rendered.contains("unexpected config section `personas`"));
    }

    #[test]
    fn source_map_handles_quoted_keys_as_single_path_segment() {
        let input = r#"
[database]
url = "postgres://localhost/db"

[web]
title_prefix = "Chudbot"
frontend_dir = "frontend-build"

[bot]
web_base_url = "http://localhost:1860"
default_agent = "agent.with.dot"

[bot.agents."agent.with.dot"]
provider = "missing"
system_prompt = "hi"
model = { id = "gpt-test" }
"#;
        let config = toml::from_str::<RuntimeConfig>(input).unwrap();
        let source = ConfigSource::new(PathBuf::from("config.test.toml"), input.to_string());
        let report = validate_runtime_config(&config, &source).unwrap_err();
        let rendered = report.render();

        assert!(rendered.contains("provider = \"missing\""));
        assert!(rendered.contains("[llm.missing]"));
    }

    #[test]
    fn s3_storage_reports_missing_bucket_and_local_keys() {
        let input = r#"
[database]
url = "postgres://localhost/db"

[web]
title_prefix = "Chudbot"
frontend_dir = "frontend-build"

[storage]
kind = "s3"
region = ""
endpoint_url = ""
images_dir = "images"

[bot]
web_base_url = "http://localhost:1860"
default_agent = "default"

[bot.agents.default]
provider = "openai"
system_prompt = "hi"
model = { id = "gpt-test" }

[llm.openai]
kind = "openai"
api_key = "key"
"#;
        let config = toml::from_str::<RuntimeConfig>(input).unwrap();
        let source = ConfigSource::new(PathBuf::from("config.test.toml"), input.to_string());
        let report = validate_runtime_config(&config, &source).unwrap_err();
        let rendered = report.render();

        assert!(rendered.contains("unexpected config key `storage.images_dir`"));
        assert!(rendered.contains("storage.bucket must not be empty"));
        assert!(rendered.contains("storage.region must not be empty when set"));
        assert!(rendered.contains("storage.endpoint_url must not be empty when set"));
    }

    #[test]
    fn local_storage_defaults_without_kind_and_rejects_s3_keys() {
        let input = r#"
[database]
url = "postgres://localhost/db"

[web]
title_prefix = "Chudbot"
frontend_dir = "frontend-build"

[storage]
images_dir = "images"
bucket = "assets"

[bot]
web_base_url = "http://localhost:1860"
default_agent = "default"

[bot.agents.default]
provider = "openai"
system_prompt = "hi"
model = { id = "gpt-test" }

[llm.openai]
kind = "openai"
api_key = "key"
"#;
        let config = toml::from_str::<RuntimeConfig>(input).unwrap();
        let source = ConfigSource::new(PathBuf::from("config.test.toml"), input.to_string());
        let report = validate_runtime_config(&config, &source).unwrap_err();
        let rendered = report.render();

        assert!(matches!(config.storage, StorageConfig::Local(_)));
        assert!(rendered.contains("unexpected config key `storage.bucket`"));
    }

    #[test]
    fn closest_name_accepts_provider_prefix_typos() {
        let names = BTreeSet::from(["anthropic".to_string(), "grok".to_string()]);

        assert_eq!(closest_name("grok_typo", &names), Some("grok"));
    }
}
