use std::collections::{BTreeMap, BTreeSet};
use std::error::Error as _;
use std::fmt;
use std::fmt::Write as _;
use std::net::SocketAddr;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use chudbot_api::ProviderName;
use chudbot_bot::{GenerationBinding, TranscriptionBinding, VideoGenerationRateLimit};
use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};

use crate::config::RuntimeConfig;

const DATETIME_FIELD: &str = "$__toml_private_datetime";
const MAX_SOURCE_LINE_CHARS: usize = 160;
const SOURCE_WINDOW_CHARS: usize = 120;

#[derive(Debug, Clone)]
pub(crate) struct ConfigSource {
    path: PathBuf,
    input: String,
    root: Option<toml::Spanned<SourceNode>>,
    lines: LineIndex,
}

impl ConfigSource {
    pub(crate) fn new(path: PathBuf, input: String) -> Self {
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

    fn span_for(&self, path: &[PathPart<'_>]) -> Option<Range<usize>> {
        let mut value = self.root.as_ref()?;
        for part in path {
            value = match (value.get_ref(), part) {
                (SourceNode::Table(entries), PathPart::Key(key)) => entries.get(*key)?,
                (SourceNode::Array(items), PathPart::Index(index)) => items.get(*index)?,
                _ => return None,
            };
        }
        Some(value.span())
    }

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

#[derive(Debug, Clone)]
enum SourceNode {
    Scalar,
    Array(Vec<toml::Spanned<SourceNode>>),
    Table(BTreeMap<String, toml::Spanned<SourceNode>>),
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
                let _ = map.next_value::<IgnoredAny>()?;
                return Ok(SourceNode::Scalar);
            }
            let value = map.next_value::<toml::Spanned<SourceNode>>()?;
            entries.insert(key, value);
        }
        Ok(SourceNode::Table(entries))
    }
}

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

#[derive(Debug, Clone)]
pub(crate) struct ConfigValidationReport {
    path: PathBuf,
    input: String,
    lines: LineIndex,
    diagnostics: Vec<ConfigDiagnostic>,
}

impl ConfigValidationReport {
    fn new(source: &ConfigSource, diagnostics: Vec<ConfigDiagnostic>) -> Self {
        Self {
            path: source.path.clone(),
            input: source.input.clone(),
            lines: source.lines.clone(),
            diagnostics,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.diagnostics.len()
    }

    pub(crate) fn render(&self) -> String {
        let mut out = String::new();
        for (index, diagnostic) in self.diagnostics.iter().enumerate() {
            if index > 0 {
                out.push('\n');
            }
            diagnostic.render(&self.path, &self.input, &self.lines, &mut out);
        }
        if self.diagnostics.len() > 1 {
            let _ = writeln!(
                out,
                "\nerror: aborting due to {} config errors",
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

    fn render(&self, path: &Path, input: &str, lines: &LineIndex, out: &mut String) {
        let _ = writeln!(out, "error: {}", self.message);
        if let Some(primary) = self.labels.first() {
            let (line, column) = lines.line_col(input, primary.span.start);
            let _ = writeln!(out, "  --> {}:{}:{}", path.display(), line + 1, column + 1);
            write_gutter_separator(out, line + 1);
            for label in &self.labels {
                render_label(input, lines, label, out);
            }
        }
        for note in &self.notes {
            let _ = writeln!(out, "  = note: {note}");
        }
        if let Some(help) = &self.help {
            let _ = writeln!(out, "  = help: {help}");
        }
    }
}

#[derive(Debug, Clone)]
struct DiagnosticLabel {
    span: Range<usize>,
    message: String,
}

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

fn write_gutter_separator(out: &mut String, max_line_number: usize) {
    let width = max_line_number.to_string().len();
    let _ = writeln!(out, "{:width$} |", "", width = width);
}

fn render_label(input: &str, lines: &LineIndex, label: &DiagnosticLabel, out: &mut String) {
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
                let line_text = input.get(line_range.clone()).unwrap_or_default();
                let highlight_start_byte = if line == start_line {
                    start.saturating_sub(line_range.start)
                } else {
                    0
                };
                let highlight_end_byte = if line == end_line {
                    end.saturating_sub(line_range.start)
                } else {
                    line_range.len()
                };
                let start_col = byte_to_char_col(line_text, highlight_start_byte);
                let end_col = byte_to_char_col(line_text, highlight_end_byte).max(start_col + 1);
                let window = SourceLineWindow::new(line_text, start_col, end_col);
                let _ = writeln!(out, "{:>width$} | {}", line + 1, window.text, width = width);
                let caret_len = window.end_col.saturating_sub(window.start_col).max(1);
                let _ = writeln!(
                    out,
                    "{:width$} | {}{} {}",
                    "",
                    " ".repeat(window.start_col),
                    "^".repeat(caret_len),
                    label.message,
                    width = width
                );
            }
            RenderedLine::Ellipsis => {
                let _ = writeln!(out, "{:width$} | ...", "", width = width);
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

pub(crate) fn validate_runtime_config(
    config: &RuntimeConfig,
    source: &ConfigSource,
) -> Result<(), ConfigValidationReport> {
    let mut diagnostics = Vec::new();
    validate_database(config, source, &mut diagnostics);
    validate_logging(config, source, &mut diagnostics);
    validate_bot_config(config, source, &mut diagnostics);
    validate_memory_durations(config, source, &mut diagnostics);
    validate_runtime_references(config, source, &mut diagnostics);
    validate_web(config, source, &mut diagnostics);

    if diagnostics.is_empty() {
        Ok(())
    } else {
        Err(ConfigValidationReport::new(source, diagnostics))
    }
}

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

pub(crate) fn render_toml_error(path: &Path, input: &str, error: &toml::de::Error) -> String {
    let source = ConfigSource::new(path.to_path_buf(), input.to_string());
    let mut diagnostic =
        ConfigDiagnostic::new(format!("could not parse config file `{}`", path.display()))
            .with_note(error.message().to_string());
    if let Some(span) = error.span() {
        diagnostic.labels.push(DiagnosticLabel {
            span,
            message: "TOML could not be decoded here".to_string(),
        });
    } else {
        diagnostic = diagnostic.with_note(error.to_string());
    }
    let mut out = String::new();
    diagnostic.render(source.path(), source.input(), &source.lines, &mut out);
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
    fn closest_name_accepts_provider_prefix_typos() {
        let names = BTreeSet::from(["anthropic".to_string(), "grok".to_string()]);

        assert_eq!(closest_name("grok_typo", &names), Some("grok"));
    }
}
