//! Small Server-Sent Events decoder shared by streaming provider adapters.
//!
//! The decoder is intentionally transport-agnostic: providers feed byte chunks
//! from reqwest and receive complete SSE events with joined `data:` lines.

/// One decoded Server-Sent Event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerSentEvent {
    /// Optional `event:` field.
    pub event: Option<String>,
    /// Joined `data:` field lines.
    pub data: String,
}

/// Incremental SSE decoder.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    /// Create an empty decoder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a transport chunk and return all complete events it contains.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<ServerSentEvent>, SseDecodeError> {
        self.buffer.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some((index, boundary_len)) = find_event_boundary(&self.buffer) {
            let event = self.buffer.drain(..index).collect::<Vec<_>>();
            self.buffer.drain(..boundary_len);
            if let Some(event) = parse_event(&event)? {
                events.push(event);
            }
        }
        Ok(events)
    }

    /// Finish the stream and decode a trailing unterminated event, if any.
    pub fn finish(&mut self) -> Result<Option<ServerSentEvent>, SseDecodeError> {
        if self.buffer.is_empty() {
            return Ok(None);
        }
        let event = std::mem::take(&mut self.buffer);
        parse_event(&event)
    }
}

/// SSE decoding failure.
#[derive(Debug, thiserror::Error)]
pub enum SseDecodeError {
    /// Provider emitted non-UTF-8 SSE data.
    #[error("SSE event was not valid UTF-8: {0}")]
    Utf8(#[from] std::str::Utf8Error),
}

fn find_event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let mut index = 0;
    while index < buffer.len() {
        if buffer.get(index..index + 2) == Some(b"\n\n") {
            return Some((index, 2));
        }
        if buffer.get(index..index + 4) == Some(b"\r\n\r\n") {
            return Some((index, 4));
        }
        index += 1;
    }
    None
}

fn parse_event(bytes: &[u8]) -> Result<Option<ServerSentEvent>, SseDecodeError> {
    let text = std::str::from_utf8(bytes)?;
    let mut event = None;
    let mut data = Vec::new();
    for raw_line in text.lines() {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() || line.starts_with(':') {
            continue;
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => event = Some(value.to_string()),
            "data" => data.push(value.to_string()),
            _ => {}
        }
    }
    if event.is_none() && data.is_empty() {
        return Ok(None);
    }
    Ok(Some(ServerSentEvent {
        event,
        data: data.join("\n"),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_split_sse_events() {
        let mut decoder = SseDecoder::new();
        assert!(
            decoder
                .push(b"event: message\ndata: {\"a\"")
                .unwrap()
                .is_empty()
        );
        let events = decoder.push(b":1}\n\n").unwrap();
        assert_eq!(
            events,
            vec![ServerSentEvent {
                event: Some("message".to_string()),
                data: "{\"a\":1}".to_string(),
            }]
        );
    }

    #[test]
    fn joins_multiple_data_lines() {
        let mut decoder = SseDecoder::new();
        let events = decoder.push(b"data: one\ndata: two\r\n\r\n").unwrap();
        assert_eq!(events[0].data, "one\ntwo");
    }
}
