//! Utility functions for the Animus project

/// Safely truncate a string to a maximum number of characters,
/// respecting UTF-8 character boundaries.
pub fn safe_truncate(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        None => s,
        Some((idx, _)) => &s[..idx],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_truncate_ascii() {
        let s = "Hello, world!";
        assert_eq!(safe_truncate(s, 5), "Hello");
        assert_eq!(safe_truncate(s, 20), "Hello, world!");
    }

    #[test]
    fn test_safe_truncate_utf8() {
        let s = "🦀 Rust is awesome! 🦀";
        // 🦀 is 1 character but 4 bytes
        assert_eq!(safe_truncate(s, 1), "🦀");
        assert_eq!(safe_truncate(s, 6), "🦀 Rust");
    }
}
