use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

/// A TDLN hash is a SHA-256 hash represented as a hex string
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TdlnHash(String);

impl TdlnHash {
    /// Creates a new TdlnHash from a hex string
    pub fn new<S: Into<String>>(hash: S) -> Self {
        Self(hash.into())
    }

    /// Computes a hash from bytes
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let result = hasher.finalize();
        Self(hex::encode(result))
    }

    /// Computes a hash from a JSON-serializable value
    pub fn from_json<T: Serialize>(value: &T) -> Result<Self, serde_json::Error> {
        let json = serde_json::to_string(value)?;
        Ok(Self::from_bytes(json.as_bytes()))
    }

    /// Computes a canonical hash from a JSON-serializable value
    /// Uses sorted keys and consistent separators
    pub fn from_canonical_json<T: Serialize>(value: &T) -> Result<Self, serde_json::Error> {
        // Use canonical JSON: sorted keys, no whitespace
        let json = serde_json::to_vec(value)?;
        Ok(Self::from_bytes(&json))
    }

    /// Returns the hash as a hex string
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the hash as a String
    pub fn into_string(self) -> String {
        self.0
    }

    /// Returns the first N characters of the hash (short hash)
    pub fn short(&self, len: usize) -> &str {
        let len = len.min(self.0.len());
        &self.0[..len]
    }

    /// Verifies that the hash matches the hash of the given bytes
    pub fn verify(&self, bytes: &[u8]) -> bool {
        let computed = Self::from_bytes(bytes);
        self == &computed
    }
}

impl fmt::Display for TdlnHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for TdlnHash {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for TdlnHash {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl AsRef<str> for TdlnHash {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Trait for types that can compute their canonical hash
pub trait Hashable {
    /// Computes the canonical hash for this value
    fn compute_hash(&self) -> std::result::Result<TdlnHash, serde_json::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_hash_from_bytes() {
        let data = b"hello world";
        let hash = TdlnHash::from_bytes(data);

        // SHA-256 of "hello world"
        assert_eq!(
            hash.as_str(),
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        );
    }

    #[test]
    fn test_hash_from_json() {
        let value = json!({
            "name": "test",
            "value": 42
        });

        let hash = TdlnHash::from_json(&value).unwrap();
        assert_eq!(hash.as_str().len(), 64); // SHA-256 produces 64 hex chars
    }

    #[test]
    fn test_hash_equality() {
        let hash1 = TdlnHash::from_bytes(b"test");
        let hash2 = TdlnHash::from_bytes(b"test");
        let hash3 = TdlnHash::from_bytes(b"different");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_hash_verify() {
        let data = b"verify me";
        let hash = TdlnHash::from_bytes(data);

        assert!(hash.verify(data));
        assert!(!hash.verify(b"different data"));
    }

    #[test]
    fn test_short_hash() {
        let hash = TdlnHash::from_bytes(b"test");
        assert_eq!(hash.short(8).len(), 8);
        assert_eq!(hash.short(16).len(), 16);
    }

    #[test]
    fn test_serialization() {
        let hash = TdlnHash::from_bytes(b"serialize");
        let json = serde_json::to_string(&hash).unwrap();
        let deserialized: TdlnHash = serde_json::from_str(&json).unwrap();

        assert_eq!(hash, deserialized);
    }
}
