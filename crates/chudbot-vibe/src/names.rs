use std::collections::BTreeSet;

const BUILTIN_RESERVED: [&str; 7] = ["src", "www", "api", "admin", "static", "status", "mail"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameAvailability {
    Available,
    Unavailable,
    Invalid,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NameCheck {
    pub name: String,
    pub availability: NameAvailability,
}

#[derive(Debug, Clone)]
pub struct VibeNames {
    reserved: BTreeSet<String>,
}

impl VibeNames {
    pub fn new(configured: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let mut reserved = BUILTIN_RESERVED
            .into_iter()
            .map(str::to_string)
            .collect::<BTreeSet<_>>();
        reserved.extend(
            configured
                .into_iter()
                .map(Into::into)
                .map(|name| name.to_ascii_lowercase()),
        );
        Self { reserved }
    }

    pub fn validate(&self, name: &str) -> Result<(), &'static str> {
        if !(3..=63).contains(&name.len()) {
            return Err("name must contain 3 to 63 characters");
        }
        if !name.is_ascii() || name != name.to_ascii_lowercase() {
            return Err("name must be lowercase ASCII");
        }
        if name.starts_with("xn--") {
            return Err("internationalized DNS labels are reserved");
        }
        if self.reserved.contains(name) {
            return Err("name is reserved");
        }
        let bytes = name.as_bytes();
        if !bytes[0].is_ascii_alphanumeric() || !bytes[bytes.len() - 1].is_ascii_alphanumeric() {
            return Err("name must start and end with a letter or digit");
        }
        if !bytes
            .iter()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || *byte == b'-')
        {
            return Err("name may contain only lowercase letters, digits, and hyphens");
        }
        Ok(())
    }

    pub fn check_batch<'a>(
        &self,
        candidates: impl IntoIterator<Item = &'a str>,
        unavailable: &BTreeSet<String>,
    ) -> Result<Vec<NameCheck>, &'static str> {
        let candidates = candidates.into_iter().collect::<Vec<_>>();
        if candidates.len() > 8 {
            return Err("at most 8 names may be checked at once");
        }
        Ok(candidates
            .into_iter()
            .map(|name| NameCheck {
                name: name.to_string(),
                availability: if self.validate(name).is_err() {
                    NameAvailability::Invalid
                } else if unavailable.contains(name) {
                    NameAvailability::Unavailable
                } else {
                    NameAvailability::Available
                },
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test_case("abc", true; "minimum valid")]
    #[test_case("mortgage-curves", true; "hyphenated")]
    #[test_case("a1-2z", true; "alphanumeric")]
    #[test_case("ab", false; "too short")]
    #[test_case("-abc", false; "leading hyphen")]
    #[test_case("abc-", false; "trailing hyphen")]
    #[test_case("ABC", false; "uppercase")]
    #[test_case("a_b", false; "underscore")]
    #[test_case("xn--abc", false; "punycode")]
    #[test_case("src", false; "reserved")]
    fn validates_dns_labels(name: &str, valid: bool) {
        assert_eq!(
            VibeNames::new(Vec::<String>::new()).validate(name).is_ok(),
            valid
        );
    }

    #[test]
    fn batch_checks_preserve_candidates_and_do_not_reserve() {
        let names = VibeNames::new(["custom"]);
        let unavailable = BTreeSet::from(["taken".to_string()]);
        let checked = names
            .check_batch(["fresh", "taken", "Bad", "custom"], &unavailable)
            .unwrap();
        assert_eq!(
            checked
                .iter()
                .map(|item| item.availability)
                .collect::<Vec<_>>(),
            vec![
                NameAvailability::Available,
                NameAvailability::Unavailable,
                NameAvailability::Invalid,
                NameAvailability::Invalid
            ]
        );
        assert_eq!(
            names.check_batch(["fresh"], &BTreeSet::new()).unwrap()[0].availability,
            NameAvailability::Available
        );
    }

    #[test]
    fn batch_rejects_more_than_eight_candidates() {
        let candidates = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ];
        assert!(
            VibeNames::new(Vec::<String>::new())
                .check_batch(candidates, &BTreeSet::new())
                .is_err()
        );
    }
}
