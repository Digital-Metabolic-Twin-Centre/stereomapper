# Changelog

## [2.0.1] - 2026-11-11

### Fixed

- Pair comparison now uses the primary stereo classification. Previously, `_analyze_pair` passed
  `isRadio1`/`isRadio2` instead of `is_radio1`/`is_radio2` to
  `RelationshipAnalyser.calc_relationship`. The resulting `TypeError` was caught, so every pair
  with available structures was classified by the InChI fallback.
- The InChI fallback is used again for genuine no-classification primary results
  (`Unclassified`, SMRO:0008). The check previously compared against the string
  `"No classification"`, which the classifier never produces.

### Changed

- A `TypeError`, `AttributeError` or `NameError` raised during primary analysis now ends the
  run, instead of silently sending the pair to the fallback. Other errors, such as invalid
  structures, still fall back as before.

### Compatibility notes

- Results produced before this fix were derived from the fallback and carry the
  `used_fallback_method` penalty. Regenerate them. Classification values and scores may change
  for affected pairs. The CLI, output fields and SQLite schema are unchanged.
