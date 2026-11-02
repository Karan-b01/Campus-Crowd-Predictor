# Chronological Data Split Summary

Rows are sorted by parsed UTC timestamp before splitting. The configured fractions are 70% train, 15% validation, and 15% test; integer boundaries use floor rounding, with any remainder assigned to test. The test partition is reserved for final reporting and is not used in baseline selection.

| Partition | Rows | First timestamp (UTC) | Last timestamp (UTC) |
| --- | ---: | --- | --- |
| Train | 43,528 | 2015-08-15T00:00:11+00:00 | 2016-09-11T02:26:48+00:00 |
| Validation | 9,328 | 2016-09-11T02:36:51+00:00 | 2016-11-18T03:47:16+00:00 |
| Test | 9,328 | 2016-11-18T03:57:22+00:00 | 2017-03-19T02:22:51+00:00 |
