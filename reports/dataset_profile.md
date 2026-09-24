# Dataset forensics report

_Generated 2026-09-25 by `python run_pipeline.py profile` (`src/business_entity_resolution/data/profiling.py`). Every number below is computed from the supplied files; sampled statistics are labelled as such._

## 1. Dataset overview

All four columns are `entity_id`, `business_name`, `business_address`, `country`; every column is read as a string (`dtype=str`, `quoting=QUOTE_NONE`, `na_filter=False`).

| file | TSV MB | data rows | cols | malformed rows (≠4 fields) | rows containing `"` | rows with CR | BOM | Parquet MB | in-memory GB |
|---|---|---|---|---|---|---|---|---|---|
| train_source1.tsv | 210.1 | 2,206,821 | 4 | 0 | 4 | 0 | False | 136.7 | 0.272 |
| train_source2.tsv | 489.3 | 5,034,616 | 4 | 0 | 6 | 0 | False | 332.0 | 0.63 |
| train_source3.tsv | 503.7 | 5,285,603 | 4 | 0 | 0 | 0 | False | 338.9 | 0.652 |
| test_source1.tsv | 175.0 | 1,732,544 | 4 | 0 | 134 | 0 | False | 110.5 | 0.224 |
| test_source2.tsv | 509.5 | 4,887,273 | 4 | 0 | 349 | 0 | False | 336.0 | 0.646 |
| test_source3.tsv | 506.0 | 5,082,316 | 4 | 0 | 330 | 0 | False | 335.6 | 0.648 |

**Integrity checks on the loaded frames** (all should be 0 except `n`):

| file | n | duplicate_ids | bad_prefix | empty_id | empty_country |
|---|---|---|---|---|---|
| train_s1 | 2,206,821 | 0 | 0 | 0 | 0 |
| train_s2 | 5,034,616 | 0 | 0 | 0 | 0 |
| train_s3 | 5,285,603 | 0 | 0 | 0 | 0 |
| test_s1 | 1,732,544 | 0 | 0 | 0 | 0 |
| test_s2 | 4,887,273 | 0 | 0 | 0 | 0 |
| test_s3 | 5,082,316 | 0 | 0 | 0 | 0 |

## 2. Source statistics and country analysis

| file | rows | France | India | US |
|---|---|---|---|---|
| train_s1 | 2,206,821 | 0 (0.0%) | 883,188 (40.0%) | 1,323,633 (60.0%) |
| train_s2 | 5,034,616 | 0 (0.0%) | 2,017,799 (40.1%) | 3,016,817 (59.9%) |
| train_s3 | 5,285,603 | 0 (0.0%) | 2,115,547 (40.0%) | 3,170,056 (60.0%) |
| test_s1 | 1,732,544 | 259,452 (15.0%) | 809,986 (46.8%) | 663,106 (38.3%) |
| test_s2 | 4,887,273 | 703,378 (14.4%) | 2,312,565 (47.3%) | 1,871,330 (38.3%) |
| test_s3 | 5,082,316 | 731,615 (14.4%) | 2,405,000 (47.3%) | 1,945,701 (38.3%) |

Records per S1 entity:

| split | S2 / S1 | S3 / S1 |
|---|---|---|
| train | 2.28 | 2.4 |
| test | 2.82 | 2.93 |

## 3. Missingness

| file | empty names | empty addresses | empty address % | empty country | address contains NULL/N/A literal |
|---|---|---|---|---|---|
| train_s1 | 0 | 0 | 0.00% | 0 | 0.00% |
| train_s2 | 0 | 168,967 | 3.36% | 0 | 3.49% |
| train_s3 | 0 | 175,916 | 3.33% | 0 | 3.31% |
| test_s1 | 0 | 0 | 0.00% | 0 | 0.00% |
| test_s2 | 0 | 129,408 | 2.65% | 0 | 2.88% |
| test_s3 | 0 | 136,098 | 2.68% | 0 | 2.77% |

Empty-address rate by country:

| file | ALL | France | India | US |
|---|---|---|---|---|
| train_s1 | 0.00% | – | 0.00% | 0.00% |
| train_s2 | 3.36% | – | 2.87% | 3.68% |
| train_s3 | 3.33% | – | 3.07% | 3.50% |
| test_s1 | 0.00% | 0.00% | 0.00% | 0.00% |
| test_s2 | 2.65% | 3.06% | 2.28% | 2.94% |
| test_s3 | 2.68% | 2.94% | 2.46% | 2.84% |

## 4. Duplicate analysis

| file | distinct_names | rows_sharing_exact_name | distinct_addresses | rows_sharing_exact_address | rows_sharing_name_and_address | rows_sharing_name_address_country | rows_sharing_normalized_name | distinct_normalized_names |
|---|---|---|---|---|---|---|---|---|
| train_s1 | 1,539,229 | 845,385 | 2,130,606 | 116,304 | 0 | 0 | 866,894 | 1,520,684 |
| train_s2 | 4,402,009 | 872,386 | 4,337,261 | 949,860 | 50,969 | 50,933 | 1,526,821 | 3,930,601 |
| train_s3 | 4,651,609 | 892,270 | 4,632,764 | 859,813 | 37,283 | 37,241 | 1,545,063 | 4,176,057 |
| test_s1 | 1,238,867 | 623,632 | 1,677,483 | 86,457 | 0 | 0 | 637,346 | 1,227,453 |
| test_s2 | 4,311,041 | 799,335 | 4,224,783 | 977,780 | 44,552 | 44,550 | 1,356,607 | 3,913,509 |
| test_s3 | 4,521,929 | 798,543 | 4,456,435 | 894,539 | 32,178 | 32,154 | 1,361,833 | 4,111,915 |

Most frequent exact business names (count):

- **train_s1**: `Primary Care Group` (253), `Ear Nose & Throat Group` (251), `Pediatric Group` (222), `Womens Health Group` (220), `Physical Therapy Group` (218), `Pediatric Dental Group` (216), `Behavioral Health Group` (215), `Chiropractic Group` (209)
- **train_s2**: `Primary Care` (320), `Physical Therapy` (307), `Urgent Care` (297), `Womens Health` (297), `Behavioral Health` (296), `Internal Medicine` (288), `Pediatric Dentistry` (274), `earnosethroat.com` (269)
- **train_s3**: `Primary Care` (421), `Physical Therapy` (399), `Pediatric Dental` (393), `Womens Health` (379), `Urgent Care` (377), `Pediatric Dentistry` (361), `Behavioral Health` (330), `Pediatric` (321)
- **test_s1**: `Bordeaux Club SARL` (205), `Nantes Club SARL` (157), `Lille Club SARL` (147), `Urgent Care Group` (123), `Lille Club SAS` (122), `Bordeaux Club SAS` (118), `Nantes Club SAS` (117), `Bordeaux Amicale SARL` (116)
- **test_s2**: `CC` (302), `PC` (197), `LC` (194), `AC` (174), `SC` (174), `DC` (159), `Urgent Care` (156), `Womens Health` (152)
- **test_s3**: `CC` (387), `SC` (358), `AC` (297), `PC` (286), `LC` (245), `MC` (243), `BC` (226), `CS` (218)

Most frequent exact addresses (count):

- **train_s1**: `108 Norle Street, College Twp, PA` (14), `104 Roadrunner Circle, Elephant Butte, NM` (13), `218 Phillips Street, Storm Lake, IA` (13), `43 Franklin Street, Delaware, OH` (13)
- **train_s2**: `842 38, CALEDONAI TOWNSHIP, WI` (18), `1948 WEAVER FOREST WAY, MORRISVILLE, NC` (18), `66 FRANK EDGE, ALTO, TX` (17), `G-58/1, BENSON CROSS ROAD, BANGALORE., Karnataka` (16)
- **train_s3**: `Ground Floor, Bangalore, KA` (29), `Floor, Mumbai, MH` (26), `18, Kolkata, Howrah, WB` (26), `303, Mumbai, MH` (25)
- **test_s1**: `12 RUE Lyderic, Lille, Hauts-de-France` (99), `27 RUE Jean Bart, Lille, Hauts-de-France` (85), `30 RUE des Meuniers, Lille, Hauts-de-France` (72), `27 RUE Jean Bart, Maison des Associations, Lille, Hauts-de-France` (67)
- **test_s2**: `CALAIS` (81), `27 RUE JEAN BART, LILLE` (79), `BORDEAUX` (64), `RUE LYDERIC, LILLE` (64)
- **test_s3**: `Calais` (89), `27 Rue Jean Bart, Lille` (84), `Calais, Pas-de-Calais` (74), `12 Rue Lyderic, Lille, Nord` (72)

## 5. Name statistics

### Length

Characters:

|  | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| train_s1 | 8 | 12.0 | 18.0 | 24.0 | 30.0 | 37.0 | 42.0 | 105.0 | 24.0 |
| train_s2 | 8 | 12.0 | 19.0 | 25.0 | 31.0 | 40.0 | 48.0 | 104.0 | 25.1 |
| train_s3 | 6 | 11.0 | 18.0 | 25.0 | 31.0 | 42.0 | 50.0 | 123.0 | 25.2 |
| test_s1 | 8 | 12.0 | 18.0 | 24.0 | 29.0 | 36.0 | 42.0 | 92.0 | 23.8 |
| test_s2 | 8 | 12.0 | 19.0 | 25.0 | 32.0 | 42.0 | 49.0 | 102.0 | 25.7 |
| test_s3 | 6 | 11.0 | 19.0 | 25.0 | 32.0 | 42.0 | 50.0 | 103.0 | 25.7 |

Whitespace tokens:

|  | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| train_s1 | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 16.0 | 3.55 |
| train_s2 | 1 | 1 | 3 | 4 | 4 | 5 | 6 | 15.0 | 3.5 |
| train_s3 | 1 | 1 | 3 | 4 | 4 | 6 | 7 | 18.0 | 3.53 |
| test_s1 | 2 | 2 | 3 | 4 | 4 | 5 | 6 | 14.0 | 3.52 |
| test_s2 | 1 | 2 | 3 | 4 | 4 | 5 | 6 | 15.0 | 3.59 |
| test_s3 | 1 | 1 | 3 | 4 | 4 | 6 | 7 | 16.0 | 3.6 |

Median name length (chars) by country:

| file | medians |
|---|---|
| train_s1 | India: 27 | US: 22 |
| train_s2 | India: 27 | US: 23 |
| train_s3 | India: 27 | US: 23 |
| test_s1 | France: 19 | India: 27 | US: 22 |
| test_s2 | France: 20 | India: 28 | US: 24 |
| test_s3 | France: 20 | India: 28 | US: 24 |

### Casing

| file | ALL UPPER | all lower | mixed |
|---|---|---|---|
| train_s1 | 0.00% | 0.00% | 100.00% |
| train_s2 | 18.90% | 5.75% | 75.35% |
| train_s3 | 2.96% | 6.25% | 90.80% |
| test_s1 | 0.00% | 0.00% | 100.00% |
| test_s2 | 17.49% | 5.03% | 77.48% |
| test_s3 | 3.20% | 5.53% | 91.27% |

### Noise patterns (share of rows, all countries)

| pattern | train_s1 | train_s2 | train_s3 | test_s1 | test_s2 | test_s3 |
|---|---|---|---|---|---|---|
| non_ascii | 0.00% | 15.19% | 11.48% | 2.35% | 18.99% | 14.51% |
| has_digit | 1.61% | 5.08% | 5.12% | 1.17% | 3.89% | 3.98% |
| ocr_digit_in_word | 0.00% | 1.79% | 1.90% | 0.00% | 1.37% | 1.48% |
| ampersand | 5.07% | 4.15% | 4.10% | 5.02% | 4.27% | 4.19% |
| acronym_only | 0.00% | 0.05% | 0.25% | 0.00% | 0.24% | 0.47% |
| domain_name | 0.00% | 4.00% | 3.99% | 0.00% | 3.19% | 3.23% |
| alias_marker | 0.00% | 0.00% | 2.39% | 0.00% | 0.00% | 1.75% |
| unusual_punct | 0.50% | 2.06% | 2.15% | 0.32% | 1.60% | 1.70% |
| multi_space | 0.00% | 11.01% | 10.89% | 0.00% | 10.24% | 10.22% |
| lead_trail_space | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |

By country — **train_s2**:

| pattern | India | US | ALL |
|---|---|---|---|
| non_ascii | 27.87% | 6.70% | 15.19% |
| has_digit | 3.10% | 6.40% | 5.08% |
| ocr_digit_in_word | 1.39% | 2.06% | 1.79% |
| ampersand | 3.37% | 4.68% | 4.15% |
| acronym_only | 0.01% | 0.08% | 0.05% |
| domain_name | 3.40% | 4.40% | 4.00% |
| alias_marker | 0.00% | 0.00% | 0.00% |
| unusual_punct | 1.38% | 2.51% | 2.06% |
| multi_space | 10.04% | 11.66% | 11.01% |
| lead_trail_space | 0.00% | 0.00% | 0.00% |

By country — **test_s2**:

| pattern | France | India | US | ALL |
|---|---|---|---|---|
| non_ascii | 24.54% | 27.61% | 6.25% | 18.99% |
| has_digit | 1.11% | 2.88% | 6.18% | 3.89% |
| ocr_digit_in_word | 0.22% | 1.25% | 1.94% | 1.37% |
| ampersand | 5.87% | 3.38% | 4.76% | 4.27% |
| acronym_only | 1.50% | 0.01% | 0.07% | 0.24% |
| domain_name | 3.49% | 2.75% | 3.63% | 3.19% |
| alias_marker | 0.00% | 0.00% | 0.00% | 0.00% |
| unusual_punct | 0.81% | 1.26% | 2.32% | 1.60% |
| multi_space | 9.11% | 9.73% | 11.28% | 10.24% |
| lead_trail_space | 0.00% | 0.00% | 0.00% | 0.00% |

### Unicode scripts in names (share of rows containing ≥1 char of the script; sampled 200k rows/file)

- **train_s1**: none
- **train_s2**: LATIN 5.74%, DEVANAGARI 5.44%, TELUGU 0.77%, KANNADA 0.69%, TAMIL 0.65%, BENGALI 0.63%, GUJARATI 0.60%, ZERO 0.46%, MALAYALAM 0.37%, ORIYA 0.14%, GURMUKHI 0.13%
- **train_s3**: LATIN 6.21%, DEVANAGARI 3.10%, TELUGU 0.42%, KANNADA 0.41%, BENGALI 0.35%, TAMIL 0.35%, GUJARATI 0.33%, ZERO 0.27%, MALAYALAM 0.21%, ORIYA 0.08%, GURMUKHI 0.06%
- **test_s1**: LATIN 2.36%, UNKNOWN 0.00%
- **test_s2**: LATIN 7.79%, DEVANAGARI 6.36%, TELUGU 0.95%, KANNADA 0.91%, TAMIL 0.79%, GUJARATI 0.73%, BENGALI 0.73%, ZERO 0.66%, MALAYALAM 0.46%, ORIYA 0.17%, GURMUKHI 0.15%, UNKNOWN 0.00%
- **test_s3**: LATIN 8.17%, DEVANAGARI 3.54%, TELUGU 0.54%, KANNADA 0.53%, TAMIL 0.45%, BENGALI 0.43%, GUJARATI 0.40%, ZERO 0.35%, MALAYALAM 0.25%, ORIYA 0.10%, GURMUKHI 0.10%, UNKNOWN 0.00%

Most frequent non-ASCII characters in names (sampled):

- **train_s1**: none
- **train_s2**: `्`×28852 `ट`×24530 `ि`×23604 `े`×22659 `र`×20315 `ल`×16762 `ा`×13662 `स`×12849 `प`×11914 `म`×11548 `ड`×11416 `इ`×10216 `్`×9871 `व`×9871 `್`×9130 `்`×7967 `ं`×6935 `क`×6798 `न`×6209 `്`×5678
- **train_s3**: `्`×15812 `ट`×13329 `ि`×13052 `े`×12238 `र`×11207 `ल`×9154 `ा`×7495 `स`×6974 `प`×6540 `म`×6233 `ड`×6192 `इ`×5657 `व`×5357 `್`×5257 `్`×5252 `்`×3938 `ं`×3822 `क`×3654 `न`×3305 `്`×2983
- **test_s1**: `é`×2988 `É`×1315 `è`×1230 `ç`×8 `â`×2 `ê`×2 ``×1 `ô`×1
- **test_s2**: `्`×34904 `ट`×29172 `ि`×27678 `े`×27016 `र`×24466 `ल`×20068 `ा`×16018 `स`×15639 `प`×13855 `म`×13594 `ड`×13552 `్`×12724 `್`×12202 `इ`×12059 `व`×11501 `்`×9851 `ं`×8629 `क`×8458 `न`×7556 `്`×6922
- **test_s3**: `्`×18698 `ट`×15789 `ि`×14657 `े`×14444 `र`×12990 `ल`×10720 `ा`×8595 `स`×8336 `प`×7429 `ड`×7321 `म`×7278 `్`×6917 `್`×6841 `इ`×6525 `व`×6218 `é`×5631 `்`×5361 `ं`×4469 `क`×4385 `न`×3987

ASCII punctuation in names (sampled counts):

- **train_s1**: `.`×27687 `,`×18792 `&`×10183 `)`×4282 `(`×4276 `'`×4256 `-`×1280 `!`×643 `/`×456 `+`×323 `#`×136 `@`×44 `<`×9 `_`×9 `>`×9 `:`×6 `[`×1 `]`×1
- **train_s2**: `.`×41241 `,`×17173 `-`×11418 `(`×10410 `)`×10399 `&`×8373 `[`×5948 `]`×5948 `'`×3982 `*`×1851 `/`×1430 `#`×1365 `>`×1273 `+`×1017 `@`×710 `:`×623 `|`×608 `!`×564 `<`×11 `_`×11
- **train_s3**: `.`×39593 `,`×16439 `-`×12214 `(`×10435 `)`×10433 `&`×8303 `[`×6232 `]`×6231 `/`×4587 `'`×3892 `*`×1833 `#`×1355 `:`×1281 `+`×1259 `>`×1249 `@`×700 `|`×604 `!`×563 `<`×11 `_`×11
- **test_s1**: `.`×19981 `,`×11930 `&`×10177 `)`×7228 `(`×7224 `'`×2728 `-`×1837 `!`×408 `/`×339 `+`×221 `#`×76 `"`×64 `@`×29 `<`×22 `_`×12 `>`×12 `:`×4 `[`×1 `]`×1
- **test_s2**: `.`×37858 `(`×12295 `)`×12286 `,`×10864 `-`×10154 `&`×8457 `[`×5352 `]`×5352 `'`×2501 `/`×1477 `*`×1446 `#`×1077 `>`×998 `+`×919 `@`×514 `|`×511 `:`×489 `!`×368 `"`×76 `<`×26
- **test_s3**: `.`×35189 `(`×12687 `)`×12677 `,`×10805 `-`×10657 `&`×8493 `[`×5719 `]`×5719 `/`×3825 `'`×2502 `*`×1377 `+`×1088 `#`×1052 `>`×1040 `:`×993 `@`×546 `|`×456 `!`×368 `"`×40 `<`×12

### Normalised name tokens

Vocabulary size (distinct normalised tokens) by country:

| file | vocab |
|---|---|
| train_s1 | India: 50,639 | US: 70,685 |
| train_s2 | India: 285,517 | US: 533,594 |
| train_s3 | India: 320,304 | US: 559,299 |
| test_s1 | France: 34,435 | India: 49,738 | US: 56,840 |
| test_s2 | France: 87,957 | India: 288,995 | US: 337,099 |
| test_s3 | France: 94,299 | India: 323,038 | US: 349,609 |

Top tokens by document frequency (share of names containing token) — per country:

- **train_s1 / India**: limited 59.1%, private 48.9%, ltd 16.5%, pvt 13.8%, india 6.9%, llp 4.4%, services 2.7%, solutions 2.4%, brothers 2.3%, trading 2.3%, co 2.2%, technologies 1.9%, international 1.8%, foundation 1.6%, global 1.6%, tech 1.5%, industries 1.5%, enterprises 1.4%, consultants 1.4%, technology 1.4%
- **train_s1 / US**: llc 26.9%, inc 18.0%, and 4.3%, s 4.2%, c 4.1%, l 3.7%, care 3.2%, of 2.8%, associates 2.8%, center 2.4%, group 2.4%, partners 2.2%, p 2.2%, d 2.1%, corp 2.1%, pc 2.0%, health 2.0%, clinic 1.6%, pllc 1.6%, medicine 1.3%
- **train_s2 / India**: limited 42.3%, private 27.0%, ltd 15.3%, praivet 11.7%, pvt 9.4%, india 5.8%, services 3.7%, com 3.4%, llp 3.2%, center 2.9%, co 2.2%, li 2.0%, pra 2.0%, industries 1.9%, enterprises 1.9%, group 1.8%, brothers 1.8%, sri 1.8%, public 1.8%, ventures 1.8%
- **train_s2 / US**: llc 17.6%, inc 13.8%, l 5.1%, center 4.8%, partners 4.5%, com 4.4%, corp 4.1%, and 4.0%, c 3.9%, s 3.8%, group 3.6%, co 3.3%, ltd 3.0%, care 2.6%, of 2.6%, services 2.5%, holdings 2.4%, associates 2.2%, d 1.7%, health 1.7%
- **test_s1 / France**: sarl 28.3%, sas 20.2%, club 8.7%, france 8.0%, de 6.9%, eurl 6.5%, ecole 6.0%, amicale 5.5%, comite 5.3%, sa 4.9%, maison 4.2%, du 4.1%, sasu 4.1%, centre 4.0%, union 3.3%, sci 3.2%, sportive 3.2%, des 3.0%, college 2.7%, amis 2.7%
- **test_s1 / India**: limited 59.1%, private 48.9%, ltd 16.5%, pvt 13.7%, india 6.9%, llp 4.4%, services 2.8%, solutions 2.4%, trading 2.3%, brothers 2.3%, co 2.2%, technologies 1.9%, international 1.8%, foundation 1.6%, global 1.6%, tech 1.5%, industries 1.5%, enterprises 1.4%, consultants 1.4%, technology 1.4%
- **test_s1 / US**: llc 26.9%, inc 18.0%, and 4.3%, s 4.2%, c 4.0%, l 3.7%, care 3.2%, of 2.9%, associates 2.8%, center 2.4%, group 2.4%, partners 2.2%, p 2.2%, corp 2.1%, d 2.1%, pc 2.0%, health 2.0%, clinic 1.7%, pllc 1.5%, medicine 1.4%
- **test_s2 / France**: sarl 20.9%, sas 14.1%, france 10.0%, club 7.4%, s 7.1%, de 6.1%, eurl 5.9%, ecole 5.1%, sa 4.9%, amicale 4.8%, comite 4.7%, sasu 4.4%, a 4.0%, sci 3.9%, groupe 3.8%, du 3.7%, maison 3.6%, com 3.5%, developpement 3.4%, centre 3.4%
- **test_s2 / India**: limited 43.4%, private 26.8%, ltd 15.2%, praivet 11.6%, pvt 9.5%, india 5.9%, llp 3.6%, services 3.6%, com 2.8%, center 2.6%, industries 2.6%, enterprises 2.6%, group 2.5%, public 2.4%, ventures 2.4%, co 2.2%, exports 2.1%, li 2.0%, pra 2.0%, holdings 2.0%
- **test_s2 / US**: llc 17.1%, inc 13.8%, partners 5.6%, corp 5.1%, group 4.7%, l 4.7%, center 4.7%, co 4.3%, ltd 4.1%, and 4.0%, s 3.8%, c 3.7%, com 3.6%, holdings 3.5%, care 2.8%, of 2.6%, services 2.5%, associates 2.2%, d 1.8%, health 1.7%

Last name token (legal-suffix proxy), share of names:

- **train_s1 / India**: limited 59.1%, ltd 16.5%, llp 4.4%, co 1.9%, trust 0.8%, company 0.8%, corp 0.8%, corporation 0.8%, group 0.7%, society 0.5%, clinic 0.5%, associates 0.5%, partners 0.5%, care 0.4%
- **train_s1 / US**: llc 26.9%, inc 18.0%, c 3.4%, corp 2.0%, group 2.0%, pc 1.9%, pllc 1.6%, center 1.4%, associates 1.3%, lp 1.2%, partners 1.1%, clinic 1.0%, care 1.0%, co 0.6%
- **train_s2 / India**: limited 36.8%, ltd 13.4%, private 6.0%, com 3.4%, center 2.5%, llp 2.4%, li 2.0%, services 1.9%, pvt 1.8%, co 1.4%, limitet 1.4%, elelpi 1.3%, company 0.9%, corporation 0.9%
- **train_s2 / US**: llc 13.6%, inc 10.9%, com 4.3%, corp 3.4%, center 3.3%, co 2.8%, ltd 2.6%, c 2.6%, partners 2.5%, group 2.4%, services 2.0%, holdings 1.8%, lp 1.4%, corporation 1.2%
- **train_s3 / India**: limited 34.5%, ltd 14.1%, private 5.9%, com 3.6%, center 3.2%, llp 2.8%, services 2.4%, pvt 1.8%, co 1.4%, li 1.1%, service 1.1%, partners 1.0%, company 0.8%, corporation 0.8%
- **train_s3 / US**: llc 14.2%, inc 11.0%, com 4.2%, center 3.4%, corp 3.1%, partners 2.6%, co 2.6%, c 2.5%, group 2.4%, ltd 2.3%, services 2.1%, holdings 1.7%, lp 1.3%, corporation 1.1%
- **test_s1 / France**: sarl 28.3%, sas 20.2%, eurl 6.5%, sa 4.9%, sasu 4.1%, sci 3.2%, ei 1.6%, club 1.0%, jean 0.7%, france 0.6%, ecole 0.6%, amicale 0.6%, comite 0.5%, sainte 0.5%
- **test_s1 / India**: limited 59.1%, ltd 16.5%, llp 4.4%, co 1.9%, trust 0.8%, corporation 0.8%, corp 0.8%, company 0.8%, group 0.7%, clinic 0.6%, society 0.5%, associates 0.5%, partners 0.5%, care 0.4%
- **test_s1 / US**: llc 26.9%, inc 18.0%, c 3.4%, group 2.0%, corp 2.0%, pc 1.9%, pllc 1.5%, center 1.5%, associates 1.3%, lp 1.3%, partners 1.1%, clinic 1.0%, care 1.0%, co 0.6%
- **test_s2 / France**: sarl 16.8%, sas 11.2%, eurl 4.8%, sa 4.0%, sasu 3.7%, com 3.5%, sci 3.3%, france 2.7%, groupe 1.9%, fils 1.8%, s 1.8%, club 1.7%, developpement 1.7%, l 1.5%
- **test_s2 / India**: limited 38.2%, ltd 13.4%, private 5.7%, llp 2.8%, com 2.7%, center 2.3%, li 2.0%, services 1.8%, pvt 1.8%, elelpi 1.5%, limitet 1.4%, co 1.3%, corporation 0.9%, corp 0.9%
- **test_s2 / US**: llc 13.3%, inc 11.1%, corp 4.1%, com 3.6%, co 3.6%, ltd 3.4%, group 3.1%, partners 3.1%, center 3.0%, holdings 2.6%, c 2.4%, services 1.9%, lp 1.4%, corporation 1.2%
- **test_s3 / France**: sarl 16.7%, sas 11.2%, eurl 4.7%, sa 4.0%, sasu 3.6%, com 3.4%, sci 3.2%, france 2.6%, groupe 1.8%, fils 1.8%, club 1.7%, s 1.7%, developpement 1.6%, l 1.4%
- **test_s3 / India**: limited 36.1%, ltd 14.3%, private 5.8%, llp 3.2%, center 2.9%, com 2.9%, services 2.3%, pvt 1.7%, co 1.3%, li 1.1%, service 1.0%, partners 0.9%, corporation 0.8%, company 0.8%
- **test_s3 / US**: llc 13.9%, inc 11.1%, corp 3.9%, com 3.5%, co 3.3%, partners 3.2%, center 3.2%, ltd 3.1%, group 3.1%, holdings 2.5%, c 2.3%, services 2.0%, lp 1.2%, service 1.0%

## 6. Address statistics

### Length (non-empty addresses)

Characters:

|  | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| train_s1 | 22.0 | 27.0 | 33.0 | 41.0 | 70.0 | 103.0 | 124.0 | 256.0 | 52.1 |
| train_s2 | 21.0 | 25.0 | 31.0 | 37.0 | 63.0 | 97.0 | 118.0 | 249.0 | 47.8 |
| train_s3 | 19.0 | 27.0 | 35.0 | 42.0 | 55.0 | 92.0 | 116.0 | 240.0 | 48.3 |
| test_s1 | 24.0 | 28.0 | 36.0 | 50.0 | 74.0 | 105.0 | 126.0 | 268.0 | 57.2 |
| test_s2 | 21.0 | 25.0 | 32.0 | 43.0 | 68.0 | 99.0 | 120.0 | 269.0 | 51.8 |
| test_s3 | 19.0 | 25.0 | 36.0 | 44.0 | 59.0 | 95.0 | 118.0 | 267.0 | 50.1 |

Whitespace tokens:

|  | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| train_s1 | 4 | 5 | 5 | 7 | 10.0 | 15.0 | 19.0 | 43.0 | 8.03 |
| train_s2 | 4 | 4 | 5 | 6 | 9 | 15.0 | 18.0 | 46.0 | 7.54 |
| train_s3 | 3 | 4 | 5 | 6 | 9 | 14.0 | 18.0 | 43.0 | 7.42 |
| test_s1 | 4 | 5 | 6 | 8 | 11.0 | 16.0 | 19.0 | 43.0 | 8.59 |
| test_s2 | 4 | 5 | 5 | 7 | 10.0 | 15.0 | 19.0 | 43.0 | 8.01 |
| test_s3 | 3 | 4 | 5 | 7 | 9 | 15.0 | 19.0 | 43.0 | 7.72 |

Comma-separated components:

|  | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| train_s1 | 3 | 3 | 3 | 3 | 5 | 8 | 9 | 19.0 | 4.16 |
| train_s2 | 3 | 3 | 3 | 3 | 4 | 7 | 9 | 19.0 | 3.84 |
| train_s3 | 3 | 3 | 3 | 3 | 4 | 7 | 9 | 20.0 | 3.86 |
| test_s1 | 3 | 3 | 3 | 4 | 5 | 8 | 9 | 21.0 | 4.31 |
| test_s2 | 2 | 3 | 3 | 3 | 5 | 7 | 9 | 20.0 | 3.93 |
| test_s3 | 2 | 3 | 3 | 3 | 4 | 7 | 9 | 21.0 | 3.91 |

Median address length (chars) by country:

| file | medians |
|---|---|
| train_s1 | India: 76 | US: 34 |
| train_s2 | India: 68 | US: 32 |
| train_s3 | India: 58 | US: 39 |
| test_s1 | France: 48 | India: 76 | US: 34 |
| test_s2 | France: 40 | India: 68 | US: 32 |
| test_s3 | France: 41 | India: 59 | US: 39 |

### Address patterns by country

**train_s1**

| pattern | India | US | ALL |
|---|---|---|---|
| empty | 0.00% | 0.00% | 0.00% |
| non_ascii | 0.06% | 0.00% | 0.03% |
| has_digit | 91.28% | 100.00% | 96.51% |
| token_5digit | 0.30% | 10.95% | 6.69% |
| token_6digit | 0.02% | 0.13% | 0.08% |
| spaced_pin_ddd_ddd | 0.34% | 0.60% | 0.49% |
| zip_plus4 | 0.00% | 0.00% | 0.00% |
| starts_with_number | 26.63% | 85.97% | 62.22% |
| hash_number | 1.28% | 0.26% | 0.66% |
| zero_padded_number | 2.79% | 0.15% | 1.20% |
| landmark_marker | 13.45% | 0.01% | 5.39% |
| unit_marker | 26.06% | 15.43% | 19.69% |
| po_box | 0.00% | 0.00% | 0.00% |
| null_literal | 0.01% | 0.00% | 0.00% |

**train_s2**

| pattern | India | US | ALL |
|---|---|---|---|
| empty | 2.87% | 3.68% | 3.36% |
| non_ascii | 23.71% | 0.00% | 9.50% |
| has_digit | 91.32% | 90.20% | 90.65% |
| token_5digit | 1.13% | 10.45% | 6.72% |
| token_6digit | 0.02% | 1.39% | 0.84% |
| spaced_pin_ddd_ddd | 1.16% | 0.49% | 0.76% |
| zip_plus4 | 0.00% | 0.00% | 0.00% |
| starts_with_number | 22.87% | 75.23% | 54.24% |
| hash_number | 10.65% | 5.15% | 7.35% |
| zero_padded_number | 6.15% | 5.03% | 5.48% |
| landmark_marker | 11.50% | 0.01% | 4.61% |
| unit_marker | 22.31% | 0.10% | 9.00% |
| po_box | 0.00% | 1.67% | 1.00% |
| null_literal | 2.93% | 3.86% | 3.49% |

**train_s3**

| pattern | India | US | ALL |
|---|---|---|---|
| empty | 3.07% | 3.50% | 3.33% |
| non_ascii | 22.53% | 0.00% | 9.02% |
| has_digit | 90.06% | 91.30% | 90.80% |
| token_5digit | 1.05% | 10.48% | 6.71% |
| token_6digit | 0.02% | 1.33% | 0.81% |
| spaced_pin_ddd_ddd | 1.10% | 0.51% | 0.75% |
| zip_plus4 | 0.00% | 0.00% | 0.00% |
| starts_with_number | 22.25% | 75.68% | 54.29% |
| hash_number | 10.00% | 7.61% | 8.57% |
| zero_padded_number | 5.69% | 4.83% | 5.18% |
| landmark_marker | 8.72% | 0.01% | 3.50% |
| unit_marker | 18.44% | 8.15% | 12.27% |
| po_box | 0.00% | 1.57% | 0.94% |
| null_literal | 2.75% | 3.68% | 3.31% |

**test_s1**

| pattern | France | India | US | ALL |
|---|---|---|---|---|
| empty | 0.00% | 0.00% | 0.00% | 0.00% |
| non_ascii | 28.27% | 0.06% | 0.00% | 4.26% |
| has_digit | 99.58% | 91.27% | 100.00% | 95.85% |
| token_5digit | 0.41% | 0.28% | 10.95% | 4.38% |
| token_6digit | 0.01% | 0.02% | 0.12% | 0.06% |
| spaced_pin_ddd_ddd | 0.00% | 0.35% | 0.60% | 0.39% |
| zip_plus4 | 0.00% | 0.00% | 0.00% | 0.00% |
| starts_with_number | 86.01% | 26.55% | 86.01% | 58.21% |
| hash_number | 0.00% | 1.28% | 0.25% | 0.69% |
| zero_padded_number | 0.07% | 2.84% | 0.15% | 1.40% |
| landmark_marker | 0.00% | 13.43% | 0.01% | 6.28% |
| unit_marker | 0.14% | 25.98% | 15.42% | 18.07% |
| po_box | 0.00% | 0.00% | 0.00% | 0.00% |
| null_literal | 0.00% | 0.01% | 0.00% | 0.00% |

**test_s2**

| pattern | France | India | US | ALL |
|---|---|---|---|---|
| empty | 3.06% | 2.28% | 2.94% | 2.65% |
| non_ascii | 24.05% | 23.85% | 0.00% | 14.75% |
| has_digit | 93.15% | 93.03% | 92.05% | 92.67% |
| token_5digit | 0.51% | 1.10% | 10.68% | 4.68% |
| token_6digit | 0.02% | 0.02% | 1.43% | 0.56% |
| spaced_pin_ddd_ddd | 0.00% | 1.17% | 0.50% | 0.75% |
| zip_plus4 | 0.00% | 0.00% | 0.00% | 0.00% |
| starts_with_number | 69.84% | 23.32% | 76.89% | 50.53% |
| hash_number | 3.32% | 11.14% | 5.25% | 7.76% |
| zero_padded_number | 3.38% | 6.07% | 5.13% | 5.32% |
| landmark_marker | 0.00% | 11.55% | 0.01% | 5.47% |
| unit_marker | 0.15% | 22.36% | 0.10% | 10.64% |
| po_box | 0.00% | 0.00% | 1.69% | 0.65% |
| null_literal | 0.00% | 2.92% | 3.89% | 2.88% |

**test_s3**

| pattern | France | India | US | ALL |
|---|---|---|---|---|
| empty | 2.94% | 2.46% | 2.84% | 2.68% |
| non_ascii | 24.32% | 22.92% | 0.00% | 14.35% |
| has_digit | 93.38% | 91.88% | 92.92% | 92.49% |
| token_5digit | 0.53% | 1.01% | 10.72% | 4.66% |
| token_6digit | 0.02% | 0.02% | 1.37% | 0.54% |
| spaced_pin_ddd_ddd | 0.00% | 1.13% | 0.51% | 0.73% |
| zip_plus4 | 0.00% | 0.00% | 0.00% | 0.00% |
| starts_with_number | 70.48% | 22.63% | 77.19% | 50.40% |
| hash_number | 3.20% | 10.57% | 7.80% | 8.45% |
| zero_padded_number | 3.23% | 5.61% | 4.94% | 5.01% |
| landmark_marker | 0.00% | 8.67% | 0.01% | 4.11% |
| unit_marker | 0.15% | 18.40% | 8.11% | 11.84% |
| po_box | 0.00% | 0.00% | 1.60% | 0.61% |
| null_literal | 0.00% | 2.83% | 3.73% | 2.77% |

### Unicode scripts in addresses (sampled 200k rows/file)

- **train_s1**: UNKNOWN 0.02%, LATIN 0.02%
- **train_s2**: DEVANAGARI 5.44%, KANNADA 0.76%, TAMIL 0.70%, TELUGU 0.69%, BENGALI 0.63%, GUJARATI 0.62%, MALAYALAM 0.29%, GURMUKHI 0.14%, ORIYA 0.12%, LATIN 0.02%, UNKNOWN 0.02%, INVERTED 0.00%, VULGAR 0.00%
- **train_s3**: DEVANAGARI 5.26%, KANNADA 0.79%, TAMIL 0.67%, TELUGU 0.66%, BENGALI 0.61%, GUJARATI 0.59%, MALAYALAM 0.27%, GURMUKHI 0.12%, ORIYA 0.12%, LATIN 0.02%, UNKNOWN 0.02%, INVERTED 0.00%, VULGAR 0.00%
- **test_s1**: LATIN 4.25%, RIGHT 0.13%, UNKNOWN 0.03%, CENT 0.00%, INVERTED 0.00%, VULGAR 0.00%
- **test_s2**: DEVANAGARI 6.49%, LATIN 2.62%, KANNADA 0.89%, TAMIL 0.84%, TELUGU 0.79%, BENGALI 0.75%, DEGREE 0.75%, GUJARATI 0.74%, MALAYALAM 0.39%, MASCULINE 0.24%, GURMUKHI 0.15%, ORIYA 0.13%, RIGHT 0.10%, UNKNOWN 0.02%, INVERTED 0.00%
- **test_s3**: DEVANAGARI 6.26%, LATIN 2.70%, KANNADA 0.89%, TAMIL 0.79%, TELUGU 0.76%, GUJARATI 0.75%, BENGALI 0.69%, DEGREE 0.67%, MALAYALAM 0.33%, MASCULINE 0.21%, ORIYA 0.15%, GURMUKHI 0.14%, RIGHT 0.11%, UNKNOWN 0.02%, INVERTED 0.00%

### Last address component (typically state / region), share of rows

- **train_s1 / India**: `Maharashtra` 18.3%, `Delhi` 11.8%, `Uttar Pradesh` 7.0%, `Karnataka` 6.7%, `Tamil Nadu` 6.0%, `Gujarat` 5.5%, `West Bengal` 5.4%, `Telangana` 5.3%, `Haryana` 3.4%, `Kerala` 3.4%
- **train_s1 / US**: `TX` 8.7%, `NY` 6.7%, `NC` 6.2%, `OH` 5.6%, `IL` 4.9%, `TN` 3.9%, `VA` 3.9%, `MA` 3.7%, `AZ` 3.7%, `IN` 3.2%
- **train_s2 / India**: `Maharashtra` 13.9%, `Delhi` 8.6%, `Uttar Pradesh` 5.3%, `Karnataka` 5.1%, `महाराष्ट्र` 4.7%, `Tamil Nadu` 4.6%, `Gujarat` 4.2%, `West Bengal` 4.2%, `Telangana` 3.3%, `दिल्ली` 2.9%
- **train_s2 / US**: `TX` 8.4%, `NY` 6.5%, `NC` 6.0%, `OH` 5.4%, `IL` 4.8%, `VA` 3.8%, `TN` 3.8%, `` 3.7%, `AZ` 3.6%, `MA` 3.6%
- **train_s3 / India**: `MH` 12.7%, `DL` 8.1%, `UP` 4.9%, `KA` 4.6%, `महाराष्ट्र` 4.2%, `TN` 4.2%, `WB` 3.8%, `GJ` 3.8%, `` 3.1%, `TG` 3.0%
- **train_s3 / US**: `Texas` 8.0%, `New York` 6.1%, `North Carolina` 5.7%, `Ohio` 5.1%, `Illinois` 4.5%, `Virginia` 3.6%, `Tennessee` 3.6%, `` 3.5%, `Massachusetts` 3.4%, `Arizona` 3.4%
- **test_s1 / France**: `Hauts-de-France` 33.9%, `Nouvelle-Aquitaine` 28.4%, `Pays de la Loire` 24.2%, `Bordeaux` 1.1%, `Nantes` 0.9%, `Lille` 0.9%, `Tourcoing` 0.5%, `Dunkerque` 0.4%, `Calais` 0.4%, `Roubaix` 0.4%
- **test_s1 / India**: `Maharashtra` 18.2%, `Delhi` 11.7%, `Uttar Pradesh` 7.0%, `Karnataka` 6.7%, `Tamil Nadu` 6.1%, `Gujarat` 5.5%, `West Bengal` 5.5%, `Telangana` 5.3%, `Haryana` 3.4%, `Kerala` 3.4%
- **test_s1 / US**: `TX` 8.6%, `NY` 6.6%, `NC` 6.2%, `OH` 5.5%, `IL` 4.9%, `TN` 3.9%, `VA` 3.9%, `AZ` 3.7%, `MA` 3.7%, `IN` 3.2%
- **test_s2 / France**: `Hauts-de-France` 10.9%, `Gironde` 9.2%, `Nord` 9.2%, `Nouvelle-Aquitaine` 9.1%, `Loire-Atlantique` 7.8%, `Pays de la Loire` 7.8%, `BORDEAUX` 4.9%, `NANTES` 4.3%, `LILLE` 4.0%, `` 3.1%
- **test_s2 / India**: `Maharashtra` 14.0%, `Delhi` 8.6%, `Uttar Pradesh` 5.4%, `Karnataka` 5.2%, `Tamil Nadu` 4.7%, `महाराष्ट्र` 4.7%, `Gujarat` 4.2%, `West Bengal` 4.2%, `Telangana` 3.3%, `दिल्ली` 2.9%
- **test_s2 / US**: `TX` 8.4%, `NY` 6.5%, `NC` 6.0%, `OH` 5.4%, `IL` 4.8%, `TN` 3.8%, `VA` 3.8%, `AZ` 3.6%, `MA` 3.6%, `IN` 3.1%
- **test_s3 / France**: `Hauts-de-France` 11.7%, `Nouvelle-Aquitaine` 9.8%, `Nord` 9.0%, `Gironde` 8.9%, `Pays de la Loire` 8.5%, `Loire-Atlantique` 7.6%, `Bordeaux` 5.3%, `Nantes` 4.7%, `Lille` 4.2%, `` 2.9%
- **test_s3 / India**: `MH` 12.9%, `DL` 8.1%, `UP` 5.0%, `KA` 4.8%, `महाराष्ट्र` 4.3%, `TN` 4.3%, `GJ` 3.9%, `WB` 3.9%, `TG` 3.0%, `दिल्ली` 2.7%
- **test_s3 / US**: `Texas` 8.1%, `New York` 6.2%, `North Carolina` 5.8%, `Ohio` 5.1%, `Illinois` 4.6%, `Tennessee` 3.7%, `Virginia` 3.6%, `Arizona` 3.5%, `Massachusetts` 3.4%, `Indiana` 3.0%

## 7. Raw examples (random, seeded)

**train_s1**

| entity_id | business_name | business_address |
|---|---|---|
| S1-431204989 | High Agro Private Limited | 40 Second Floor, Gali No-2, Village Kotla, Mayur Vihar Ph-1, Delhi, New Delhi, Delhi |
| S1-510706999 | Stingray Trading Private Limited | Flat No 210 Block No B13 Ssm Nagar Puthur Road Alapakkam, Kanchipuram, Kancheepuram, Tamil Nadu |
| S1-937030132 | Reliable Energy Corp | Main Road, At Narainapur Po & Ps - Ramnagar, West Champaran, Bihar |
| S1-680601617 | Continental Express Purecycle LLC | 428 Thompson Drive, Fairborn, OH |
| S1-588592748 | Pediatric Dental Medicine LLC | 116 Meade Street, Luray Town, VA |
| S1-675073603 | AT Interstate Online Group | Franklin, Unit 305, MA, 5 Independence Way |

**train_s2**

| entity_id | business_name | business_address |
|---|---|---|
| S2-397897602 | Sea Trading | NO.##91 , CORAL MARCHANT, STREET MUTHIALPET, TONDIARPET FORT ST GEORGE, Tamil Nadu |
| S2-539972470 | SHRI PURVI TRUST EXPORTS | GAT NO. 671/7 KUDALWADI, CHIKHALI, PUNE, Maharashtra |
| S2-825460345 | Dream Consultants Private (Limited) | FORBES BUILDING CHARANJIT RAI MARG, FORT, Maharashtra |
| S2-101400770 | Baten Rithm [Partners] | NC, 393 MILLS STREET, COLUMBUS |
| S2-282581939 | Filia Parker Apex Tax Service | 111 JONES DANCY STEET, NORTH WILKESBORO, NC |
| S2-556889266 | Dermatology Modern Hleasth Group | 196 KENT, YOUNGSVILLE, NC |

**train_s3**

| entity_id | business_name | business_address |
|---|---|---|
| S3-155749128 | Sahara Center | Khasra No -581 New Karhera Colony Mohan Nagar, Ghaziabad, उत्तर प्रदेश |
| S3-975658754 | Out Finance Private  Limited | T.s.noh973, D.no. 7-18-8, Waltair Ward, Plot No-39, Flat No. B-3, Vepa Heights, Kirlampudi Lay, Out, Vishakhapatnam, AP |
| S3-543395345 | Mega Power Private | Godrej Coliseum, Mumbai, Mumbai City, MH |
| S3-550897175 | footankle.com | 49 Honeysuckle Terrace, Perinton, New York |
| S3-965318486 | A Cure 9 IT Ltd | 12626-C Chanler Ln, Bowie, Maryland |
| S3-103174358 | The  Urban Yoga! South LLC | 2049 State Avenue, null, Washington, Arizona |

**test_s1**

| entity_id | business_name | business_address |
|---|---|---|
| S1-188425588 | Chasseurs Maternelle SA | 43 bis Rue Armand Dulamon, Bordeaux, Nouvelle-Aquitaine |
| S1-433991437 | OB Compagnie SCI | 32 Rue Charyau, Nantes, Pays de la Loire |
| S1-304698390 | Tourcoing Comite SARL | 18 Rue des Acacias, Tourcoing, Hauts-de-France |
| S1-367833155 | Shiv Projects Pvt Ltd | No. 26, Ground Floor, Railway Station Road Selvaraj Nagar Main Road, Urapakkam, Chengalpattu, Chennai, Tamil Nadu |
| S1-616423496 | Consulting Team Samarth Private Limited | No. 11, I Cross Street, Vinayaga Nagar Vedhanarayanapuram, Venpakkam, Chengalpe, T- 603 011, Chengalpet, Kancheepuram, Tamil Nadu |
| S1-297404632 | Malhotra Solar Private Limited | 22-101/1, 3Rd Fr Above Maa, Furniture B06 R.K.Nr, Tirumalagiri, Hyderabad, Telangana |
| S1-828687732 | Green Retail Company LLC | 912 Elizabeth Court, Cuyahoga Falls, OH |
| S1-156146345 | Beacon Crystal Bancorp LLC | 197 Brent Lane, Clinton, NC |
| S1-998417637 | Electricians Local Union 45 | Richmond City, 4606 Sylvan Road, VA |

**test_s2**

| entity_id | business_name | business_address |
|---|---|---|
| S2-333974492 | Agricole du 14 | Gironde, 38 CHEMIN MAURICE LAGARDERE, BORDEAUX |
| S2-510251812 | MB Societe (SAS) | Gironde, 11 RUE DÉTROIS, BORDEAUX |
| S2-588712339 | Communale Primaire (Frànce) | NO. 154 R. DU CAIRE, ROUBAIX |
| S2-954870617 | PRIVATE SURAT CITY SYSTEMS LÍMITED | SURAT, SURAT CITY, WING E, Gujarat |
| S2-627340214 | NEW CONSULTANTS CORPORATION CORPORATION | S NO 012/1 TO 7/B, TALUKA MULSHI, PUNAWALE, NEAR SAI PUNE CITY, PUNE CITY, महाराष्ट्र |
| S2-666662534 | Kaushik Maanngmte Private Limited | পশ্চিমবঙ্গ, BD-A-114, SECTOR-1, SALT LASE, KOLKATA, KOLKATA |
| S2-63947919 | Hendrix Ínnovative Seafood of 8rookline Ltd | 61 DWIGHT STREET, BROOKLINE, MA |
| S2-255319908 | Supreme Keystone Standard | 3870 1/2 MCLANE PIKE, RED HOUSE, WV |
| S2-931570498 | Hanlon [Pineapple] | 1710 OLIVE ST, EUGENET OWNSHIP, OR |

**test_s3**

| entity_id | business_name | business_address |
|---|---|---|
| S3-382588947 | Marioles & CIE SA | 12 R De Toulouse Lautrec, Bordeaux |
| S3-970321834 | Etudiantes Elementaire Sarl | 11 Avenue De Doëlan, Nantes |
| S3-201484931 | Maison Jyôti SARL | Rue De L’amiruté, Dunkerque, Nord |
| S3-531808477 | Atharv & Partners Co | B-2/13africa Avenue Safadareung Enclave, South Delhi, New Delhi, DL |
| S3-988014108 | Protech Co & | C-178, South Delhi, DL |
| S3-36741668 | EA Marble Private | Offcie No. 414, Thane, महाराष्ट्र |
| S3-570679749 | Wise And Carter Athena | 134 Lowery Lane, Franklin, North Carolina |
| S3-332549882 | 24HR Locksmith Westgate | 00446 Celosia Loop, Kyle, Texas |
| S3-750496484 | Eye Ace-Associates LLC | 10401 Pond Creek Road, Dardanelle, Arkansas |


## 8. Ground-truth statistics

**Integrity:**

| check | value |
|---|---|
| gt_rows | 2,206,821 |
| duplicate_gt_rows | 0 |
| s1_missing_from_gt | 0 |
| gt_not_in_s1 | 0 |
| matched_ids_not_in_sources | 0 |
| matched_ids_bad_prefix | 0 |
| match_ids_linked_to_multiple_s1 | 0 |

### Match cardinality (number of S2+S3 matches per S1 entity)

| matches | S1 entities | share |
|---|---|---|
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | 0.00% |

- Mean matches per S1: **3.461**; among non-singletons: **3.666**
- Mean matches by country: India 3.465, US 3.459
- Total positive links: S2 3,693,619, S3 3,944,746
- S1 entities with ≥2 matches **within the same source** (S2 or S3 duplicates): 76.80%

Cardinality distribution by country (8 = 8+):

| matches | India | US |
|---|---|---|
| 0 | 5.59% | 5.58% |
| 1 | 5.37% | 5.42% |
| 2 | 16.98% | 17.02% |
| 3 | 24.00% | 24.09% |
| 4 | 21.93% | 21.94% |
| 5 | 14.64% | 14.55% |
| 6 | 7.53% | 7.43% |
| 7 | 2.91% | 2.89% |
| 8 | 1.06% | 1.07% |

Matches per S1 from each source (6 = 6+):

| count | S1 with this many S2 matches | … S3 matches |
|---|---|---|
| 0 | 287,745 | 266,276 |
| 1 | 789,108 | 716,417 |
| 2 | 652,779 | 668,375 |
| 3 | 333,957 | 372,443 |
| 4 | 119,078 | 145,116 |
| 5 | 24,154 | 35,378 |
| 6 | 0 | 2,816 |

Source mix of matches:

| mix | S1 entities | share |
|---|---|---|
| S2_and_S3 | 1,776,047 | 80.48% |
| S2_only | 143,029 | 6.48% |
| S3_only | 164,498 | 7.45% |
| none | 123,247 | 5.58% |

### Singleton statistics

- Overall singleton rate: **5.58%**
- By country: India 5.59%, US 5.58%
- By whether the S1 exact name is repeated elsewhere in S1: repeated=False: 5.58%, repeated=True: 5.59%
- By whether the S1 address contains a 5–6 digit token: has_postal=False: 5.58%, has_postal=True: 5.63%

### Linkage from the S2/S3 side

- S2 records linked to some S1: **73.36%** (India 73.37%, US 73.36%)
- S3 records linked to some S1: **74.63%** (India 74.65%, US 74.62%)

### Pair space

- Full cross product S1 × (S2+S3): 22,774,876,013,799
- Same-country cross product: 11,839,670,856,657
- Positive pairs: 7,638,365
- Negatives per positive (same-country all-pairs): **1,550,027 : 1**

## 9. Agreement between true matches (positives) vs random same-country pairs

Sample: 150,000 non-singleton train S1 entities → 549,645 positive pairs, and the same number of random same-country S1–(S2∪S3) pairs as a baseline. Names/addresses use the forensic normaliser (NFKC → anyascii transliteration → lowercase → non-alphanumerics to space). "Rare" token = not among the 300 most frequent normalised name tokens in train.

| signal | true matches | random pairs |
|---|---|---|
| country_equal | 100.00% | 100.00% |
| name_exact_norm | 25.84% | 0.00% |
| name_share_token | 90.44% | 18.86% |
| name_share_rare_token | 70.69% | 0.08% |
| addr_share_token | 95.59% | 21.18% |
| addr_share_number | 82.40% | 2.93% |
| addr_first_number_in_other | 78.98% | 1.09% |
| both_have_postal | 5.35% | 0.79% |
| postal_equal | 5.01% | 0.00% |
| no_rare_name_and_no_number | 4.68% | 96.99% |
| b_name_non_ascii | 13.84% | 13.25% |
| b_addr_empty | 4.39% | 3.32% |

Positives by country:

| signal | India | US |
|---|---|---|
| country_equal | 100.00% | 100.00% |
| name_exact_norm | 18.69% | 30.62% |
| name_share_token | 87.91% | 92.13% |
| name_share_rare_token | 63.97% | 75.18% |
| addr_share_token | 96.08% | 95.27% |
| addr_share_number | 83.74% | 81.50% |
| addr_first_number_in_other | 80.41% | 78.02% |
| both_have_postal | 0.25% | 8.75% |
| postal_equal | 0.24% | 8.19% |
| no_rare_name_and_no_number | 5.24% | 4.30% |
| b_name_non_ascii | 23.40% | 7.45% |
| b_addr_empty | 3.91% | 4.71% |

Positives by source:

| signal | S2 | S3 |
|---|---|---|
| country_equal | 100.00% | 100.00% |
| name_exact_norm | 25.34% | 26.31% |
| name_share_token | 90.23% | 90.63% |
| name_share_rare_token | 69.99% | 71.34% |
| addr_share_token | 95.55% | 95.64% |
| addr_share_number | 81.88% | 82.88% |
| addr_first_number_in_other | 78.60% | 79.34% |
| both_have_postal | 5.28% | 5.41% |
| postal_equal | 4.92% | 5.09% |
| no_rare_name_and_no_number | 4.89% | 4.48% |
| b_name_non_ascii | 15.78% | 12.02% |
| b_addr_empty | 4.45% | 4.34% |

Similarity scores (0–100; Jaccard 0–1):

| score | true median | true p10 | random median | random p90 |
|---|---|---|---|---|
| name_token_set | 100.0 | 71.0 | 34.1 | 52.8 |
| name_ratio | 88.1 | 64.5 | 34.1 | 51.2 |
| name_jaccard | 0.667 | 0.125 | 0 | 0.143 |
| addr_token_set | 94.1 | 74.3 | 36.6 | 44.8 |

Distribution of name token-set similarity among true matches:

| bin | share |
|---|---|
| (-1.0, 20.0] | 0.77% |
| (20.0, 40.0] | 1.55% |
| (40.0, 60.0] | 4.06% |
| (60.0, 80.0] | 9.22% |
| (80.0, 99.99] | 25.25% |
| (99.99, 100.0] | 59.16% |

Distribution of address token-set similarity among true matches:

| bin | share |
|---|---|
| (-1.0, 20.0] | 4.39% |
| (20.0, 40.0] | 0.05% |
| (40.0, 60.0] | 0.92% |
| (60.0, 80.0] | 9.80% |
| (80.0, 99.99] | 61.17% |
| (99.99, 100.0] | 23.66% |

**Exact normalised-name rule** (sampled S1 vs. the entire train S2∪S3, same country): 1,679,292 hits, precision **8.46%**, recall of positives **25.84%**, S1 with ≥1 hit 77.39%.

**Weak-name positives** (name token-set < 40): 2.20% of positives; of these, 87.53% share an address number with the S1 record. Examples:

| S1 name | match name | S1 address | match address |
|---|---|---|---|
| Shree Syringes Private Limited | Novivio+ | 14, South Masi Street, Tenkasi, Tamilnadu, Nellai Kattabomman, Dt, Tamil Nadu | Tamil Nadu, South Masi Street, Tenkasi, Tamilnadu, 14, Nellai Kattabomman, Dt |
| First Israel | Dovazeta | 162 Easy Hollow, West Hamlin, WV | 162 Easy Hollow, West Hamlin, West Virginia |
| Gold Builders Private Limited | Avikelo | 69 S K Bole Roaddagdiwadi Dadar, Mumbai, Mumbai City, Maharashtra | Maharashtra, Mumbai City, Mumbai, 69 S K Bole Roaddagdiwadi Dadar |
| Ambattur Network Private Limited | CIRAKELO | No 2, 2Nd Floor, Velavan Nagar, Paper Mills Road, Above Domino Pizza And Hive, Kolathur, Ambattur, Tiruvallur, Tamil Nadu | NO 2, 2ND FLOOR, VELAVAN NAGAR, PAPER MILLS ROAD, ABOVE DOMINO PIZZA AND HIVE, KOLATHUR, AMBATTUR, TIRUVALLUR, தமிழ்நாடு |
| Systems Panchsheel Motors Private Limited | Dovacalojax | 305, Palladium Business, Hub Opp 4-D Square Mall, Ahmedabad, Gujarat | Ahmedabad, 305, Ahmedabad, Gujarat, Palladium Business, Hub Opp 4-D Square Mall |
| Universal Investments Pvt Ltd | Evotavodova | 377, Bhera Enclave, Paschim Vihar, New Delhi, Delhi | 77, Bhera Enclave, West Delhi, New Delhi, DL |
| Car Global (India) Private Limited | ARCFAYE | 54-C 4, Bihari Baug, 3Rd Bhoiwada, Mumbai, Mumbai City, Maharashtra | MUMBAI CITY, MUMBAI, BIHARI BAUG, 3RD BHOIWADA, महाराष्ट्र, #54-C 4 |
| Indian Foundation | इंडियन फाउंडेशन | Uttar Pradesh, Lucknow, 36 New Hanuman Temple, Lucknow, B.P.P.R.D. | 36 NEW HANUMAN TEMPLE, LUCKNOW, उत्तर प्रदेश |

Most frequent normalised name tokens (treated as generic): `limited`, `private`, `llc`, `inc`, `ltd`, `pvt`, `center`, `l`, `s`, `india`, `and`, `c`, `partners`, `group`, `services`, `corp`, `care`, `co`, `of`, `com`, `associates`, `praivet`, `llp`, `d`, `p`, `holdings`, `health`, `clinic`, `solutions`, `pc`, `global`, `trading`, `industries`, `enterprises`, `corporation`, `ventures`, `brothers`, `lp`, `technologies`, `a`

## 10. Computational / memory assessment

Machine: 16.8 GB RAM, 20 logical CPUs. Profiling peak working set: 7.47 GB; profiling runtime 12.2 min.

| file | TSV MB | raw stream scan s | load s (Parquet if cached) | in-memory GB (Arrow strings) |
|---|---|---|---|---|
| train_s1 | 210.1 | 1.19 | 0.322 | 0.272 |
| train_s2 | 489.3 | 3.95 | 0.455 | 0.63 |
| train_s3 | 503.7 | 3.98 | 0.572 | 0.652 |
| test_s1 | 175.0 | 1.39 | 0.271 | 0.224 |
| test_s2 | 509.5 | 4.42 | 0.565 | 0.646 |
| test_s3 | 506.0 | 4.37 | 0.617 | 0.648 |

- All three train sources together: **1.55 GB** in memory; all test sources: **1.52 GB**.

## 11. Important observations

1. **Files are clean structurally.** 0 malformed rows, 0 duplicate IDs, no BOM/CR, every ground-truth ID resolves. But 349/330 rows in test S2/S3 contain a literal `"` — CSV-style quoting must stay disabled.
2. **Source 1 is a clean canonical form; S2/S3 are noisy derivatives.** Train S1 names: 0 non-ASCII, 0 multi-space, 0 all-uppercase; S2/S3: 15.2%/11.5% non-ASCII names, ~11% multi-space, 18.9% of S2 names ALL-CAPS, 1.8% OCR-style digit-for-letter swaps, 4.0% domain-style names (`xyz.com`), 2.4% of S3 names carry f/k/a / d/b/a markers.
3. **Sources have systematic, source-specific formats.** S2 US addresses end in 2-letter state codes; S3 US addresses use full state names (`Texas`, `New York`) while S3 India uses short codes (`MH`, `DL`, `KA`). State names also appear in native scripts (`महाराष्ट्र`, `दिल्ली`). The same region therefore has ≥3 surface forms — an equivalence map must be *learned from training pairs*, not typed in.
4. **Transliteration is pervasive.** In sampled S2 names, Devanagari appears in 5.4% of rows plus Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya, Gurmukhi. Offline anyascii romanisation gives phonetic spellings that do not equal the English source word (Devanagari `प्राइवेट`→`praivet`, `एलएलपी`→`elelpi`; Tamil `லிமிடெட்`→`limitet`): `praivet` alone is in 11.7% of India S2 names. Token-level romanised↔English equivalences can be mined from train positives.
5. **Match cardinality is high and variable; singletons are rare.** Singletons 5.6%, exactly-one 5.4%, ≥2 matches 89.0%; mean 3.46 matches per S1 (3.67 among non-singletons). 76.8% of S1 entities have ≥2 matches *inside the same source*, i.e. S2 and S3 themselves contain duplicate records of one business.
6. **Each S2/S3 record belongs to at most one S1 entity** (0 violations over 7,638,365 links), and 26.0% of train S2/S3 records belong to *no* S1 entity (distractors). Linkage rate is identical across countries (S2 73.4%, S3 74.6%).
7. **Singletons are indistinguishable from S1-side features.** Singleton rate is 5.6% (India) vs 5.6% (US), 5.6% vs 5.6% for repeated vs unique S1 names. The no-match decision must come from candidate evidence (the absence of a strong candidate), not from properties of the S1 record.
8. **Address numbers are the strongest single signal.** A true match shares an address number with the S1 record in 82.4% of pairs vs 2.9% of random same-country pairs; the S1 *first* number appears in the match in 79.0% vs 1.1%.
9. **Postal/PIN codes are nearly absent.** Only 5.3% of true pairs have a 5–6 digit code on both sides (India 0.2%, US 8.8%). Postal blocking cannot be a primary key.
10. **Names alone are unsafe.** Exact normalised name equality holds for only 25.8% of true pairs, and the exact-name rule has precision 8.5% against the full S2∪S3 universe (1,679,292 hits for 150,000 S1 entities). Generic names repeat heavily (`Primary Care Group` ×253 in train S1, `Bordeaux Club SARL` ×205 in test S1).
11. **Rare name tokens are highly discriminative.** Sharing a non-generic name token: 70.7% of true pairs vs 0.1% of random pairs (any token: 90.4% vs 18.9%). India is harder (64.0%) than US (75.2%) because of transliteration.
12. **Some matches have unrelated names.** 2.2% of true pairs have name token-set similarity < 40 (random invented trade names such as `Dovazeta`, `Avikelo`, or native-script names); 87.5% of those still share an address number. 4.4% of true matches have an empty address, so for those only the name can link.
13. **Simple evidence union bounds blocking.** 95.3% of true pairs share either a rare name token or an address number; the remaining 4.7% need fuzzy / character-level / transliteration-aware retrieval.
14. **Country always agrees on true pairs** (100.0% of 549,645 sampled positives), so blocking within country is lossless on train; country is used as an open-set partition key, never a whitelist.
15. **Train→test shift.** France is 15.0% of test S1 and absent from train. Test has 5.75 S2+S3 records per S1 vs 4.68 in train, so either more matches per entity or more distractors — threshold priors may not transfer. France brings new legal forms (`sarl` 28.3%, `sas` 20.2%, `eurl`, `sasu`, `sci`), acronym-only names in S2/S3 (1.5% of France S2), extremely repetitive names/addresses (`12 RUE Lyderic, Lille, Hauts-de-France` ×99 in S1), accents, `°`/`º` characters (seen only in test S2/S3 addresses), and S2/S3 regions given as départements (`Gironde`, `Nord`, `Loire-Atlantique`) where S1 uses régions (`Nouvelle-Aquitaine`, `Hauts-de-France`).
16. **Memory is not the bottleneck; pair volume is.** Each split's three sources occupy ≈1.6 GB as Arrow strings and reload from Parquet in <1 s. The raw same-country pair space is 1.18e+13 (1,550,027 negatives per positive), so blocking must avoid anything O(N²).

## 12. Implications for blocking

- Partition by `country` value (open set, derived from the data at run time) — lossless on train.
- Core keys to evaluate, driven by observations 8, 11, 13: (a) address house/first number + a street/locality token; (b) rare (IDF-weighted) name tokens after transliteration; (c) character n-gram TF-IDF nearest neighbours on names (typos, OCR swaps, joined words like `Roaddagdiwadi`); (d) TF-IDF nearest neighbours on addresses (invented trade names, empty/garbled names); (e) combined name+address n-gram retrieval.
- Postal-code blocking is low-coverage (observation 9); measure it, but do not rely on it.
- Exact-name blocks explode on generic names (observation 10); cap block sizes / use IDF-weighted scoring and rank-limited retrieval instead of raw equality blocks.
- Target: candidate recall well above the ≈95% simple-evidence bound (observation 13) with an average candidate list that keeps test pairs tractable (test S1 = 1,732,544; e.g. 50 candidates → 87M pairs).
- All blocking statistics (IDF, n-gram vocabularies) are computed per split from the split's own records, so France is handled from test-set statistics without labels.

## 13. Implications for feature engineering

- Keep original and normalised text. Normalise: NFKC → transliterate (anyascii) → lowercase → punctuation/space collapse; strip OCR digit-for-letter swaps inside alphabetic tokens (`8rothers`, `NATI0NAL`).
- Legal-suffix handling: separate the legal-form tokens (`limited/ltd/private/pvt/praivet/llc/inc/sarl/sas/…`) into their own feature instead of deleting them. The suffix inventory is mined from token statistics (top last tokens per country) — no hard-coded country lists.
- Learn token equivalences (romanised↔English, abbreviations like `st`↔`street`, state code↔state name) from aligned tokens in train positive pairs; plus generic prefix/abbreviation matching (`r`↔`rue`, `av`↔`avenue`) that transfers to France without labels.
- Numeric features: first-number match, number-set Jaccard, conflicting numbers, zero-padding-insensitive and `#`-insensitive comparison, unit/flat numbers.
- IDF-weighted name/address token overlap, with IDF computed per country on the split being scored.
- Missingness flags (empty address, `null` literals), name genericness (how many S1/S2/S3 records share the normalised name / address), domain-name and alias-marker flags, native-script flags.
- Candidate-level context: rank of the candidate for this S1, score gap to the best candidate, and how many S1 entities compete for the same S2/S3 record (observation 6).
- Avoid a raw `country` one-hot in the model (France unseen); use country-agnostic features so the model transfers.

## 14. Implications for modelling and decisions

- Per-entity F₀.₅ arithmetic: with 4 true matches, missing one gives 0.938; adding one wrong match to 4 correct gives 0.833; any match on a true singleton gives 0. With a mean of 3.67 matches per non-singleton entity and only 5.6% singletons, recall still matters. The policy should keep every candidate above a precision-oriented threshold, not just the top 1.
- Exploit the one-S1-per-record constraint (observation 6): when an S2/S3 record is claimed by several S1 entities, keep at most the best-scoring one. Test this as a graph/assignment step on validation.
- Within-source duplicates (observation 5) enable consistency features: candidates that are near-duplicates of an already-accepted candidate are likely matches too.
- Validation: split by S1 entity, stratified by country × cardinality bucket. Additionally run a **cross-country transfer check** (train on one country, validate on the other) as a proxy for the unseen France slice.
- Negatives come from the blocked candidate set (hard negatives); random negatives are trivially separable (e.g. random pairs share a rare name token 0.1% of the time).
