# Expanding the search: which databases to add

*Researched 2026-10-09 (live API counts where marked). Constraint that shapes every
choice: the binary index lives in RAM on a 7.8 GB, 4-vCPU server, so each
1M documents costs ~48 MB of RAM and ~0.1 s more brute-force scan per search.
Additions must bring unique (non-PubMed) content with clean licensing and a
practical incremental update path.*

## Short answer

| Rank | Source | Docs | RAM | Why |
|---|---|---|---|---|
| 1 | ClinicalTrials.gov — **added 2026-10-09** (`update_database/clinicaltrials_update.py`, weekly) | 606,601 trials | 28 MB | Unique content, US-gov data (attribution only), daily delta filter |
| 2 | Europe PMC preprints (minus bioRxiv/medRxiv/arXiv) | ~800k | ~27–38 MB | Research Square, Preprints.org, PsyArXiv, Authorea; per-record licence; daily deltas; `HAS_PUBLISHED_VERSION` for dedup |
| 3 | Crossref posted-content: ChemRxiv, TechRxiv, EarthArXiv, SocArXiv, ESS Open Archive | ~190k | ~9 MB | Almost no PubMed overlap; pull by DOI prefix |
| 4 | NIH RePORTER grant abstracts (one per core project) | ~0.6–1.0M (estimate) | ~30–50 MB | Distinct content type ("who is funded to work on X"); best as a separate "Grants" filter |
| 5 | Europe PMC Agricola (conditional) | ~703k | ~34 MB | Agriculture/veterinary coverage; dedupe vs PubMed and confirm reuse terms first |

All five together: ~2.5M documents, ~130 MB RAM, ~+0.25 s per search — affordable
once the backend holds the only index copy (see deploy/).

**Later, deliberately:** a domain-filtered OpenAlex subset (~15–18M docs, ~0.8 GB RAM,
~+1.5 s/search). CC0 dataset, but abstracts only as an inverted index and free updates
are quarterly. Only after moving off brute-force search (sharding or an IVF/HNSW binary index).

**Avoid:** Semantic Scholar (restrictive licence, duplicates PubMed), SSRN (personal-use
terms), Lens.org (negotiated access), CORE (paid dumps, noisy), Europe PMC patents
(frozen), Cochrane (already in PubMed), bulk Crossref journal articles (duplicates,
per-publisher copyright), Zenodo/DOAJ (noise). Use PMC OA and Sciety/eLife reviews as
**links on existing results**, not as new vectors.

## How to add a source in this codebase

1. **Pipeline** (`update_database/`, runs on the GPU desktop): write `<name>_df/*.parquet`
   with at least `title, abstract, date (YYYY-MM-DD), doi, authors, journal` and the
   matching `*.npy` (BGE-small, `"title. abstract"`, **no** query prefix, `np.packbits(emb > 0)`),
   row i of each parquet = row i of its `.npy`. Use `row_group_size=2000`.
2. **Config**: add a `<name>_config` stanza to `config_mss.yaml` (same keys as the others).
3. **Code**: add the name in `config_loader.SOURCE_KEYS`, `search_logic._SOURCE_NAMES`,
   `mcp_server.SOURCES`, `search_api.SourceName`, and the UI's `SOURCE_OPTIONS`,
   `ui_components.SOURCE_NAMES` / `SOURCE_COLORS`; add link rules in `paper_links.py`.
4. **Check** with `eval/check_doc_prefix.py` that the stored vectors match the recipe,
   then restart `mss-backend` — it builds the new chunks on startup.

---

## Full research notes

RAM cost = records × 48 B. Search cost is about +0.1 s per 1M docs. "Live" counts come from API queries run today: Europe PMC REST `hitCount`, the Crossref REST `total-results` with `has-abstract:true`, OpenAlex `meta.count`, ClinicalTrials.gov `/api/v2/stats/size` and the RePORTER `/v2/projects/search` total.

## Source-by-source

| Source | Records with abstracts (live unless noted) | Overlap | Licence / terms | Bulk + incremental | Added RAM |
|---|---|---|---|---|---|
| **Europe PMC preprints (SRC:PPR)** | 1.25M preprints, 1.25M with abstracts. **801k excluding bioRxiv/medRxiv/arXiv**: Research Square 483k, Preprints.org 131k, PsyArXiv 66k, Authorea 49k, SSRN 18k, ChemRxiv 9k (life-science subset only), and others. About 187k added in 2025 | `HAS_PUBLISHED_VERSION:y` holds for 234k of the non-bio/med/arXiv preprints, so the journal version of those is probably already in PubMed | Per-record licence field. 783k carry CC licences ([downloads](https://europepmc.org/downloads/preprints)). Research Square is CC BY 4.0 only ([RS terms](https://www.researchsquare.com/legal/in-review)). Preprints.org is CC BY 4.0 ([report](https://www.preprints.org/blog/post/annual-report-2025)) | FTP `/ftp/preprint_abstracts` (XML), plus the REST API with cursorMark and a `FIRST_PDATE`/`UPDATE_DATE` filter for daily deltas ([downloads](https://europepmc.org/downloads/preprints)). Content is updated daily ([NAR 2024](https://academic.oup.com/nar/article/52/D1/D1668/7442539)) | ~38 MB (all 801k). ~27 MB if published versions are dropped |
| **Crossref `posted-content`** (other preprint servers) | 3.16M with abstracts in total, 363k from 2025. By prefix: ChemRxiv 56.7k, TechRxiv 30.2k, OSF Preprints 94k, PsyArXiv 64.6k, SocArXiv 25.6k, Authorea/ESS Open Archive 68.6k, EarthArXiv 6.5k, Copernicus/EGUsphere 208k, SSRN 730k, AACR 360k (probably meeting abstracts; **unverified**) | Low for ChemRxiv, TechRxiv, EarthArXiv and SocArXiv. Research Square and Preprints.org duplicate Europe PMC | Crossref says abstracts **are copyrighted** (by the publisher, or by the author for OA). Crossref redistributes them under its member terms, and the licence the member attached still applies ([Crossref licence doc](https://crossref.org/documentation/retrieve-metadata/rest-api/rest-api-metadata-license-information)). ChemRxiv uses author-chosen CC BY / BY-NC / BY-NC-ND ([FAQ](https://chemrxiv.org/engage/assets/public/chemrxiv/term/faq.htm)) | Annual public data file (~180M records, 208 GB JSONL, torrent or AWS) ([2026 PDF](https://crossref.org/blog/2026-public-data-file-now-available/)). Deltas via REST `from-index-date` + `prefix`. Polite pool is about 10 req/s, with list queries lower ([rate limits](https://crossref.org/blog/announcing-changes-to-rest-api-rate-limits/)) | ChemRxiv + TechRxiv + EarthArXiv + SocArXiv + ESS ≈ 190k ≈ **9 MB** |
| **ClinicalTrials.gov** | **606,387 studies**, each with a brief summary | Essentially none. Trials are not PubMed abstracts | US government data, free to all requesters. Attribute the source and give the processing date ([terms, classic](https://classic.clinicaltrials.gov/ct2/about-site/terms-conditions)). The current terms page did not render, so this is **unverified for 2026** | API v2 (`/api/v2/studies`, pageToken) or a full JSON zip download. Daily updates; filter by `LastUpdatePostDate`. About 50 req/min reported, **third-party figure** ([ref](https://cdn.jsdelivr.net/npm/@saibolla/ada@0.1.3/skills/clinicaltrials-database/references/api_reference.md)) | ~29 MB |
| **NIH RePORTER / ExPORTER** | 2.98M application records, about 76k per fiscal year. **Unique projects** after deduplicating renewals on `core_project_num` are an estimated ~0.6–1.0M (**my estimate, unverified**) | None with papers. Grant abstracts are a distinct content type | US government (NIH) data. No restrictive licence stated ([API page](https://api.reporter.nih.gov/)) | ExPORTER CSV files, weekly and yearly ([data.gov](https://catalog.data.gov/dataset/nih-research-portfolio-online-reporting-tools-expenditures-and-results-reporter)). API is ≤1 req/s, at most 500 records per call, offset ≤14,999, so delta queries have to be sliced by date ([api](https://api.reporter.nih.gov/)) | ~30–50 MB |
| **Europe PMC Agricola (SRC:AGR)** | 1.03M records, **703k with abstracts**. 67k added in 2025, 16k so far in 2026 | Partial overlap with MEDLINE journals (e.g. J Colloid Interface Sci); 0 have a PMID flag. Dedupe by DOI/title | **Unclear.** Abstracts are publisher text, and I found no explicit Europe PMC reuse statement | Same Europe PMC REST/FTP route as the preprints | ~34 MB |
| **OpenAlex** | 202M works. Live filter for English article/preprint/review, ≥2000, no PMID, not arXiv: Life 4.9M, Health 7.4M, Physical 19.3M, Social 15.8M. By field: Agri/Bio 2.7M, Env 2.4M, Chemistry 1.1M, Materials 1.6M, Engineering 7.4M | `has_pmid:false` is probably undercounted, so some of these are still PubMed items | Dataset is **CC0** (bucket LICENSE.txt). Abstracts are served **only as `abstract_inverted_index`** "due to legal constraints" ([pyalex](https://pypi.org/project/pyalex)). OpenAlex removed Springer Nature abstracts over copyright ([thread](https://groups.google.com/g/openalex-users/c/ptFDD7qWvYw/m/kXWDG3o5BAAJ)) | Free S3 snapshot (~745 GB JSONL) is **quarterly**, partitioned by `updated_date`, with `deleted_ids` ([snapshot](https://help.openalex.org/access/snapshot/)). Daily snapshots and changefiles are paid. The API needs a key: a free $1/day covers ~10k list calls ([auth](https://developers.openalex.org/api-reference/authentication.md)) | Life+Health+Agri+Env+Chem ≈ 15–18M → **0.7–0.9 GB** and +1.5 s/search |
| **Semantic Scholar** | Abstracts dataset of 100M records, release 2026-09-29 ([release](https://api.semanticscholar.org/datasets/v1/release/latest)) | Very high overlap with PubMed and arXiv | API licence: non-sublicensable; must display the S2 name and logo; commercial use needs a separate licence; terms can change unilaterally ([licence](https://api.semanticscholar.org/license)) | Datasets API with a key; monthly releases with diffs | n/a (avoid) |
| **CORE** | 85.6M with abstracts (2018 figure; **no 2025 count published**) | High; also noisy (repository duplicates) | Old dumps are ODC-By, newer dumps need a paid licence or membership ([dataset](https://core.ac.uk/services/dataset)) | Registration-only dump plus API | n/a |
| **PMC OA** | Records that are PMC-only (no PMID): 994k, of which **110k have an abstract** | ~All other PMC content is in PubMed | Per-article CC licences | — | Use it for **full-text/PDF links**, not new vectors |
| **SSRN** | 730k abstracts in Crossref | Low | Elsevier ToU: "solely for personal, non-commercial use… may not… repost" ([Elsevier](https://www.elsevier.support/ssrn/answer/can-i-repurpose-content-available-on-ssrn)) | — | Avoid |
| **DOAJ** | Article dumps (count not published) | Mostly duplicates Crossref and OpenAlex | Article metadata **CC0** ([terms](https://www.doaj.org/terms/)) | Dumps granted case by case; monthly ([dump](https://doaj.org/public-data-dump)) | Low marginal value |
| **AGRIS (FAO)** | >13M records in the open data set; abstract share unknown | Some overlap with Agricola and Crossref; multilingual | CC BY 3.0 IGO ([FAO](https://www.fao.org/agris/download)) | Zip download (AP/RDF); update cadence not documented | Up to ~0.6 GB; needs filtering |
| **Patents** | Europe PMC PAT has 2.65M abstracts but is **effectively frozen** (≤11 new records/yr since 2021). PatentsView (USPTO grants since 1976) is CC BY 4.0 ([PatentsView](https://patentsview.org/download/data-download-tables)) | None | — | — | Millions of docs; off-mission |
| **Zenodo** | Very mixed record types (datasets, posters, software) | — | Metadata CC0 ([ref](https://indico.in2p3.fr/event/24341/contributions/95714/attachments/64368/89268/20210526_OSSR_harvest_and_retrive.pdf)) | OAI-PMH | Low signal-to-noise |
| **Lens.org** | 225M+ works | Aggregates the sources above | Bulk access only via negotiated institutional plans ([guide](https://researchguides.library.syr.edu/lensguide/institutions)) | — | Avoid |
| **Cochrane** | CDSR reviews | **Already indexed in PubMed/MEDLINE** (general knowledge, not re-verified) | Wiley copyright | — | Skip |
| **eLife / Review Commons / Sciety** | Peer-review evaluations, not abstracts ([Sciety](https://crossref.org/blog/evolving-the-preprint-evaluation-world-with-sciety/)) | Attach to existing bioRxiv/medRxiv records | Mostly CC BY | DocMaps / APIs | 0 MB: show as badges/links instead |
| **EThOS (ETH), Chinese Biol. Abstracts (CBA)** | 100k and 141k | — | — | Both static (CBA had 0 records in 2025; EThOS has been offline since 2023) | Skip |

## Ranked recommendation (unique value per MB)

1. **ClinicalTrials.gov (~29 MB, +0.06 s).** This is the cleanest licence of anything here (US government data, attribution only). The content does not appear in any current index, there is a daily delta filter, and each record links to its study page. It is a strong fit for a biomedical user base.
2. **Europe PMC preprints, excluding servers already indexed (~27–38 MB).** This adds Research Square, Preprints.org, PsyArXiv, Authorea and others, about 800k records, through one API you probably already use. It gives per-record licence fields, daily deltas and `HAS_PUBLISHED_VERSION` links for deduplication against PubMed. Licensing is mostly CC BY. Drop or flag non-CC records, and exclude the SSRN subset.
3. **Crossref posted-content for non-life-science servers (~9 MB).** These are ChemRxiv, TechRxiv, EarthArXiv, SocArXiv/OSF and the ESS Open Archive. They are tiny and almost entirely non-PubMed, and you pull them by prefix with `from-index-date`. Respect author-selected CC licences; store all of them, and for NC/ND records show the abstract with attribution and a link.
4. **NIH RePORTER grant abstracts (~30–50 MB once deduplicated by core project).** This is public data and a distinctive content type ("who is funded to work on X"). Updates are weekly (CSV or API). It is best as a separate, filterable "Grants" source.
5. **(Conditional) Europe PMC Agricola (~34 MB).** It adds agriculture and veterinary coverage that PubMed lacks. Deduplicate it against PubMed by DOI/title first, and confirm abstract reuse terms with Europe PMC.

**Larger, deliberate step:** a domain-filtered OpenAlex subset (life, health, agri, environment, chemistry; ~15–18M records; ~0.8 GB RAM and +1.5 s per query). The dataset is CC0, but the abstracts are publisher text that is only available as an inverted index, and free updates are quarterly unless you pay. Do this only after sharding or moving to an IVF/HNSW binary index.

**Avoid:**
- Semantic Scholar: restrictive licence, logo/attribution duties, and it duplicates PubMed.
- SSRN: Elsevier's personal-use terms.
- Lens: negotiated licence.
- CORE: paid current dumps and noisy data.
- Europe PMC patents: stale.
- Cochrane: already in PubMed.
- Broad Crossref journal-articles: ~39M abstracts, mostly duplicates, with per-publisher copyright.
- Zenodo, DOAJ: noise or duplication.
- PMC OA as new vectors: use it for full-text links instead.

Preprint reviews (Sciety/eLife) should be links, not vectors.

**Uncertainties:**
- The ClinicalTrials.gov and RePORTER rate limits, and the RePORTER unique-project count, are not fully verified.
- I found no explicit Europe PMC statement on reusing abstracts.
- Of the Crossref posted-content, the AACR and Copernicus records are probably mostly conference abstracts.
- The counts behind OpenAlex `has_pmid:false` are likely inflated by missing PMID links.
