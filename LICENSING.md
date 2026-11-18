# Licensing and release boundaries

## Original data: CC BY 4.0

The author licenses the original datasets and data documentation under
**Creative Commons Attribution 4.0 International**. See
[`data/LICENSE.md`](data/LICENSE.md) for scope and attribution, and
[`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt) for the full terms.
This covers original probes, the AI-assisted 120-question panel/references/rubric,
original factual-source summaries, numerical masks and original recipe metadata.
It does not grant exclusive copyright or relicense cited material.

## Software: MIT

Original project software and software documentation are licensed under the
**MIT License**, provided in [`LICENSE`](LICENSE). This includes `src/`,
`scripts/` and `data/scripts/`. The MIT grant does not relicense
datasets, model assets, or third-party material. Original datasets and data
documentation retain their separate CC BY 4.0 terms above.

## No downloaded datasets are redistributed

The published `data/` contains only original project data and numerical
provenance/selection metadata. Downloaded content is excluded even when it has a
permissive license, and even after filtering, normalization, paraphrasing,
translation or tokenization. Readers acquire sources under their own applicable
permissions and generate derivatives in ignored workspaces.

| External material | Terms and conditions still relevant to readers |
|---|---|
| Gretel SQL | Apache-2.0; preserve attribution, full license and transformation notices. |
| Spider questions, SQL and schema adaptations | CC BY-SA 4.0; attribution, change notices, share-alike and applicable database rights. |
| Spider SQLite databases | Originating notices and third-party rights incompletely traced. Not bundled; use official acquisition and retain notices. |
| Guided discovery text, variants and reversible plans | Mixed-source rights; unresolved ARC conflict, MMLU origin rights, WIQA permission and passage attribution. Not bundled. |

Full upstream Apache and CC BY-SA legal texts remain in `LICENSES/` for reference.
They do not license independent code or override the original-data license.
The numerical recipes and external expected hashes do not contain benchmark
questions, reference passages, SQL schemas or tokenized copies.

## Discovery-specific cautions

Natural Instructions' blanket Apache grant excludes task instances. The original
WinoGrande data grant is CC BY 2.0, unlike a historical inherited Apache label.
ARC's publisher archive specifies noncommercial research/no redistribution,
conflicting with a later mirror license field. MMLU examination-source rights,
WIQA originating data permission, and passage-level attribution remain unresolved.
See [`data/DISCOVERY.md`](data/DISCOVERY.md).

These are conservative redistribution findings, not a declaration that all
historical research use was prohibited. A model's permissive generation license
does not erase source rights. Token IDs are not a way around source restrictions.

## Models and data distribution

No model weights, adapters, tokenizer, optimizer state, runtime binaries or
historical predictions/judgments are included. Readers acquire the pinned
Gemma 3 4B PT base under its custom Gemma terms, not Gemma 4's Apache grant.
Later distribution of weights or adapters needs separate derivative-model review.

The package follows the original-only policy and CC BY 4.0 grant above.
[`data/manifest.json`](data/manifest.json) is the shipped-data inventory;
[`data/external_manifest.json`](data/external_manifest.json) lists unshipped
expected outputs. Source-specific qualifications remain documented in
[`data/DISCOVERY.md`](data/DISCOVERY.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

Before public release, review the final tree for credentials, personal data and
accidentally staged downloaded content.
If excluded content was previously committed, deleting it from the current tree
does not remove it from Git history. Check the history before publishing it.
This is not legal advice or a warranty of title, privacy clearance or noninfringement.
