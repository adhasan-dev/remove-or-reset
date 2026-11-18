# Third-party notices and transformations

These notices document reader-acquired sources and historical transformations,
not bundled third-party dataset content or a blanket software license.
The current policy excludes all downloaded datasets and their processed
derivatives. Original data are CC BY 4.0. See [LICENSING.md](LICENSING.md).

## Gretel synthetic_text_to_sql

Source: [Gretel's dataset](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql),
revision `740ab236e64503fba51be1101df7a1be83bf455d`, released in April 2024
under Apache-2.0.

Publisher credits: Yev Meyer, Marjan Emadi, Dhruv Nathawani, Lipika Ramaswamy,
Kendrick Boyd, Maarten Van Segbroeck, Matthew Grossman, Piotr Mlocek, and
Drew Newberry. Preserve the upstream attribution and any applicable notices.

License: [full Apache-2.0 text](LICENSES/Apache-2.0.txt).
The pinned publisher card expressly identifies the dataset as Apache-2.0:
[pinned card](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql/blob/740ab236e64503fba51be1101df7a1be83bf455d/README.md).

Changes for this study include filtering, read-only SQL validation, possible
transpilation to SQLite, canonical schema serialization, duplicate removal,
schema-grouped splitting, and fixed prompt/target formatting. Source-derived
IDs and original-split metadata remain in the prepared records. Compressed
reader-generated files preserve the frozen prepared payload bytes, with compressed
and uncompressed hashes recorded separately.

## Spider 1.0

Source: [Yale LILY Spider](https://yale-lily.github.io/spider).
Pinned question/query mirror:
[`xlangai/spider`](https://huggingface.co/datasets/xlangai/spider),
revision `0c350918f3f29ec754f1181c65cdce76cd6c133c`.

Credit:

Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li,
James Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and
Dragomir Radev. 2018. *Spider: A Large-Scale Human-Labeled Dataset for
Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*.
EMNLP. [DOI: 10.18653/v1/D18-1425](https://doi.org/10.18653/v1/D18-1425).

License: [full CC-BY-SA-4.0 text](LICENSES/CC-BY-SA-4.0.txt).
See the [pinned dataset card](https://huggingface.co/datasets/xlangai/spider/blob/0c350918f3f29ec754f1181c65cdce76cd6c133c/README.md)
and the official publisher's archive notice.

Reader-generated records/adaptations retain CC-BY-SA-4.0. Changes include filtering,
canonical schema serialization, grouping/splitting, panel selection, and
prompt/target formatting. These are not endorsed by the Spider authors.
Share-alike applies to adaptations and, where applicable, database rights
in a resulting compilation; it does not license unrelated software.

The SQLite database payloads are not redistributed. Their IDs and expected
hashes identify reader-acquired official fixtures. Retain the complete
original archive's source notices when acquiring them. Its verified archive
SHA-256 is
`00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b`.
The separately licensed official Spider evaluator is not bundled; this
study uses its own explicitly described execution-agreement evaluator.

## Original synthetic and held-out inputs

The relation graphs, authored auxiliary probes, numerical masks and
120-prompt held-out panel are research inputs rather than trained models.
The held-out panel has **AI-assisted original authoring; it is not claimed
to be human-authored**. It retains factual source references and declared
derivations, without republishing full source pages. First-party licensing
is CC BY 4.0; see [the original-data license](data/LICENSE.md).

GPT-6 Astra assisted design, GPT-6.1 Sol assisted implementation/execution,
and GPT-6 Luna was used for the later semantic judging. This release contains
the evaluation inputs and rubric, not historical answers or judge decisions.
Multiple Luna agent instances are not independent judge-model families.

## Other historical inputs not shipped

The source-restricted guided corpus and its generation are described in
[the discovery documentation](data/DISCOVERY.md). Their mention does
not grant rights to the underlying data or imply that model weights are
bundled. In particular, Natural Instructions' Apache license excludes its
task instances; the originating dataset's terms remain applicable.

There is no implication of endorsement by Google, Gretel, Yale LILY,
Allen Institute for AI, dataset authors, or model providers.
