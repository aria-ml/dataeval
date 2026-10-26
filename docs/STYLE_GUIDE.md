# Documentation style guide

This guide records how DataEval documentation is written. Follow it when you add or change a notebook, a concept page,
a getting-started page or a docstring.

Each rule says how it is checked. A rule with a named test or session fails the build when you break it. A rule marked
"review" has no automated check, so a reviewer holds you to it. The last sections list the commands to run before you
open a merge request.

## Choose the kind of page

DataEval follows the [Diataxis](https://diataxis.fr) framework. Every page has one purpose.

| Kind | Purpose | Lives in | Name |
| ---- | ------- | -------- | ---- |
| Tutorial | Teaches by working end to end on a realistic dataset. The reader learns by doing. | `docs/source/notebooks` | `tt_<topic>.py` |
| How-to guide | Solves one task for a reader who already knows what they want. A recipe. | `docs/source/notebooks` | `h2_<task>.py` |
| Concept page | Explains an idea, the reasoning behind it and its limits. | `docs/source/concepts` | `CamelCase.md` |
| Getting started | Orients a new reader: installation, quickstart, which tool, what data each tool needs. | `docs/source/getting-started` | `kebab-case.md` |
| API reference | Generated from docstrings. | `src/dataeval` | not applicable |

A tutorial is heavy and a how-to is light. A tutorial takes the reader through a whole workflow in ten to thirty
minutes and ends by summarizing what they learned. A how-to is the shortest path to one result.

Rules (review):

- Teach through links. A how-to does not explain why something works. It links to the tutorial that teaches the topic
  and the concept page that explains it. A tutorial links to a concept page for depth and does not repeat it.
- Write a tutorial only when the notebook works end to end on a dataset and teaches a concept. Everything else is a
  how-to.
- Give a how-to one task. If it needs two recipes, write two how-tos.
- Link to the page that owns a recipe instead of teaching it again.
- Call a page by its kind. A how-to says "this guide" and not "this tutorial".
- Name the class or function your code calls, with a role (for example `` {class}`.Coverage` ``), and not "the
  coverage tool".

## Write the prose

Narrative text is written to the reader in the second person and kept plain. This applies to notebooks, concept pages
and getting-started pages. It does not loosen the docstring rules in the docstrings section. Pages written before this
guide may not follow it yet. Bring a page into line when you edit its prose.

### Voice

- Use the directive voice: "You should check ...", "You will create ...", "You can drop ...". Do not write "Note that
  one can ...", and do not describe what the library does when the sentence tells the reader to do something.
- Write short declarative sentences. Two plain sentences are better than one sentence with a colon, a dash and a
  dependent clause.
- Write contractions out: "do not", "it is", "you will". A possessive `'s` is not a contraction and stays.
- Use American spelling. `codespell` is configured to convert British spellings.
- Say "dataset" and never "corpus". DataEval is a computer-vision library, and "corpus" reads as text terminology. In
  a core function that takes arrays, name what the function receives ("the input", "the codes").

### What to avoid

| Avoid | Write instead |
| ----- | ------------- |
| A metaphor or figurative framing: "retire the sentinels", "a factor sat out the analysis" | The plain verb: "replace the sentinels", "one factor is still dropped" |
| An em-dash aside or a trailing flourish | Two plain sentences, or cut it |
| Rhetorical stress: italics for emphasis, bold stress, "exactly", "simply", "genuinely", "quietly", "on your behalf" | Delete the word or the markup |
| A not-X-but-Y contrast, a one-line closer, a forced triad, filler or sales language | Say the point once. Keep a contrast that carries a real qualification |
| A figurative heading, such as a numbered "lens" | A heading that names what the section does |
| Commented-out code or links left in a notebook | Remove them. Keep a commented alternative only when the text explains it |

The only em dash allowed is the separator in a Next steps entry (see the Next steps section). Review checks all of the
above.

### Emphasis

Do not use italics for emphasis. Use bold only at the first mention of a term that the page defines. Do not use bold
for a list-item label, a heading or stress. Use `*` for emphasis markup and never `_`, because a dunder such as
`__init__` is itself valid strong emphasis (`.markdownlint-cli2.yaml` enforces this).

These rules cover the cells of a table too. Do not bold a row label, and do not put an em dash in a cell. Write a column
heading in sentence case, with a capital on the first word only.

### Claims, numbers and limits

- State every limit, assumption and caveat the reader needs, in plain words. Do not soften or remove a hedge when you
  edit for style.
- Quote a number in prose only after you have checked it against the executed output. Protect it with an assertion
  cell (see the code cells section).
- Do not state a cause you did not trace. Write what you measured.
- Do not describe a library behavior in a docstring or page without running it. A docstring once said that batch size
  never changes embeddings, and a test showed that it changes their last digits.
- Do not describe future work as if it were a feature ("to follow", "planned").

### Line length and Markdown

- Wrap Markdown at 120 columns. Tables are exempt (`.markdownlint-cli2.yaml`).
- Wrap the Markdown cells of a notebook at 120 columns. No tool checks this, because `ruff` ignores line length in
  `docs/source/notebooks`, so review checks it.
- Run `uv run nox -s lint` before you push. It runs `ruff`, `codespell` and `typos`.

## Structure a tutorial

A tutorial's sections come in this order. The headings are fixed strings. A how-to has its own structure, in the next
section.

| Section | Level | Required | Content |
| ------- | ----- | -------- | ------- |
| Title | H1 | yes | An imperative verb phrase, for example `Clean a dataset` |
| Opening | | yes | One or two sentences, then the three metadata lines below |
| `What you will do` | H2 | yes | The steps, as a list |
| `What you will learn` | H2 | yes | The outcomes, as a list |
| `What you will need` | H2 | yes | The prerequisites |
| `Introduction` | H2 | yes | Why the topic matters |
| `Setup` | H2 | yes | Imports and configuration |
| Working sections | H2 | yes | One per step, named for what the reader does |
| `Conclusion` | H2 | yes | What the reader did |
| `What you learned` | H2 | no | The takeaways |
| `Next steps` | H2 | yes | See the Next steps section |
| `On your own` | H2 | yes | How to repeat the analysis on the reader's own data |

The three metadata lines follow the opening, each on one line and in this form:

```text
Estimated time to complete: 15 minutes

Relevant ML stages: [Data Engineering](../getting-started/roles/ML_Lifecycle.md#data-engineering)

Relevant personas: [Data Engineer](../getting-started/roles/data_engineer.md)
```

List one or more personas, separated by commas. Base the estimate on how long the notebook takes to execute in the
docs build and on tutorials of similar length. `tests/test_notebook_structure.py` checks the opening, the metadata
lines and the order of the sections.

## Structure a how-to

| Section | Level | Required | Content |
| ------- | ----- | -------- | ------- |
| Title | H1 | yes | `How to <verb> ...` |
| Lead | | yes | No heading. At most 120 words in up to three short blocks: the problem and the tool that solves it, with a link to its concept page; when to use it, as a sentence or a short list; and any requirement beyond `pip install dataeval` |
| Working sections | H2 | yes | One per step, named for the action, for example `Create embeddings with PyTorch` |
| `Next steps` | H2 | yes | See the Next steps section |

The hidden Colab cell and the imports cell follow the lead, with no heading above them. The lead replaces the
`Problem statement`, `When to use`, `What you will need` and `Getting started` headings, so a how-to has none of them,
and none named `Introduction`, `Setup`, `Summary`, `What you learned`, `Key takeaways` or `Conclusion`. It ends with its
Next steps. If a closing paragraph holds something the reader needs, move it into the section where the reader needs it.

Keep a how-to short (review):

- Aim for 600 words or fewer of prose outside the Next steps list. Count the words of the markdown cells, without the
  headings. The four models in this guide, `h2_cluster_analysis`, `h2_deduplicate`, `h2_measure_divergence` and
  `h2_measure_label_independence`, run from 322 to 501 words.
- Stay on the task. Leave out topics the task does not need, and keep tests out of the visible page. A check that the
  result is right, such as a comparison of two paths or a proof that a number matches, goes in a hidden assertion cell
  and not in a section.
- Write a step as a short lead-in, the code, and at most two sentences on what to read in the output. The lead-in is at
  most two sentences, or one short list when the step implements a contract. Use five steps or fewer. A how-to that
  needs more is two how-tos.
- Treat more than 80 lines of code as a prompt to check the scaffolding: a dataset, a model or a helper that the recipe
  does not need. Hide it, as the code cell rules describe, or cut it. Do not cut the recipe's own cells.
- Cut an explanation of why something works and link the tutorial or concept page that owns it. Cut a sentence that
  repeats what a code line says or what an output just showed. Keep every limit, assumption and caveat.
- A how-to that needs more than the budget because it carries two tasks or several alternative paths is two how-tos. A
  how-to that has to teach a concept to be useful is a tutorial.

`tests/test_notebook_structure.py` fails on a lead over 120 words, on more than five steps and on a boilerplate or
teaching heading, and warns on more than 600 words of prose. Review checks the rest.

## Write headings

These rules apply to every heading in a notebook. Apply them to the headings of other pages as well.

| Rule | Statement | Example |
| ---- | --------- | ------- |
| Sentence case | Capitalize the first word and proper names only | `Marginal class probability shift`, not `Marginal Class Probability Shift` |
| Titles | A how-to is `How to <verb> ...`. A tutorial is an imperative verb phrase. No `Introduction to`, no gerund, no colon subtitle | `Clean a dataset`, not `Introduction to data cleaning` |
| No numbering | No `Step 3:` and no `3.`. Position carries the order, and numbers go stale when steps move | `` Choose your level with `at()` ``, not `` Step 3: Choose your level with `at()` `` |
| Plain text | No bold, no em dash and no trailing colon. Use backticks only around an identifier | `Quick decision table`, not `**Quick decision table:**` |
| One H1, no skips | A notebook has one H1, and heading levels never skip | |
| Fixed template headings | Use the strings in the two structure tables | `What you will do`, not `What you'll do` |
| No teaching closer in a how-to | See the how-to structure | |
| Required sections present | Every required section in the structure tables exists | |

When you refer to a section in prose, name it: `see the "Measure your own factors" section`. Do not refer to it by
number.

`tests/test_notebook_structure.py` checks the case, numbering, plain-text, title and template-heading rules. Review
checks that each heading names what the reader does.

## Write the Next steps section

Every notebook ends with `Next steps`. Each entry has this form, and the em dash is the only one allowed in
documentation prose:

```text
- [Embeddings](../concepts/Embeddings.md) — Learn what embeddings represent and how they are used across DataEval.
```

`tests/test_notebook_structure.py` checks the entry form, the first two rules below and the link text. Review checks the
rest.

Rules:

- List two to four entries in a tutorial and two or three in a how-to.
- Include at most two concept pages and at most two notebooks.
- Choose the concept page of the evaluator or topic the notebook is about. Add a second only when the notebook relies
  on a second distinct idea. Leave out general pages, such as Acting on Results or Data Integrity, unless they are the
  topic.
- For a how-to, choose the tutorial that teaches the topic first, then the natural next task. For a tutorial, choose
  the how-to that applies its main step in depth, or the next tutorial in the learning path.
- Leave out a link the body already makes for a side detail, such as an optional setting or an alternative way to do
  the step. Keep the topic's concept page and the natural next task even when the body links them. Leave out a link to
  a notebook about an unrelated evaluator and a link to a notebook that is going to be removed.
- Use the exact title of the target page as the link text, and a one-sentence description. When the link has an
  anchor, use the heading that the anchor names, as the page shows it. Do not repeat the link text in the description.
  A tutorial and a how-to use the same form.

## Write the code cells

### Format

- Write the notebook as a `py:percent` script in `docs/source/notebooks`. The `.py` file is the committed source. The
  `.ipynb` file is generated and ignored by git.
- Copy the header from a neighboring notebook, including its `jupytext_version`.
- Do not run `nox -s docsync`. It rewrites every notebook, including ones you did not touch. `uv run nox -s docs`
  converts the scripts for you.
- Import from the public modules shown in the API reference. Never import from a module whose name starts with an
  underscore.
- Keep notebooks self-contained so they run on Colab: no helper imported from the repository. Use `maite-datasets` or
  a small synthetic dataset, and list what the reader needs under `What you will need` in a tutorial or at the end of
  the lead in a how-to.
- Keep scaffolding short. The recipe is the point of a how-to. Label a model or dataset that stands in for the
  reader's own as a stand-in.
- Hide scaffolding. Tag a cell that only builds a dataset or model for the example, and that the reader would replace
  with their own, with `tags=["hide-input"]`. The page shows the cell collapsed under "Show code cell source", the cell
  still runs, and the `.py` file and Colab show it in full. Do not hide a cell that is part of the recipe, and do not
  hide a cell whose code the text refers to.
- Comment only what is not obvious from the code, and keep the comment short.

### The Colab cell

Put the install line in a hidden cell at the start of the setup. List the extras and packages the notebook needs.

```python
# %% tags=["remove_cell"]
try:
    import google.colab  # noqa: F401

    # specify the version of DataEval (==X.XX.X) for versions other than the latest
    # %pip install -q dataeval maite-datasets
except Exception:
    pass
```

### Assertion cells

A sentence that states a result needs a check that fails when the result changes. Put the check in a hidden cell:

```python
# %% tags=["remove_cell"]
# TEST ASSERTION CELL ###
assert 0 < len(outliers) <= 0.05 * len(dataset)
```

The docs build executes a notebook whose code cells changed and fails if an assertion fails
(`nb_execution_raise_on_error`). Assert a threshold or a range for anything that varies between hardware or runs.
Training results differ between CPU and GPU, and some statistics vary slightly between runs. When you merge, trim or
delete a notebook, keep every assertion cell it had or replace it with an equivalent one.

In a how-to, the visible sections show the task and the assertion cells hold the tests.

### Seeds, devices and output

- Set the seed once near the top with `set_seed`. Set the device once with `set_device`, and do not pass `device=` to
  each extractor unless the example is about devices.
- Do not let the prose depend on output that varies between machines, such as paths, timings, the device or training
  results.
- The notebook cache is keyed on the code cells, comments included. Editing a code comment re-executes that notebook
  in the next build. A change in `src/` does not invalidate the cache, so run `uv run nox -e docs -- clean` when a
  library change could alter outputs.

## Create embeddings the standard way

Every notebook, page and docstring example that creates embeddings follows one recipe.
[How to create embeddings](source/notebooks/h2_create_embeddings.py) runs it with PyTorch, ONNX and the extractors that
need no model.

```python
extractor = TorchExtractor(model, transforms=transforms, layer_name="avgpool")
embeddings = Embeddings(dataset, extractor=extractor, batch_size=64)
```

Rules (review):

1. Pass the dataset positionally and everything else by keyword: `Embeddings(dataset, extractor=..., batch_size=...)`.
2. Always pass `batch_size=` to `Embeddings`, so the value is visible in the cell. Set an extractor's own `batch_size`
   only to bound GPU memory, and say so.
3. Set the device once with `set_device`.
4. Subset the dataset and not the embeddings: `View(dataset, Limit(n))`.
5. Pass the `Embeddings` object itself to evaluators and `core` functions. Use `np.asarray(embeddings)` when you need
   a plain array. Never write `np.asarray(embeddings[:])`.
6. Build one `Embeddings` for each dataset and share it. Build one extractor for each configuration.
7. For PyTorch, use a pre-trained model, pass `transforms=` by keyword and name the embedding layer with `layer_name`.
   Never replace the head with an untrained layer.
8. For ONNX, mirror PyTorch: `to_encoding_model(path)`, then
   `OnnxExtractor(encoding_model, transforms=..., output_name=layer)`, with the same model and preprocessing as the
   PyTorch example.
9. Call `FlattenExtractor` a baseline wherever you use it. It suits small images, such as MNIST, and is not an
   assessment.
10. Fit `BoVWExtractor` explicitly on a reference set before it goes into `Embeddings` or `Duplicates`. An unfitted
    one fits on its first batch only.
11. Keep extractors that return one row for each detection, or a dictionary, out of `Embeddings`. That includes
    `ScoresExtractor` on a detection model, `DetectionGeometryExtractor`, `TorchExtractor(postprocess_fn=...)` and
    `ClasswiseUncertaintyExtractor`. Pass them to a drift detector as `extractor=`, or call them and pass the array.
    For `ClasswiseUncertaintyExtractor`, select one class's array from the result.
12. Do not rescale embeddings by hand. `Coverage` rescales them, and `coverage_adaptive` takes
    `force_unit_interval=True`.

`Embeddings` checks the row count of every batch. A matching count does not prove that each row belongs to the image
at the same position, so rule 11 matters even when no error is raised. Different batch sizes can change embeddings in
their last digits. They do not change them otherwise.

## Links, roles and fences

### Roles

Write roles in the MyST spelling in Markdown, and in the reST spelling in docstrings.

| Where | Spelling | Example |
| ----- | -------- | ------- |
| Markdown pages and notebook cells | MyST | `` {class}`.Coverage` ``, `` {func}`.cluster` ``, `` {meth}`.Metadata.agg` ``, and the same form for `attr`, `mod`, `term` and `doc` |
| Docstrings | reST | `` :class:`~dataeval.scope.Coverage` `` |

`tests/test_doc_markup.py` fails when a reST role appears in Markdown, because Sphinx renders it as literal text.
`tests/test_doc_cross_references.py` resolves every role target against the public API. A role that points at a
private member fails it, so write a private name as plain code, such as `_bin`.

### Links

- Link to a notebook by its source path: `./h2_create_embeddings.py` from another notebook and
  `../notebooks/h2_create_embeddings.py` from a concept page. Never link to the `.html` or `.ipynb` file.
- Use the exact title of the target page as the link text, or the heading that an anchor names. When you retitle a page,
  search for the old title and update every link that repeats it. `tests/test_notebook_structure.py` checks the links
  in Next steps.
- Heading anchors are generated to depth 4, and the docs build fails on a missing anchor. Prefer a link to a page over
  a link to a section of a notebook. Links from outside the documentation to a section cannot be checked, and they
  break when you rename the heading.
- Use an explicit label, `(name)=` on the line before a heading, for an anchor that must survive a rename.

### Fences and admonitions

- Put the opening and the closing fence of a directive on lines of their own. Do not put text on the fence line.
- A fence closes only with a fence of the same length or longer. Put a code example that uses three backticks inside a
  directive that uses four.
- Use `:::{note}` or a backtick fence for admonitions.

`tests/test_doc_markup.py` checks that every fence closes the block it opened and that no code block is empty. When
you re-wrap a paragraph by script, stop at fence lines, or the script joins them into the text.

## Register, rename or remove a notebook

### Add a notebook

1. Add the file name to the `toctree` of the matching section in `docs/source/tutorials/index.md` or
   `docs/source/how-to/index.md`.
2. Add a row to the list table of that section, copying a neighboring row. The title comes from the notebook's H1 when
   you write `[](../notebooks/<name>.py)`. Add a one-sentence description and the Colab badge.
3. Define the Colab reference at the end of the page, for example
   `[label]: https://colab.research.google.com/github/aria-ml/dataeval/blob/docs-artifacts/<version>/notebooks/<name>.ipynb`.
4. Link the notebook from the concept page it belongs to, under `See this in practice`.

### Rename or remove a notebook

- Search every reference with `git grep -n <name>`, and update the toctree entry, the list row, the Colab reference
  and every link in the same commit.
- The Colab link points at the artifacts of the last release, and the release script rewrites only the version in the
  URL. The file name in the URL must therefore be right. A new or renamed notebook does not open from its badge until
  the next release publishes it.
- A rename changes the published address. The documentation has no redirects.
- Check the open merge requests for new links to the old name before you merge the rename.

### Concept page structure

End a concept page with `Related concept pages`, then `See this in practice` (with `How-to guides` and `Tutorials`
lists), then `References`, when they apply.

## Write docstrings

- Use numpydoc. Write full blocks (Parameters, Returns, Raises, Notes, Example) on every public member, including
  dunder methods and property setters. Give the reason for a design choice in the docstring.
- Write a one-line summary, or a short paragraph, on a private member. Do not add Parameters or Returns sections.
- Run the examples in docstrings with `uv run nox -s doctest`, and pass `batch_size=` to `Embeddings` in them.
- Edit a public docstring only when its content is wrong, and say in the merge request that the edit is a correctness
  fix.

## Check your work

Run the checks that apply before you open a merge request.

| Command | What it checks |
| ------- | -------------- |
| `uv run nox -s lint` | `ruff`, `codespell` (American spelling) and `typos` |
| `uv run nox -s test -- tests/test_doc_markup.py tests/test_doc_cross_references.py tests/test_deprecated_docs.py tests/test_quickstart.py tests/test_notebook_structure.py --cov-fail-under=0` | reST roles in Markdown, fence closing, role targets, deprecated documentation, the quickstart code, and the headings, template sections and Next steps of the notebooks |
| `uv run nox -s type` | `pyright` over `src`, `tests`, `verification` and `docs/source/notebooks`. Run it after you change a code cell in a notebook or a test, because it checks them too |
| `npx --yes markdownlint-cli2 '*.md' 'docs/**/*.md' '#node_modules'` | The Markdown rules that CI enforces, such as 120 columns, one blank line between blocks and no repeated heading in a page. Run it after every change to a `.md` file. It needs Node, and `npx` fetches the tool on first use |
| `uv run nox -s docs` | Converts the notebooks, executes the ones whose code cells changed, runs their assertions and builds the site with `--fail-on-warning`, which catches broken links, anchors and toctree entries |
| `uv run nox -s docs -- skip` | The same build without executing notebooks, for a change to prose only |
| `uv run nox -e docs -- clean` | Clears the notebook cache and re-executes everything, for a library change that could alter outputs |
| `uv run nox -s doctest` | The examples in docstrings |

A single run of the documentation tests with `--no-cov` exits non-zero even when every test passes, so use
`--cov-fail-under=0` as shown.

What is not automated today: voice, emphasis, contractions, what a heading or a Next steps description says, the choice
of Next steps links, the `See this in practice` lists of the concept pages, the 600-word limit (it warns) and the width
of notebook text. A reviewer checks them.

A notebook that needs more memory than your machine has cannot execute in your local docs build. Run its cells one at
a time instead, and rely on the CI docs job for the full run. Say in the merge request what you could and could not
run.

## Review a prose-only change

A change that only rewrites wording must leave these unchanged in every notebook: code cells, headings, link targets,
roles, labels, numbers and the values in tables. It must also leave the three tutorial metadata lines and the Next
steps separators as they are. The emphasis and the column heading case of a table follow the rules above, so the pass may
change them.

Compare the before and after trees to confirm it. Read the diff for sentences that gained a condition or lost a hedge.
"We recommend using image embeddings" becoming "Use image embeddings if you can" adds a condition and is a wording
change that alters the meaning.

## Open a documentation merge request

- Start the title with `[docs]`, and write it so that it reads as a changelog entry. The changelog is generated from
  the title and from commit subjects. [BRANCHING.md](../BRANCHING.md) explains the prefixes.
- Make one deliverable for each merge request. Do not bundle unrelated documentation changes, because they become one
  changelog line.
- Write each commit message as one line, `[docs] Subject`, with no body and no trailer.
- Leave the cache alone. The main pipeline reuses cached notebook outputs unless a code cell or a shared execution
  input such as `src/` changed.
- List the checks you ran and the ones you could not run.
