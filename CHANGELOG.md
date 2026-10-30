# DataEval Change Log

## v1.2.0-rc1

🌟 **Feature Release**

- `407694c2a` - [feat] Let VideoTorchExtractor embed MAITE video streams
- `d6dd84a11` - [feat] Widen the evaluator dataset types with ImageDataset and VideoDataset, and measure videos that carry no tracking targets or metadata

👾 **Fixes**

- `f890a1913` - [fix] Split on the raw values of a grouping factor so every video is its own group at any size
- `4ab7a26d1` - [fix] Refuse a split that leaves a partition empty and name groups in split errors
- `b7497dbe9` - [fix] Accept a tracking dataset in the type of Outliers.evaluate and return results keyed by source address

📝 **Miscellaneous**

- `eeafefb3e` - [docs] Point the zero-numerator reference at PubMed instead of the unreachable McGill PDF
- `0e16a63d3` - [docs] Mark the SharpnessMeasure protocol experimental with the PERSISTENCE maps
- `9d66a9dd3` - [docs] State the PERSISTENCE evidence as findings and tighten the prose of the new video how-tos and migration notes
- `266544e79` - [test] Check the video workflow end to end and register its verification evidence
- `ce4261787` - [docs] Define the frame and target tuples in the video notebooks instead of requiring maite-datasets
- `f7cc814aa` - [docs] Add a how-to that selects frames from a video dataset for image tools with Stride, EvenlySpaced and Representative
- `71cd62977` - [docs] Add a how-to that wraps on-disk video and tracking annotations as a MOT dataset with PyAV and serves its named factors
- `aa424bda4` - [devops] Publish to PyPI with PEP 740 attestations through the PyPI publish action

## v1.2.0-rc0

🌟 **Feature Release**

- `a0544e490` - [feat] Evaluate Outliers on a tracking dataset in video terms
- `32a8996ec` - [feat] Export the experimental DriftKnownDomains detector from dataeval.shift
- `844314206` - [feat] Add VideoStats PERSISTENCE: per-video maps of what is fixed to the camera and of local blur
- `d3f954658` - [feat] Default an unset Duplicates hash_radius to 6 for video and 0 for images, resolved per evaluation
- `ea4e62d5c` - [feat] Add the experimental Feasibility evaluator, which decides whether a classification accuracy or AP target is reachable on a dataset and returns the evidence behind each answer, and keep ber_knn lower bounds below the Bayes error
- `c6176a15e` - [feat] Exclude a burned-in overlay from camera ego-motion with mask= on ego_stats, video_stats and CameraTravel
- `83b869aed` - [feat] Add CameraTravel, keeping a frame each time the estimated camera travel reaches a set fraction of the frame, and refuse ego readings optical flow cannot make
- `3350abb08` - [feat] Select frames by image quality or annotations with QualityGate, TrackEvents, BestPerTrack and HasLabels, and report what each selection stage kept
- `a04988de0` - [feat] Compose frame selectors with Chain, and make frames_represented an exact count of the source frames each kept frame stands for
- `930667639` - [feat] Trace frame-view findings to their source videos with SequenceFrames.to_source, a selection fixed by its first walk, and partial keyed source_index
- `7bba36f35` - [feat] Report where Metadata.add_factors values disagree with existing factors, keep overwritten columns as <name>_original, and record each factor's source name
- `f321b876c` - [feat] Add VideoStats and video_stats for per-video statistics
- `0da47ce38` - [feat] Make Parity valid over grouped data with per-cell findings, and label_parity a two-sample homogeneity test
- `9d4c266a5` - [feat] Add label_cooccurrence stats and stats on Metadata factors
- `eb2bae419` - [feat] Judge an annotated dataset through the Metadata built for it, attach dataset and metadata to Outliers output, and prefix per-level metric names
- `5a36ed12e` - [feat] Add known metadata factors with validated definitions, documented on the ExtrinsicFactors concept page
- `7f154f6b8` - [feat] Add DriftKnownDomains detector
- `8514871c6` - [feat] Judge metadata factors against their sequence neighbors in Outliers with neighbors= and dataeval.core.sequence_residuals
- `4bf5042de` - [feat] Stratify Outliers, floor small populations, judge categorical rarity and flag unusual factor combinations
- `487eba35e` - [feat] Score how unusual each row is as a combination of metadata factors with dataeval.core.combination_scores
- `d4375267c` - [feat] Judge Outliers at named levels with levels= and level views
- `af8148b30` - [feat] Filter dataset views by class and metadata factor, replacing ClassFilter
- `764cf3e89` - [feat] Metadata analysis in Outliers evaluator
- `dfc6146be` - [feat] Duplicates.deduplicate and Outliers.prune return a RemovalPlan of items, detections and tracks, which Indices(plan, exclude=True) removes
- `75ced475c` - [feat] chunked() takes incomplete to keep, drop or append the reference's short final chunk
- `b27eb0cc9` - [feat] Metadata.unusable_rows names the items and rows that hold an unusable factor's values
- `770f5a8c6` - [feat] Metadata analysis in Duplicates evaluator
- `5593b270a` - [feat] Generate the metarepo test-results.log from nox -s verify
- `ea34f376f` - [feat] Add ego_stats for per-frame camera motion factors
- `eb12e58e9` - [feat] Let stats producers declare how their factors roll up to a coarser level
- `ca6610293` - [feat] Add alpha prerelease tier selected with PRERELEASE_TYPE
- `e3dd95330` - [feat] Take parity's G-test evidence against entities rather than rows
- `9dddcbfbb` - [feat] Count the distinct entities a level offers over the rows in view
- `da74444c7` - [feat] Take the chance correction against entities rather than rows in mutual_info
- `cd71378b8` - [feat] Integrate new frame selection methods
- `867beac61` - [feat] Enable MOT dataset for Duplicates evaluator
- `9b5faa249` - [feat] Persist last_aggregation so a restored metadata keeps its coverage diagnostics
- `c049f1312` - [feat] Accept corrections and aggregations as constructor arguments
- `8ca0a65d3` - [feat] Export the factor vocabularies and array helpers as public API
- `8204487cc` - [feat] Allow Metadata to handle mixed data inputs and apply corrective measures
- `8088b79c9` - [feat] Metadata Aggregator to move metadata from bottom up
- `56f41a3b8` - [feat] Add ability to validate MOT datasets
- `dfba223b0` - [feat] First class metadata feature extractor
- `c0809307e` - [feat] Allow custom classed_by axis for Metadata
- `22cb6ee38` - [feat] Allow band groups for stats calculations
- `77f95fb63` - [feat] Expose source item and target addresses on DetectionCrops
- `f0993f59c` - [feat] Add SourceLocator for SourceIndex lookup
- `3a63bf56c` - [feat] SourceIndex becomes a universal address
- `e8789fc49` - [feat] Merge FMV development branch work

🚧 **Deprecations and Removals**

- `0ec454836` - [depr] Remove all deprecated functionality declared in v1.1

🛠️ **Improvements and Enhancements**

- `ed86ddae7` - [impr] Remove Duplicates frame_sample so frames are chosen only through SequenceFrames, and name the selector in execution records
- `3b4331e9b` - [impr] Warn when BoVWExtractor fits its vocabulary inside a single call, and fit it explicitly in the augmentation tutorial
- `c509aa6cf` - [impr] Convert RGB frames to grayscale about 5x faster at 4K, bit-identical to before, by summing luma into one float64 buffer
- `fdefe88e0` - [impr] Tidy VideoStats follow-ups: abstract calculator hooks, grouped hash warnings, docs and tests
- `b84fa6cac` - [impr] Move image_stats into its own _image_stats module
- `20635ae2d` - [impr] Rename compute_stats to image_stats
- `3f295f06a` - [impr] Cache the docs datasets in Harbor from the docs CI job, outside the docs build
- `e8244ff5b` - [impr] Deprecate compute_ratios in favor of the rel_* statistics and Metadata instance and unit columns
- `1da6a8500` - [impr] Run BoVW on threads so set_max_processes speeds it up
- `a19219f61` - [impr] Drop python 3.10 support
- `b48fb639f` - [impr] Publish main-<variant> from the branch and point latest at the newest release
- `5173baec5` - [impr] Publish only :latest from main and version tags from releases
- `b1238053c` - [impr] Expand dup_types input to accept str and add some tests
- `859a58f03` - [impr] Add dropped_indices attribute and drop_empty param
- `07881cb25` - [impr] Share the effective sample size helpers between the two corrected statistics
- `3e41e3743` - [impr] Return an empty tuple instead of None for DriftOutput.feature_names
- `f6b8948ab` - [impr] Make repair answer with a copy, as aggregate already does
- `63312b8b4` - [impr] Use metadata levels first as a qualifier when resolving factor names
- `4997d3b01` - [impr] Remove id from metadata factors
- `f485345b7` - [impr] Add exclude option for Indices
- `f0bd45ea5` - [impr] Declare each task's level model once, outside the metadata package

👾 **Fixes**

- `faa46a007` - [fix] Key ego_stats frame rows by each frame's own number so they attach to the Metadata rows of a selected or offset video
- `8de951b85` - [fix] Fix ClassBalance failing on a tracking dataset with KeyError and the stock image extractors returning a row of objects for a video stream
- `449f1a130` - [fix] Fix the naive coverage radius overflow on wide embeddings and the unclear errors from image statistics on tracking datasets and from Prioritize
- `6b6734344` - [fix] Keep each split_on group in one partition when splitting detection and frame datasets, which v1.1 ignored
- `9e2c55452` - [fix] Raise from image_stats when images and boxes differ in length instead of dropping boxes or leaving images unannotated
- `b89e6934b` - [fix] Return a full-image box, width and height the right way round, for a missing box on a non-square or 2-D image
- `45566d5c6` - [fix] Correct the quickstart's torchvision requirement, the JATIC name and two page descriptions
- `67b5844a3` - [fix] State the inputs that Duplicates and Outliers accept on the input requirements page
- `53b9bfbc8` - [fix] Correct wrong and overstated claims on the concept pages
- `b4739140f` - [fix] Refuse an ego fit that settles between the camera and a large mover, judged per frame of gap
- `1d33bae9b` - [fix] Make Parity and Balance treat the source video or image as the independent unit over frame, crop and merged views
- `b1609b538` - [fix] Show the missing figure in the clean dataset tutorial and correct statements in eight notebooks that the library or the environment contradict
- `33abca407` - [fix] Correct statements in six tutorials that the library or the tutorials' own results contradict
- `6d5894557` - [fix] Stop Duplicates warning about the hash_radius default when no hash statistic is requested
- `18c374d72` - [fix] Make dataeval.log reuse its default handler so repeated calls stop duplicating log lines
- `1232c3c1e` - [fix] Improve tests and fix warnings and other lint issues
- `ac0487423` - [fix] Compute GMM covariances without an N x K x D x D tensor so OODReconstruction fits large reference sets
- `755687e29` - [fix] Correct the quickstart Duplicates example and run it in a smoke test
- `e1c72dac7` - [fix] Make Embeddings reject extractor output with the wrong row count or no array, and state the one-row-per-item contract
- `3ef99ecc8` - [fix] Fail the docs tests on reST roles and mismatched fences in Markdown, and correct the ones found
- `3e06ed8a4` - [fix] Replace the private API imports in two notebooks and correct stale text in four
- `328ab8e0e` - [fix] Refuse an ego reading whose fitted motion does not explain the new frame, so CameraTravel keeps the first frame after a hard cut or field-of-view switch
- `e4a5afedc` - [fix] Scale integer images deeper than 8 bits into grayscale from their decoded encoding, so 16-bit IR no longer saturates to a flat white frame
- `6466823cb` - [fix] Read variability across unrecorded readings, so a thinned frame column no longer rolls up to null
- `e6b7032fa` - [fix] Add Metadata.dataset and stop embedding Metadata factor rows as images
- `dccd319a4` - [fix] Cap xdist workers at the CPUs the container may use
- `50c8d1724` - [fix] Re-baseline the DriftKNeighbors time budget for the 2-CPU k8s medium runner
- `2c5b843f1` - [fix] Read xdist workers from TEST_WORKERS, 2 on the medium CI runner
- `0ad54d3a0` - [fix] Mark the project dir as a git safe.directory on the k8s runners
- `61c1e2087` - [fix] polars version-aware categorical in outliers tests
- `31974a5f4` - [fix] Stop using polars' deprecated Categorical ordering argument and torch.jit.script
- `faf9ff3ed` - [fix] Keep missing metadata values null and report factors mixing booleans with numbers or text as mixed_types
- `ba1b4ed50` - [fix] Build the aggregate working frame with its group column so polars 2 does not fail to broadcast
- `354e5fbe0` - [fix] Update how-to assertion for duplicate
- `98f6949d7` - [fix] ClassFilter and Indices keep an (H, W) segmentation label map whole, where they cut rows of pixels when its height matched the detection count
- `1ae6fde6e` - [fix] Outliers.evaluate reads pixel statistics in stored units, agreeing with...
- `e1dbbac79` - [fix] ChunkedDrift accepts any Chunker the protocol describes rather than requiring a private split method
- `d6171b209` - [fix] ChunkedDrift.fit rejects a reference that splits into fewer than 3 chunks unless its threshold is constant
- `07d95468b` - [fix] Force tag fetches so a re-cut release tag does not fail the job
- `8b9789980` - [fix] Spell the Harbor robot name inline so GitLab cannot expand it
- `c7ea2e368` - [fix] Rebuild the dataeval wheel so images carry the resolved version
- `38332157a` - [fix] Generate Dockerfiles without depending on git tag history
- `d43d84da4` - [fix] Keep main from overwriting the released verification evidence
- `90376fb5f` - [fix] Update docstrings for litert extras
- `b3219852d` - [fix] Fail loudly on incorrect params and typing improvements
- `78c4eeb59` - [fix] Harden release cutting against stale refs and missing series tags
- `4fe9dd5be` - [fix] Mark prerelease tags as prereleases on the GitHub release
- `bc6822004` - [fix] Versioning constraints for scipy
- `e79326861` - [fix] Accept integer feature matrices in the ranking core
- `6c40fb416` - [fix] MaskedTarget is not properly wrapping MAITE targets
- `66d6e09a5` - [fix] Update the Parity evaluator surface for the corrected statistic and flag
- `550e3dbcd` - [fix] Compute parity from Pearson's chi-square and judge sufficiency by Cochran's rule
- `680bc8693` - [fix] Stop Parity rejecting independence on replication alone
- `3be01134a` - [fix] Stop Balance reading replication as correlation between coarse factors
- `acfdb18f1` - [fix] Read a threshold spec written as lists, as JSON and YAML deliver it
- `d832d8e1c` - [fix] Read a descriptor's corrections back through the encoding constructor path
- `846a626f0` - [fix] Bump descriptor version to 3 for the four-kind correction vocabulary
- `1b2be30b5` - [fix] Adjust Relabel to match API structure of v1.1 patch release
- `39d395b25` - [fix] Read detection scores down to one confidence per detection
- `63f50ed55` - [fix] Allow merge_datasets to accept packed collections of datasets
- `1a001a92f` - [fix] Reject a nested column as a key instead of panicking on polars 1.0.0

📝 **Miscellaneous**

- `ab4a87c7c` - [devops] Install the graphviz neato layout plugin for the docs build
- `7a04812de` - [docs] Add Outliers on tracking datasets, DriftKnownDomains and the persistence limits to the roadmap and v1.2 migration guide
- `81e84e575` - [devsecops] Quiet test output for doctest and min deps
- `fe0a68f78` - [docs] Update the roadmap and migration guide for the v1.2 release candidate
- `19781f918` - [docs] Rewrite the Functional overview around the dataset task, with a Video tracking column, an adapters section, the missing functions and corrected task flags
- `c9be8594e` - [docs] Say that Outliers cluster_distance reports the raw distance to the cluster center, not standard deviations
- `ebd651574` - [docs] Use sentence case for every title, rename the Beginner's guide page to Quickstart and correct the JATIC and platform wording
- `bd8127f2f` - [docs] Describe the real-footage video studies on the Validation and Trust page, with their footage, method and results
- `71180ba40` - [docs] Bring the getting-started and index pages into house voice
- `704ae1fca` - [docs] Bring the concept pages into house voice
- `13323149a` - [docs] Bring the Leakage concept page into house voice
- `e089e51c8` - [devsecops] Reduce verbosity of test logs
- `a518e06e6` - [devsecops] Improve docs notebook cache information
- `aad3fd002` - [misc] Drop Sphinx style attribute comments
- `2ff70deac` - [docs] Wrap the Markdown cells of the tutorials and how-tos at 120 columns and check the width in the structure test
- `58ff626e2` - [test] Check the voice and emphasis rules of the style guide in the notebook structure test and fix the four sentences that break them
- `ab46070ec` - [docs] Rewrite the monitoring how-tos in house style
- `07390fff3` - [docs] Rewrite the ontology, configuration and model development how-tos in house style
- `a3d943022` - [docs] Rewrite the metadata and bias how-tos in house style
- `4b6d72331` - [docs] Rewrite the data quality how-tos in house style
- `e61eaa0d8` - [docs] Rewrite the data preparation how-tos in house style
- `eb7ec7ba6` - [docs] Correct the classes with the fewest objects and images in the clean dataset tutorial and check them
- `218353ce1` - [docs] Rewrite the monitoring tutorials in house style
- `5dbf36e71` - [docs] Rewrite the clean dataset, surroundings and bias tutorials in house style
- `822c456bf` - [docs] Keep the only-those-detections meaning in the image-level filter sentence of the levels tutorial
- `a6485ced4` - [docs] Rewrite the levels and duplicate video tutorials in house style
- `9da8b94ec` - [docs] Rewrite the data space, dataset gaps and class axis tutorials in house style
- `527228f30` - [docs] Rewrite the Feasibility tutorial in house style
- `00dfa0f63` - [docs] Rewrite the augmentation duplicates tutorial in house style and apply the emphasis rules to tables
- `c78c424b8` - [test] Add assertion cells to seven notebooks whose prose quotes results
- `691ba9976` - [docs] Update the v1.2 release checklist with the Duplicates decisions and two correctness fixes
- `e843865a3` - [devsecops] Skip the documentation and repository tests in the conda lane
- `9dea354bd` - [devsecops] Quiet the git tag fetch in docs job
- `0855f8d2a` - [devsecops] Use cache invalidation for notebook execution on main
- `0532dcd85` - [docs] Cut the metadata preparation in the bias tutorial to one short step and check the numbers its findings quote
- `ea61e0910` - [test] Check that Next steps links use the title of their target page and correct the ones that do not
- `1b0b8a76e` - [devsecops] Run typecheck on merge requests that change the notebooks, verification or pyproject.toml
- `95d35e958` - [docs] Link the unlinked how-tos and tutorials from their concept pages and use their titles in the See this in practice lists
- `03028a618` - [docs] Group the data engineering how-tos by task and refresh stale descriptions on the how-to and tutorial index pages
- `ff75cd19c` - [test] Check the headings, template sections and Next steps of the tutorials and how-tos
- `25ace3750` - [docs] List every reason an ego reading is refused, on the concept page and in the ego_trusted field doc
- `d9b79c6fb` - [docs] Add the missing Conclusion sections to three tutorials and bring two how-tos within the template limits
- `1cd7691a2` - [docs] Bring the Feasibility tutorial, how-to and concept pages into house style
- `7ce6e7d4c` - [docs] Replace the unsupported guidance in the OOD and augmentation tutorials with observations tied to their printed results
- `e3aa430d6` - [docs] Point the tool-choice guide at Coverage and say what it adds in the data space tutorial
- `ec7a448f5` - [docs] Fix the pyright error in the hidden assertion cell of the level tutorial
- `b51d969d8` - [docs] Replace the docsync advice on the how-to and tutorial index pages with the docs session and jupytext
- `0230526c0` - [docs] Cut the label parity section from the monitor-shift tutorial and link the label independence how-to
- `cfd0b77c8` - [docs] Remove the cleaning-issues how-to and add its classwise step to the clean-a-dataset tutorial
- `dd92f7c53` - [docs] Fold the bin-by-level how-to into the level tutorial, trim the tutorial and move its sync and cache steps into a how-to
- `839505d82` - [docs] Move the transform placement rule to the concept pages and turn the place-transforms how-to into the StatsInvalidatedWarning recipe
- `6f94ab35f` - [docs] Share a base class between the DataFrame adapters so the detection adapter type-checks without ignores
- `aedfba52e` - [docs] Fix pyright errors in the intrinsic factors and DataFrame wrapping how-tos
- `334697a37` - [docs] Combine the DataFrame wrapping and lazy image loading how-tos into one how-to
- `7bf62fca6` - [docs] Apply the simplified how-to template to the intrinsic factor, factor binning and channel group how-tos and trim them to the recipe
- `426a9f0f9` - [devops] Prestage MilitaryVehicles for the docs build so the Harbor datasets cache stores it
- `7fcc792d7` - [test] Seed the outliers from_clusters no-outliers test data so an unlucky draw cannot flag a point
- `a6c8b729f` - [docs] Apply the simplified how-to template to the outlier and tracing how-tos and trim them to the recipe
- `0cd41f733` - [docs] Apply the simplified how-to template to the configuration and drift how-tos and trim them to the recipe
- `ec8b42774` - [docs] Apply the simplified how-to template to nine how-tos
- `f42f8694b` - [devsecops] Quiet system package installation in CI
- `8c33eb250` - [docs] Trim the data preparation how-tos to the recipe and simplify the how-to template
- `8b94fd105` - [docs] Promote the video duplicates how-to to the Identify duplicate video tutorial
- `9ed654a11` - [docs] Promote the custom class axis how-to to the Analyze bias against a custom class axis tutorial
- `ea9c47184` - [docs] Curate the Next steps lists of the data preparation, metadata and performance how-tos to at most four links
- `6479216bf` - [docs] Curate the Next steps lists of the quality, coverage and monitoring how-tos to at most four links
- `26bdb3e51` - [docs] Curate the Next steps lists of the tutorials to at most four links
- `a9d1022bc` - [devsecops] Prune the uv cache from hatchling's project walk so verify jobs stop failing with FileNotFoundError
- `1f6e8793c` - [docs] Add a documentation style guide and link it from CONTRIBUTING
- `34b1b9056` - [docs] Remove the summaries and takeaways that closed ten how-tos with a lesson
- `4d65088ee` - [docs] Normalize the headings of the tutorials and how-tos and update the references that used the old ones
- `f9f6ccd6b` - [docs] Measure coverage in the undersampling how-to with Coverage and label its autoencoder as a stand-in for your own model
- `56ddd2dbb` - [docs] Build the data space tutorial embeddings with the standard avgpool block and drop its unneeded normalization step
- `597d0c7bc` - [docs] Rename the embeddings how-to to h2_create_embeddings and update the links that described it as an ONNX guide
- `a9ef6be50` - [docs] Align the divergence how-to on the standard Embeddings recipe and correct its claims about the divergence values
- `a6d1ea11c` - [docs] Correct the Embeddings docstring on whether batching changes the embeddings
- `b7a011a32` - [docs] Align the notebooks on the standard Embeddings recipe and point them to the embeddings how-to
- `c75bed37f` - [docs] Rewrite the Embeddings concept page around one standard recipe and document how each tool takes embeddings
- `b5331cb3b` - [docs] Rewrite the ONNX encoding how-to as a guide to creating embeddings with PyTorch, ONNX and model-free extractors
- `5d54f7138` - [docs] Normalize the headings and rewrite the prose of the dataset gaps tutorial and the clustering how-to
- `a027a9544` - [deps] Require filelock>=4.2.1 for its fix of CVE-2026-112940
- `736a02ac8` - [deps] Require pillow>=13.0.0 for its security release fixing nine CVEs
- `08a0ac6f2` - [misc] Run main-branch docs on k8s-single-gpu
- `e6b18ec37` - [docs] Add a how-to on judging metadata factors within strata, by rarity, given another factor and in combination
- `0b41da8fd` - [misc] Run verify on k8s-medium-cpu
- `cb557df23` - [misc] Run typecheck on k8s-medium-cpu
- `4add816f5` - [misc] Move CI jobs to k8s runners, medium for test, conda, doctest and deps
- `12936b20d` - [misc] Add oup.com and nih.gov links to skip pattern
- `2941508cb` - [deps] Bump locked polars version to 2.0.0
- `4c13d026a` - [misc] Record CI job results as verification evidence, run the gate on release tags, check PyPI, and enforce type completeness and DataEval deprecation warnings
- `745a7172f` - [misc] Cite the real CI job names in the verification registry and check them in tests
- `19a018dcd` - [misc] Report a verification test case with skipped steps as skipped, not passed
- `99e3c8d34` - [misc] Publish verification evidence only from version tags and release branches
- `5a78c933a` - [misc] Generate requirements, test cases, VCRM, and test log from a step-based verification registry (port from release/v1.1)
- `5e90c587c` - [docs] Note the containerization exemption and point to DataEval-Flow
- `4f3a2547d` - [devsecops] Remove container export
- `ba5d56ea5` - [docs] Update ROADMAP.md
- `494fda0bc` - [docs] Say dataset instead of corpus in the guides, glossary and docstrings
- `3ea160f73` - [docs] Label only approved mature releases on the maturity page
- `1e2690461` - [docs] Track the v1.3 draft product standards on the maturity page
- `164b7ae18` - [docs] Add a JATIC maturity reference page recording the level each release holds
- `e88d54532` - [devops] Keep security scans out of publish-verification pipelines
- `811428a35` - [test] Register NFR-5 container packaging and map the container contract suite to it
- `c410f7e06` - [misc] Fix lint
- `39a0ba083` - [devsecops] Ship the SBOM inside the image alongside the scan report
- `f43ef1bf7` - [devsecops] Ship the vulnerability scan report inside the published image
- `20cb0a7eb` - [docs] Point the cosign link at a URL sigstore still serves
- `21216bdf5` - [docs] Document the two container tags that are actually published
- `31ccba361` - [devsecops] Add containerization infrastructure
- `f89a07db6` - [misc] Update verification tests
- `43515a9d1` - [misc] Update spelling in codebase
- `de733aa38` - [test] Add video files for internal tests
- `bd41735f3` - [docs] Align markdown table columns to satisfy MD060
- `cab7169ae` - [docs] Name the GitHub mirror as the publish path and its misleading status
- `c88fe8359` - [devops] Publish releases from version tags instead of pipeline variables
- `bc46b9f65` - [test] Add self-check for release version and category logic
- `ffe70bd65` - [devops] Validate merge request title prefix instead of release label
- `9e068e354` - [devops] Add release script to cut releases from a local checkout
- `6f6ba3436` - [devsecops] Workaround SAST path ignore list
- `3474cea2d` - [devsecops] Fix false positive SAST scans in documentation support files
- `dec3388ca` - [test] Add coverage to functions below 90%
- `7076c1218` - [deps] Fix open pin requirement for devsecops components
- `f36fd03b6` - Add capabilities to DuplicatesOutput and OutliersOutput
- `89a7a97a0` - Add video segmentation and stitching
- `40c42c950` - [devops] Rename CREATE_PRE_RELEASE to CREATE_PRERELEASE and normalize prerelease spelling
- `7a7892fe8` - [type] Wrap SciPy stats functions with typed results in dataeval.utils.scipy
- `dd98c1093` - [misc] Update docstring prose
- `de7285809` - [docs] Describe parity's statistic and sufficiency check as they now stand
- `338f7a578` - [docs] Drop the None fallback for DriftOutput.feature_names in the drift how-to
- `adf3c439c` - [type] Fix typecheck failures in scripts
- `b29ad0355` - [type] Adjust ThresholdLike protocol
- `ffed59426` - [imrp] Split internal utilities
- `268ca33bd` - [docs] Fetch intersphinx inventory with retry before generating docs
- `6afb75c25` - [docs] Update documentation with new maite-datasets
- `f3383148a` - [test] Boost per-function code coverage
- `c379b14af` - [misc] Mirror MAITE protocols to enable stable isinstance checks
- `a5fa0ac2f` - [misc] Fix py3.10 typecheck failure
- `11ee894e7` - [misc] Update typecheck dependencies for pyyaml and requests
- `95fa840df` - [test] Verify frame view provenance through View transforms
- `a34a6585d` - [misc] Fix documentation links and expand on Metadata levels
- `732ea7fb5` - [misc] Fix typos in README.md
- `38a6a2ed7` - [devsecops] Add additional test and typecheck fixes
- `759e5ef3f` - [devsecops] Switch to uv-support branch for components
- `4a131be2d` - [devsecops] Remove poetry from supported package managers
- `24e5fc5f6` - [devsecops] Simplify release branching strategy

## v1.1.0

🌟 **Feature Release**

- `b0ea258b` - [impr] Allow compute_stats to handle extended channel information
- `08620fe4` - [feat] Restructure metadata to support multiple modalities
- `53780702` - [feat] Add new view transform operations
- `dfe08764` - [feat] Add per_background statistics measuring the scene behind an image's boxes
- `dffc4ccf` - [impr] Generalize metadata levels to allow for modalities other than vision
- `ebe557fd` - [feat] Develop representation and coverage evaluators with label/ontology and embedding analysis
- `d0c4488f` - [feat] Add multi-object tracking (MOT) support: `Track` type, build_tracks, and per-track statistics
- `42410c52` - [feat] Add detection crop transform to help extract detection embeddings
- `8664c4bc` - [feat] Add ontology validation functionality to core
- `35a2f8da` - [feat] Add Conform orchestrator class and Relabel conformation class
- `15df90b0` - [feat] Implement ontology-based label alignment
- `8543cd02` - [feat] Implement class label reconciliation against ontology
- `dfc3b397` - [feat] Implement MAITE predict for prediction based features
- `bbde67cf` - [feat] Enhance uncertainty-based extractor and added Wasserstein drift detector

    - [docs] Add tutorial on drift detection via prediction uncertainty
    - [feat] Add DriftWasserstein detector with validation-calibrated baseline
    - [feat] Add Uncertainty and Classwise extractors, deprecate classifier variant
    - [feat] Add inference batching and output decoding to TorchExtractor
    - [feat] Add inference batch_size to OnnxExtractor
    - [feat] Resolve Embeddings batch size from extractor and global default
    - [impr] Support ragged inputs and pre/postprocess hooks in predict
    - [feat] Add resolve_batch_size helper to config
- `7c8243cf` - [feat] Add DriftWasserstein detector with validation-calibrated baseline
- `715cbeec` - [feat] Add Uncertainty and Classwise extractors, deprecate classifier variant
- `b031a062` - [feat] Add inference batching and output decoding to TorchExtractor
- `cd8c1c30` - [feat] Add inference batch_size to OnnxExtractor
- `1820e7c5` - [feat] Resolve Embeddings batch size from extractor and global default
- `19cafb63` - [feat] Add resolve_batch_size helper to config
- `7f2fef9e` - [feat] Add info schema and types for serializable descriptors of components

🛠️ **Improvements and Enhancements**

- `80617a40` - [impr] Expose code_names from Metadata
- `b0470902` - [impr] Metadata binning as Policy
- `aa7c0a4b` - [impr] Change balance and mutual_info binned behavior
- `2bf18e2d` - Fix some spelling errors
- `a380758f` - [impr] Normalize logging namespaces to public API
- `3a5550da` - [feat] Handle equivalence concepts in Ontology
- `8e4b5f18` - Fix add factors to work with multi-level stats
- `3d1eedd2` - [feat] Add MOTStructurer and sequence level
- `fa9acf4f` - [impr] Metadata restructure to prepare for additional dataset modalities
- `e7af0896` - Enhance Metadata add_factors to allow for combined factors
- `c51f889c` - [impr] Merge Select and Conform into View
- `9769eaef` - [impr] Expand input flexibility for all evaluators and helpers
- `d8824ab4` - [impr] Move data organization related concepts to data module
- `a6487bea` - [impr] Allow image size overrides per IR requirements
- `1b2967c9` - [impr] Support ragged inputs and pre/postprocess hooks in predict
- `6722372a` - Replace internal mirrors with MAITE imports
- `cd09eb42` - [impr] Make selections nest within a single Select class wrapper
- `05f42136` - Provide better input validation for MAITE datasets
- `3664bc1f` - Added isotropy attribute to CompletenessResult class.
- `077e5903` - Develop null model metrics for object detection
- `ed718126` - New sample-size-adaptive threshold for is_continuous()
- `f1dbe2c2` - Use our standard normalized mutual information in metadata insights.
- `7efe6a7e` - [impr] Optimize performance on compute_stats
- `bcb84589` - [impr] Expose source and root datasets from View and remove extraneous attribute from DetectionCrops

👾 **Fixes**

- `bb541b8d` - [fix] Normalize Linfoot branch by its reachable maximum and clip mixed pairs
- `577fd466` - [fix] Balance chance correction
- `2287c08f` - [fix] Typecheck issue in docs
- `1be1935c` - [fix] Drop and report level splits that hold no values at their level
- `f4d4a281` - [fix] Return NaN, not zero entropy, for a region that was never measured
- `94a00d1f` - [fix] Anchor bit depth on the whole datum so a box scales like its image
- `0f6ec081` - [fix] Drop parents listing itself as a child as trivial
- `d5f508bf` - [fix] Additional lint and type rules
- `c4a69cf8` - [fix] Counting bug in ClassBalance and update tests
- `c6a85c26` - [fix] Ensure no device mismatch on mmd and gmm tensors
- `eaf15b5d` - [fix] Address scipy>=1.18 change for NaN on mannwhitneyu test for zero variance values
- `b4382ca4` - [fix] noise points incorrectly flagged as duplicates in compare_links_to_cluster_std
- `ef14f677` - [fix] Build explicit polars dataframe to avoid inferring dtypes
- `9c75b1a5` - [fix] Avoid slicing MAITE datasets in extractor
- `ff4bd296` - [fix] Ensure MAITE datasets are compatible with extractors
- `6ae26348` - [fix] Address metadata related issues

    - fixed `Metadata.include` with a string interpreted as a sequence of chars
    - fixed `Metadata.add_factors` improperly dropping previous factors
    - fixed support for image vs target level factors in Metadata
    - fixed bias (`balance`, `diversity`) functions indexing into index2label incorrectly
- `974dc9cb` - [fix] image stats normalization bug and option to not normalize

    - fixed image hashing for images normalized to 0-1 pixel values
    - fixed `get_bitdepth` numerator sign bug
    - added parameter to `compute_stats` to skip normalizing images for stats
    - rename `dataset_index` column to `dataset_indices` column for MultiDuplicatesOutput
- `8518c1c4` - [fix] Regression in Metadata handling of classes in datasets with unobserved class labels
- `2964321b` - [fix] Ensure Metadata index2label mapping is robust
- `cf3c046a` - [fix] Bump version and update pyproject.toml authors and development status
- `a6e9f0e0` - [fix] Use integer figsize dimensions in bias tutorial notebook
- `0943a51b` - [fix] Pass dtype to ensure behavior stable on all versions of NumPy
- `af841f91` - [fix] Intermittent test failure with uninitialized memory

📝 **Miscellaneous**

- `e29313e9` - [docs] Address feedback on concept page prose
- `a4a61dbe` - [docs] Adjust findings for background object comparison
- `ab70513b` - [docs] Update binning linfoot ceiling
- `1b6c4175` - [docs] Update ROADMAP.md with longer-term vision
- `f1996b53` - [docs] Update roadmap
- `2c06676a` - [misc] Implement polish and consistency fixes
- `516e4b40` - [devsecops] Update markdownlint rules
- `7964e3d2` - [docs] Address docstring inconsistencies and broken links
- `6006940e` - [docs] Small edits to phrasing for clarity
- `5075f184` - [devsecops] Remove init script and clear up torchvision dependency
- `050463ea` - [docs] Remove mention of Conform - not shipped in 1.0
- `e95d37f9` - [docs] Add an input requirements page
- `ce9f14a7` - [docs] Add an input requirements page
- `3b042c1d` - [docs] Add migration/upgrade guide from v1.0 to v1.1
- `16d7ff36` - [docs] Shrink execution cost for tutorial
- `2bce5393` - [devsecops] Return numba cache warm script and move to test
- `540c55cf` - [docs] Add binning concept page
- `6d392d69` - [docs] Add how-to guide for detecting drift from model outputs
- `253a9dc5` - [docs] Enable more robust spellcheck, asterisk emphasis enforcement, and fix oxford commas
- `e165466a` - [docs] Tweak the grid item card ordering
- `b0db53db` - [deps] Bump maite-datasets and add typing stubs for typecheck
- `70969bb9` - [docs] Add background comparison how-to
- `b690ae10` - [devsecops] Show MR title in preview page
- `07c5298d` - [docs] Fix markdownlint errors
- `7e3402cc` - [devsecops] Fix SAST exclusion paths
- `9ad91d62` - [docs] Add metadata levels concept page
- `baf6900e` - [docs] Tweak installation notes for better pytorch instructions
- `857ca296` - [docs] Improve execution time and add more concepts to tutorial
- `3c0455c2` - [devsecops] Fix pre-release duplicate run
- `1ade5252` - [docs] Silence ipykernelapp warnings
- `3bc47ff5` - [deps] Bump onnxruntime-gpu minimum for py3.14 wheels
- `40c4aca2` - [deps] Freeze onnxruntime-gpu for cuda 12.X
- `e6663c24` - [deps] Bump dependencies
- `e4a7950f` - [docs] Fix broken xrefs and add a unit test to validate
- `18f66392` - [devsecops] Prevent accidental resurrection of orphan ipynb files in docs
- `e8697e31` - [misc] Refactor types to submodule
- `34ae329c` - [docs] Add validation and trust section to concepts page
- `6cb7c0d3` - [deps] Restrict opencv-python to <=4.12.0.88 for FIPS compatibility
- `78b3ebd3` - [devsecops] Update components to v1.2.0
- `2d24fa7c` - [test] Address code coverage gaps
- `f374426c` - [depr] Switch deprecated ClassifierUncertaintyExtractor to use new logic internally
- `885102d6` - [misc] Change default value for normalize_pixel_values
- `88a1807c` - [docs] Use layer_name feature of TorchExtractor for embedding extraction
- `3514984e` - [docs] Rename how-to for setting global defaults
- `a5f1fc6b` - [misc] Minor refactor cleanup work
- `cbd09f69` - [docs] Add more intersphinx links to scipy, sklearn and torch
- `8420c75d` - [devsecops] Skip link checks on ieeexplore.ieee.org
- `ac65b530` - [devsecops] Configure CI/CD cache key for downloaded datasets
- `4245e64d` - [misc] Fix import to prevent circular import short circuiting autoapi
- `f4e92298` - [test] Set max processes and threads to 1 to avoid oversubscription
- `5b0f7111` - [misc] Enhance changelog enumerator to include non-merge commits
- `e05b7539` - [docs] Crosslink MAITE protocol types
- `9a484d3c` - [docs] Add new how-tos for wrapping dataframe datasets and lazy-load images
- `d3dea4ea` - [misc] Add and expose MOT protocol
- `d3dde532` - [deps] Bump MAITE minimum dependency to 0.9.4 for MOT
- `b398c5b7` - [misc] Minor logging improvements
- `8cb7556a` - [deps] Bump MAITE dependency to 0.9.4
- `bb559c0f` - [docs] Add tutorial on drift detection via prediction uncertainty
- `eaa5b468` - [misc] Exclude Jupyter notebooks from ruff
- `53bcd345` - [test] Fix fragile dependency on sklearn ordering for tie-breaking
- `5d2743ef` - [docs] Fix export of symbols in root package
- `137ba4e1` - [docs] Taxonomy of Leakage
- `b9314a44` - [devsecops] Bump components branch to 1.1.0
- `a482b70d` - [docs] Add cross-linking to notebooks and instructions for local ipynb generation
- `941b9a21` - Address TR compliance issues
- `d55fb05d` - [misc] Skip sciencedirect link validation (bot blocking)
- `f7ab7bbc` - [misc] Improve typing for iter_image and unwrap_images
- `3712d222` - [devsecops] Update branching strategy markdown
- `3560c60e` - [devsecops] Bump compliance pipeline to 1.0.1
- `c367a394` - [devsecops] Add PUBLISH_VERIFICATION trigger
- `b0dd4236` - [docs] Update verification scripts
- `33e4542e` - [docs] Additional clarification for NMI and other doc fixes
- `ae5178e8` - [perf] Switch to internal nearest neighbor computation for critical value radii
- `fca2a3e7` - [docs] Address documentation inconsistencies
- `95caf1cc` - [devsecops] Ensure security jobs run on protected refs
- `f449b7cf` - [lint] Fix lints from ruff update
- `d2670847` - [devsecops] Remove CUDA 12.4
- `a9af11a5` - [type] Fix verification type check failures
- `4f6e30bc` - [devsecops] Lock poetry to 2.2.0 to match devsecops components
- `fdf486a8` - [devsecops] Update ruff rules and bump poetry dependencies
- `657c2ccb` - [devsecops] Remove harbor references and docker usage for documentation
- `014c7e2a` - [devsecops] Run compliance pipeline only on main
- `88951794` - [docs] Remove extraneous autoapisummary template
- `e7ac85cf` - Switch to python files for notebook source-of-truth
- `01726f0c` - [test] Add edge case tests for Metadata
- `6ddc30b1` - [deps] Bump maite-datasets to v0.0.12
- `d94ea95f` - [devsecops] Update changelog spacing and correct docs trigger for patch releases
- `a66fb59e` - [devsecops] Ensure tags are fetched for GitLab docs build
- `e017975f` - [devsecops] Fix release pipeline triggers and changelog bookkeeping
- `9a6af077` - [test] Fix RNG to avoid intermittent failure
- `eba1858b` - [doc] Documentation updates
- `65295535` - [devsecops] Adjust dependencies to match onnx to selected cuda
- `471e2d0f` - [devsecops] Address code coverage +skipdocsclean
- `2e6d2aed` - [devsecops] Fix test dependencies
- `811b4fa1` - [devsecops] Address semgrep findings in docs path +skipdocsclean
- `794cb0a2` - [devsecops] Add +skipdocsclean trigger to use cached docs run
- `d132c22b` - [devsecops] Update conda-forge xxhash alias
- `a3464be5` - [devsecops] Enforce matching torch and torchvision sources
- `ceeac23a` - [devsecops] Fix typecheck errors in ci components
- `32868f79` - [devsecops] Bump CUDA version support to match pytorch
- `3e6e7be6` - [devsecops] Specify explicit channel in conda environment.yml

## v1.0.0

🚀 **Major Release**

- `ed11b75a` - Add support for python 3.14

🌟 **Feature Release**

- `089072d6` - Add SIFT based BoVW extractor and implement with Duplicates

🛠️ **Improvements and Enhancements**

- `49631303` - Address all flagged issues from API review for v1.0
- `0886cd36` - [impr] Improve Duplicates and Outliers threshold capabilities
- `6d1999b2` - Improve multiprocessing implementation and stats performance
- `a54f286a` - Documentation rework
- `72e6547b` - [feat] Store duplicate and outlier outputs in dataframes
- `60e78334` - [misc] Move Prioritize to dataeval.scope
- `c143c51f` - [impr] Switch to _index suffix for source indexing
- `18cfefe6` - Refactor shift classes to utilize shared code across Drift and OOD and reduce duplication of code
- `280b3a2a` - Standardize Drift API to support chunking and cumulative tests across all detectors
- `8f1d029b` - [impr] Switch to using Threshold classes for thresholding
- `4233ae83` - Rework experimental imports and warn-once on use/instantiation
- `ee784e18` - Add framework for annotating with experimental and deprecated
- `b518d633` - Refactor encoders as extractors
- `a196e1fe` - Use dataframes for Sufficiency tabular outputs
- `2ecfa179` - Add ResetModelStrategy protocol for Sufficiency
- `4a2ef1dd` - Enhance prioritize to offer hdbscan clustering and fix for different length...

📝 **Miscellaneous**

- `7a7445db` - [docs] Update set_seed and configuration how-to
- `422108a9` - [misc] Standardize on `compute` as verb
- `8f75bcdf` - Improve logging output
- `9a9eb12b` - Expose core function result types and completeness
- `5de0d7c3` - [docs] Clean up documentation headers and structure and delete redundant how-to
- `68c7c48e` - Add augmentation duplicate detection notebook

🚧 **Deprecations and Removals**

- `f17b2270` - Mark parity and uap functions as experimental
- `239f2f43` - Rename calculate to calculate_stats

👾 **Fixes**

- `28faf1d5` - Fix split_dataset to work with OD datasets
- `4607cedc` - Exclude non-numeric values from outlier calculation
- `613a3c57` - [fix] Fallback to class label value if not present in index2label
- `fdea0473` - Fixed unwanted dtype conversion in to_numpy for int arrays

## v0.95.0

🌟 **Feature Release**

- `593f60f6` - Add optional ONNX encoder and utilities to DataEval

🛠️ **Improvements and Enhancements**

- `f8ec65b6` - Enhance duplicates with rotation/flip invariant hashing
- `bb4eb501` - Make prioritize configurable at init
- `347675c4` - Switch to using encoder protocol for encoding
- `6bf91b02` - Use pydantic for Config classes
- `ba95d5bc` - Add protocol for Metadata
- `17006ddc` - Add dhash and enhance duplicates to allow multiple reasons for flagging near duplicates
- `397d0fe1` - Unify clustering algorithms

👾 **Fixes**

- `7002c856` - New plan for computing balance metric from MI in cases where a discrete factor is present vs. cases where both factors are continuous.
- `19a77d8f` - fix missing class_label from outputs

📝 **Miscellaneous**

- `9c7515dd` - Cleanup doctest fixtures
- `52a1669b` - Feature Extractor

## v0.94.0

🌟 **Feature Release**

- `1765195b` - Adding in detection of label errors through clustering
- `7481c4c3` - Refactor evaluate into private helper methods.
- `67445937` - Release new config-based Sufficiency API
- `f37da8f1` - Add calculate_ratio function to replace boxratiostats

🚧 **Deprecations and Removals**

- `e8d6fce3` - Remove matplotlib dependency and switch to dataeval-plots package for visualization
- `ee320402` - Remove dataeval.metrics.estimators
- `8fe4ba33` - Remove dataeval.metrics.stats

🛠️ **Improvements and Enhancements**

- `d9be329b` - Overhaul OOD classes and concept page
- `938735b5` - Fix duplicates and enhance functionality
- `05fabdd2` - Add Class Balance concept page
- `711f367b` - Refactor evaluators.drift module into shift
- `0beae0e5` - Overhaul Drift classes and concept page
- `9d7789a0` - Move prioritize to be its own independent class
- `653976a2` - Add config for global batch_size
- `644c3290` - Transform mutual information rather than normalizing, for output of balance metric
- `da508d46` - Remove torchvision and ResNet18 wrapper
- `76e8de4f` - Refactor bias, performance and quality modules
- `6196502c` - Refactor data module to root
- `cb0dec38` - Switch additional output classes as polars outputs
- `faadfab2` - Update Class Balance selection to handle OD datasets
- `b4082246` - [impr] Allow object detection Metadata objects to pivot on per-image and per-target
- `a5532207` - Update bias evaluators to be stateful classes
- `3a5e875b` - Normalize parity output using Cramer's V
- `21801449` - Add logging and convert warnings in core
- `87d401c6` - Re-implementation of Completeness metric
- `5c6cca7f` - Remove `metrics` and `outputs` submodules and refactor `detectors` as `evaluators`
- `1749165c` - Normalize aspect ratio stat in dimensionstats
- `7bcc33cb` - [feat] Add selection resolution feature to Select datasets
- `1f5f702d` - [misc] Refactor and move ood metadata tools as general metadata insight functions
- `539d51ce` - [impr] Port clustering from fast_hdbscan and implement numba cache
- `b638c45c` - Update sufficiency.evaluate to use config strategies
- `0e5a4364` - Add methods to Outliers and Duplicates utilizing ClusterData for analysis
- `73340730` - Add validation for dimensionality of arrays
- `0ebc1702` - Add application-level Sufficiency dataclass and Protocols
- `cdbf5dfe` - [improvement] allow stats to be run on lower dimensional data

👾 **Fixes**

- `37362f26` - [fix] Correct the denominator for missing and zeros stats calculation
- `496c675a` - Bug fix in the MST creation algorithm
- `a6d75805` - Update fast_hdbscan to v0.2.2
- `9d739f8c` - fix bug - Drift permutation bug - p-value in drift test is either 0.0 or 1.0 and nothing in between

📝 **Miscellaneous**

- `a10f5e7f` - Update pipeline jobs to use uvx (uv tool) and switch to CUDA 12.4 for docs
- `2199d878` - Refactor utils module
- `fd4e3479` - Fix coverage and remove annotations from future
- `45fb3f35` - Remove pandas dependency
- `2a61604e` - Uncertainty-based drift detection concept page
- `8c3345f1` - Add schedulers for geometric and custom eval_at parameter
- `d49d9969` - Add how-to for using dataeval.config to set global configuration
- `af3af300` - [impr] update per_box references to per_target
- `a51817e9` - Switch to TypedDict for outputs over anonymous tuples
- `91017332` - Rework pages for user roles
- `d85115d5` - Relax typing for core functionality and standardize parameter naming
- `e48b90f8` - Add BRANCHING.md file for branching strategy
- `52d0346f` - Add grouped tables for the Algorithm summary

## v0.93.0

🚧 **Deprecations and Removals**

- `12b102e3` - Remove unused modules and duplicate/deprecated model architectures

🛠️ **Improvements and Enhancements**

- `ed01aed2` - Add ability for stats calculator to specify per_image and per_box
- `3f6116b0` - Expose dataeval core stateless functions
- `ef5f07ed` - Change embeddings to be numpy backed with memmap caching

📝 **Miscellaneous**

- `880070f7` - Split responsibility of `typing` into `protocols` and `types` for SRP
- `737453db` - Make embeddings typing a Callable to allow for generic non-torch models
- `e1a7dfd9` - Add generic typing to Array Protocol and add typing support for native python data structures

## v0.92.3

🛠️ **Improvements and Enhancements**

- `bf50c042` - Remove Embeddings from input parameters and replace with Array

📝 **Miscellaneous**

- `369fcb22` - Additional clarifications to Embeddings concept page

## v0.92.2

🛠️ **Improvements and Enhancements**

- `52b9c6da` - Added layer_name keyword to Embeddings

## v0.92.1

## v0.92.0

🚧 **Deprecations and Removals**

- `36790e59` - [depr] Remove AE Trainer and DataEvalDataset helper class

## v0.91.3

🛠️ **Improvements and Enhancements**

- `33350af3` - [deps] Add support for python 3.13
- `cd713751` - Sufficiency multiple model plots

👾 **Fixes**

- `5682073c` - [fix] Restore missing submodules for documentation

📝 **Miscellaneous**

- `e8e54816` - [misc] clean up yaml anchors
- `dc86b10f` - Remove tqdm dependency
- `2acb629d` - [misc] build docs directly on runner

## v0.91.2

👾 **Fixes**

- `a6382915` - Changed stopping criterion when stepping through kNN graph and forming clusters.

📝 **Miscellaneous**

- `56ad8657` - [misc] Remove unused code and clean up delay loaded module imports

## v0.91.1

🛠️ **Improvements and Enhancements**

- `5a00001d` - [impr] Remove torch dataset requirement from Embeddings class

👾 **Fixes**

- `696ef5bf` - New test_simple_nodes() unit test for core.mst.minimum_spanning_tree()

## v0.91.0

🚧 **Deprecations and Removals**

- `a37019ff` - Remove MAITE dataset helpers

🛠️ **Improvements and Enhancements**

- `e6a3f2d3` - added resolve_indices() and related unit tests
- `e8a32098` - Refactor stats to core module and remove dimensionstats from imagestats

👾 **Fixes**

- `dd8f1cf5` - Connect BER to MST implementation in _fast_mst.py

📝 **Miscellaneous**

- `6609e929` - Refactor core functionality to dataeval.core submodule

## v0.90.1

🛠️ **Improvements and Enhancements**

- `789fb41a` - Speedup for MST option in Divergence

👾 **Fixes**

- `a4802a52` - Issue #922 Indexing Bug Fix

📝 **Miscellaneous**

- `2a94dfa4` - Refactor NullModelMetrics for typing and encapsulation

## v0.90.0

🌟 **Feature Release**

- `7581907a` - Integrate null model metrics

📝 **Miscellaneous**

- `512a6385` - Refactor and move functions to new `functional` submodule to prepare for new API
- `0cd25665` - Minor changes to notebooks

## v0.89.1

🛠️ **Improvements and Enhancements**

- `9e0b441b` - Additional clarification to Embeddings Concept page
- `1567f818` - Sufficiency: Improve Global Minimization

📝 **Miscellaneous**

- `ecbf11bb` - Update Duplicates docstring
- `59f56fd1` - Address security vulnerabilities found in lock file

## v0.89.0

🌟 **Feature Release**

- `08984b1a` - Embeddings Concept page

🛠️ **Improvements and Enhancements**

- `356052b6` - Sufficiency: explicitly implement for operational metrics
- `3ca1b01a` - pruning concept page

📝 **Miscellaneous**

- `7ed52bd9` - Drift docstring update

## v0.88.1

🛠️ **Improvements and Enhancements**

- `b3e258d3` - Sufficiency: Adjust measure plots
- `ca707ba6` - Blended metadata exploration tools into Metadata concept page.

## v0.88.0

🚧 **Deprecations and Removals**

- `1e86ffb8` - Remove dataset builders and update min python target for the project
- `6d11a131` - Remove Python 3.9 support

🛠️ **Improvements and Enhancements**

- `fdc9b015` - 947 Sufficiency: Return data from all runs

📝 **Miscellaneous**

- `95446290` - Update docstrings to be imperative and include all methods, attributes, and properties

## v0.87.0

🌟 **Feature Release**

- `28e238ed` - New OOD_KNN detector, works with Embeddings class

🚧 **Deprecations and Removals**

- `808bf568` - Remove utility datasets from the dataeval package

🛠️ **Improvements and Enhancements**

- `75a22785` - Implement dataset validation function
- `e83b4c31` - Integrate prioritization re-ranking policies
- `14eeb978` - Switch to using is_continuous function for metadata binning

👾 **Fixes**

- `75995b42` - 412 Sufficiency inverse projection does not handle unachievable target
- `ef7da431` - Fix datasets datum metadata to be MAITE protocol compliant
- `06a5727d` - Fix balance to treat arrays of discrete distinct values as continuous

📝 **Miscellaneous**

- `bf8c7ea6` - Improved Selection docstrings

## v0.86.9

🛠️ **Improvements and Enhancements**

- `52218096` - Expanded sufficiency docstring with descriptions of parameter default values...

## v0.86.8

## v0.86.7

👾 **Fixes**

- `a93ccfa2` - Hotfix: Address metadata issues for datasets with empty targets

## v0.86.6

## v0.86.5

🛠️ **Improvements and Enhancements**

- `a5154294` - Add dataframe format to StatsOutputs

👾 **Fixes**

- `980eee6f` - Hotfix: Allow for nan values in outliers and account for them correctly
- `2a484972` - Hotfix: Labelstats do not correctly account for 0 targets on an image

## v0.86.4

👾 **Fixes**

- `1a7a6ca1` - Fix metadata and labelstats regressions

## v0.86.3

🛠️ **Improvements and Enhancements**

- `3f301870` - Remove targets and switch to DataFrames for label stats calculations

## v0.86.2

🛠️ **Improvements and Enhancements**

- `5caee32e` - Update Metadata class to use Polars DataFrame

👾 **Fixes**

- `f04da307` - MILCO has trouble reading box coordinates delimited by a variable number of spaces.

## v0.86.1

🛠️ **Improvements and Enhancements**

- `28d186c5` - Adjusting dataset loaders for uniformity and deterministic behavior
- `b0873930` - Improvements to DimensionStats and plotting
- `8ea38cfe` - Add array types for inputs to dataset helpers
- `5fe8e4de` - Fix boxratiostats calculations and add missing docstrings
- `20e8fab6` - add target size and tests
- `96fd2e23` - include empty factors

👾 **Fixes**

- `897e4e0f` - Translated MILCO box corners for MAITE compliance

## v0.86.0

🌟 **Feature Release**

- `3a6c1e8b` - Port multi-variate domain classifier from NannyML

🛠️ **Improvements and Enhancements**

- `6615745b` - Add functionality for image level factors on object detection target metadata
- `485bc051` - Make pandas a required dependency

👾 **Fixes**

- `27758e18` - Fix indexing error in subselection

## v0.85.0

🌟 **Feature Release**

- `019d011d` - Enable ClassFilter for ObjectDetectionDataset
- `9d15be15` - Add concept page for Completeness

🛠️ **Improvements and Enhancements**

- `140cec9d` - Refactor data classes (Embeddings, Images, Metadata, Targets) and function (split_dataset) to be in a first level submodule
- `9eec7e96` - Add save and load functionality to embeddings

## v0.84.1

🛠️ **Improvements and Enhancements**

- `82329adc` - Move to new MAITE/NumPy ArrayLike and loosen type restriction on Datasets
- `c31915dc` - Split ClassBalance from ClassFilter
- `48bc2426` - Update drift classes to use DataEval data structures and simplify torch utility functions

## v0.84.0

🌟 **Feature Release**

- `0c41ca26` - Add factory class method to Embeddings to create from array
- `3d585e81` - Implement completeness
- `77bba6d7` - Add caching (in-memory) to embeddings

🛠️ **Improvements and Enhancements**

- `5a2dc69e` - Change split_dataset to take in Datasets and use Metadata internally
- `e07b8297` - Add plot function to Images and change coverage plot to take Images
- `656c8166` - Add user sections to welcome page

## v0.83.0

🌟 **Feature Release**

- `22206752` - Add transforms to Select dataset class
- `fca7e907` - Add metadata_ood_mi function as find_ood_predictors

🛠️ **Improvements and Enhancements**

- `54c28111` - Adjust tagline/purpose statement to mention user effort

📝 **Miscellaneous**

- `bf415bf8` - Update datasets docstrings for completeness and move Transform type to typing module

## v0.82.1

🛠️ **Improvements and Enhancements**

- `4b6bbab8` - Intrinsic metadata using image statistics

👾 **Fixes**

- `9bae7f1a` - Perform **dict** override hack before setting other parameters
- `a9e3fde1` - HOTFIX: remove cyclical call to getattr in Select wrapper class

📝 **Miscellaneous**

- `94b31bf1` - Create a package wide configuration for random seeds
- `36746ef9` - Switch to DeviceLike for typing of torch device
- `13ff9fd0` - Better genericize output classes and add Sequence based output collection class

## v0.82.0

🌟 **Feature Release**

- `7ad22a7c` - Switch all stats classes to use dataset inputs and change `datasetstats` to `imagestats`
- `e1da1768` - Integrate meta_distribution_compare as metadata_distance

🛠️ **Improvements and Enhancements**

- `fc8b68c2` - Spike: Define which metrics comprise completeness metrics

📝 **Miscellaneous**

- `70a8e1e2` - Create lightweight dataset wrapper factory functions
- `fef938d5` - Simplify typing and add more docstrings to dataeval.typing module

## v0.81.0

🌟 **Feature Release**

- `101fea34` - Add selection feature for datasets

📝 **Miscellaneous**

- `efc90630` - Add index based selection helper
- `666e2866` - Add collate helper functions for Dataloaders
- `4e92ae90` - Restructure typing for Datasets to allow better extensibility for other data classes

## v0.80.0

🌟 **Feature Release**

- `94a10b03` - Refactor Images, Embeddings and Metadata as a stateful classes using Dataset inputs
- `353ac6bd` - Add DataProcessor class to handle extraction of images, embeddings, targets and metadata from datasets

🛠️ **Improvements and Enhancements**

- `529e5595` - Merging current WIP state of OODdetector, based on universal embeddings from sigma-optimal-VAE, along with OOD_VAE_minimal notebook.

👾 **Fixes**

- `a2a7057e` - Adjust merge to handle numpy arrays

    Adds in a numpy array check and if true returns the array as a list prior to normal processing

📝 **Miscellaneous**

- `e55cb3f2` - Use cpu as default torch device

## v0.79.0

🌟 **Feature Release**

- `841425ff` - Release Metadata OOD function most_deviated_factors

    Adds a new explanatory function using Metadata and an OODOutput

## v0.78.0

🌟 **Feature Release**

- `bff82522` - Add collate function and convert packaged datasets to MAITE protocols

  - Changes all dataset utility classes to use `MAITE` protocol formats (`MNIST`, `CIFAR10`, and `VOCDetection`)
  - Adds `collate` to aggregate (and encode) `MAITE` datasets into images/embeddings, targets, and metadata

🛠️ **Improvements and Enhancements**

- `d9e0f8b0` - Enforce embeddings on functions/methods that take embedding inputs

## v0.77.1

🛠️ **Improvements and Enhancements**

- `9a420f7d` - Update Assess the data space tutorial to fit JATIC DR-2.3
- `3ab63f3e` - Integrate clusterer speed improvements with numba

## v0.77.0

🌟 **Feature Release**

- `a1974e41` - Add global config module to control default device and max processes

👾 **Fixes**

- `c5ca814d` - Enforce unit interval in OOD detector and coverage metric
- `41c4437b` - CoverageOutput attributes renamed for clarity

    Attributes renamed:
  - `indices` -> `uncovered_indices`
  - `radii` -> `critical_value_radii`
  - `critical_value` -> `coverage_radius`
- `99631a94` - Fix ax.hist on small ranges in NumPy 2.1+

## v0.76.1

🛠️ **Improvements and Enhancements**

- `a8a4cd4f` - Remove merge from preprocess and address metadata array length inconsistencies
- `f8061eca` - Add option to return dropped keys from metadata utility functions
- `a4ddbed1` - Add pandas dependency to `all` extras option
- `5b05981e` - Expose dropped keys from nested lists and inconsistent keys in metadata merge and preprocess

📝 **Miscellaneous**

- `a20766ec` - Updates to documentation
- `961ad923` - Miscellaneous docs changes

## v0.76.0

🌟 **Feature Release**

- `4647edca` - Expose flatten metadata function and update docstring

🛠️ **Improvements and Enhancements**

- `27d34a0c` - Incorporating NAWCAD feedback to improve the documentation for the stat functions, outliers class and coverage class

📝 **Miscellaneous**

- `c9998971` - Switch themes to sphinx-immaterial, enable graphviz and restructure documentation
- `3dedde8f` - Adding templates for auto generation of docs
- `8aaa89f3` - add deep dive prototype
- `d9d902f3` - Allow for float type bounding boxes
- `9bc0f910` - Add additional code coverage tests
- `15f1ae84` - Add logging to output metadata decorator
- `01cef92b` - Split conftest for tests and doctests
- `adc8e293` - Publish MR docs and code coverage to deployment environments
- `a3fc1f6c` - Moves document link to body to match other header titles
- `69892fd3` - Visibility enhancements to BalanceOutput.plot() heatmap
- `b6ab03a6` - Simplify docker build script for docs

## v0.75.0

🌟 **Feature Release**

- `3aa12cb3` - Refactor bias metadata helpers

    Metadata preprocessing functions have been moved from `dataeval.metrics.bias.metadata_preprocessing` to `dataeval.utils.metadata`.

🛠️ **Improvements and Enhancements**

- `ed98b6b1` - Return empty string for hashes on too small images

    `pchash` now returns empty string when attempting to perform perception hashing against images or chips that are too small to meaningfully hash. `Duplicates` also ignore empty perception hashes to avoid false positive detections.
- `b144fa1c` - Change torch to be required dependency

    PyTorch is now a required dependency and the `torch` extra is no longer required for full functionality

📝 **Miscellaneous**

- `6e4474b2` - Refactor utils and fix associated docstrings, documentation and notebooks
- `ff87cee6` - Update documentation and CI pipelines to comply with SDP DR-3
- `aa7d9205` - Updated README.md format, added tagline and cdao funding acknowledgment
- `82559846` - Replace manual markdown files with autoapi generated rst files

## v0.74.2

📝 **Miscellaneous**

- `e7a284de` - Update dataset split unit tests
- `f8731a44` - Add initial logging framework and unit test
- `771dc1d1` - Add conda tests to pipeline
- `2d9fd55a` - Update RTD yaml to use uv for installation
- `0ab99a7f` - Initial prototyping of underspecification tests

## v0.74.1

📝 **Miscellaneous**

- `102664de` - Remove tensorflow from project
- `e782dad1` - Refactor OutputMetadata and clean up set_metadata decorator
- `80aae3a6` - Just use KSOutput as a MappingOutput instance instead of extracting the dict attribute it no longer has.
- `b738e01f` - Allow docker cmds within dev container
- `16839b46` - Add MappingOutput class
- `e2cfda94` - Made metadata_tools/ks_compare compatible with new KSOutput class.

## v0.74.0

🌟 **Feature Release**

- `73c1e1be` - Implement PyTorch AutoEncoder based OOD detector

    Adds initial PyTorch based Autoencoder OOD detector available when installed with the `torch` extra.

🛠️ **Improvements and Enhancements**

- `70794b5f` - Moved discretization of metadata out of bias functions

📝 **Miscellaneous**

- `4d94e602` - Added test assertions for how_to notebooks
- `7723e242` - Introduce Pytorch OOD detector, with its new training procedure, into OOD howto notebook.
- `f5ac4bdd` - Added new KSOutput class and adapted tests and other functions accordingly
- `3a01a81a` - Introduce new Pytorch OOD detection into prototype metadata demo notebooks.
- `dc155554` - Fix torch gmm functions and enable tests
- `a715c1ef` - Adjust docs to incorporate new metadata function
- `0719bad0` - Update dependencies to remove hdbscan

## v0.73.1

👾 **Fixes**

- `cac3e2b8` - Fixes drift with pre-processing and shuffles MNIST by default

📝 **Miscellaneous**

- `bacbd0e7` - Use build script specifically for docs
- `0a87e912` - docker build for docs only
- `671b60a5` - Prototype function to infer whether a 1D sample is continuous or discrete
- `d0b8004a` - Use explicit re namespace for compile, search, sub, and MULTILINE
- `502ca2df` - Change to nox for automation test scripts
- `5b46ebea` - Add new bias functional tests and set groundwork for rediscretization

## v0.73.0

🌟 **Feature Release**

- `e055acf0` - Metadata utility function to merge, extend and flatten metadata
- `95b28ae1` - Adjust bias plotting functions to return figure

📝 **Miscellaneous**

- `532f92a2` - Minimum spanning tree and Clusterer are rewritten using numba for large code speed up
- `7377e012` - Switch jobs to use uv and tox natively
- `7af75016` - Add lazyloading for tensorflow modules

## v0.72.2

🛠️ **Improvements and Enhancements**

- `ba52ef2e` - Refactor away _internal module

📝 **Miscellaneous**

- `6e55451c` - Integration of distribution compare and OOD MI metadata tools (continued)
- `e4f82173` - Streamlined tests
- `ac8fe3ee` - Fix type mismatch on training AEGMM
- `6289c7d0` - Add plotting helper functions to diversity and balance
- `14d0cfd4` - Integration of low-level metadata drift/OOD exploration functions

## v0.72.1

📝 **Miscellaneous**

- `32ba1f29` - Data split tests
- `76f73770` - Updated glossary and other files to use new style of links
- `20efd27e` - Add support for Python 3.12

## v0.72.0

🌟 **Feature Release**

- `14ef382c` - Update dependencies for conda compatibility

## v0.71.1

🛠️ **Improvements and Enhancements**

- `97849b01` - Update support for tensorflow >=2.16 with explicit keras v2

📝 **Miscellaneous**

- `85bafa30` - Swap brightness and darkness
- `96a30ad0` - Make optional checks more granular
- `55ca81d6` - Use native int for dict keys for Outliers
- `639e140b` - silence warnings for docs and doctest

## v0.71.0

🌟 **Feature Release**

- `cdae8a17` - Parallelize existing stats metric functions and introduce dedicated channelstats function

    Running statistical analysis functions take significant time against large datasets.  Due to the natural parallelism of analyzing individual images, we introduced parallel processing leveraging the `multiprocessing` library to accelerate processing times.

    Affected functions:

  - `datasetstats`
  - `dimensionstats`
  - `hashstats`
  - `pixelstats`
  - `visualstats`

    Additionally, `channelstats` was added which performs the functionality of `datasetstats` but only for the functions that support per-channel stat calculation, `pixelstats` and `visualstats`.

📝 **Miscellaneous**

- `552668a0` - Update EDA part 1 tutorial with miscellaneous changes

## v0.70.1

🛠️ **Improvements and Enhancements**

- `d1cdcda5` - API changes with supporting documentation updates

📝 **Miscellaneous**

- `5ecd4d3a` - expose datasets API
- `6c19bba7` - Make sufficiency args more permissive
- `1bc2d067` - Improving MNIST class
- `d23b3461` - Extract small-scope reusable functions from tools made for prototype Associate[Drift|OOD]withMetadataTutorial notebooks.
- `5bea9512` - remove tf-iogcs-fs

## v0.70.0

🌟 **Feature Release**

- `71e7ff06` - Integrate labelstats function
- `f40bf0e4` - Redesign stats functions for expansion to per-box, per-channel, and boxratiostats

🛠️ **Improvements and Enhancements**

- `72390edc` - Change input format of balance and diversity to be the same as parity

👾 **Fixes**

- `f598c46a` - Update pytorch to 2.2.0+

📝 **Miscellaneous**

- `b8f0d502` - Create copy on `to_numpy` by default
- `04a71337` - Fix CI docs job to load on build
- `9286f5e8` - Skip or rework MNIST based unit tests
- `704f44e3` - Investigate the use of metadata to help explain observed dataset drifts and OOD examples
- `e25f84f3` - Expose SufficiencyOutput and move class methods to output class
- `742a084c` - Adding algorithm compatability/requirements table
- `7ce85be7` - Misc concept documentation

## v0.69.4

📝 **Miscellaneous**

- `7bca6ed4` - Unified all MNIST and MNIST corrupt datasets to a single internal MNIST class
- `66ad1c92` - new drift detector: multivariate domain classifier

## v0.69.3

📝 **Miscellaneous**

- `6745e39d` - Document: Class Label Statistical Independence and Coverage Documentation
- `1f7689ac` - Adding bias tutorial (parity-balance-diversity)

## v0.69.2

📝 **Miscellaneous**

- `f7d5bac3` - Adds stats for bounding boxes
- `18be58a3` - Adding label stats
- `809d1d7a` - Always produce p-val and distance metrics for drift
- `5cd7c205` - Improving imagestats and channelstats functions
- `b379d44c` - Add dataset splitting features
- `80b68a73` - Use regex to replace markdown links
- `1d99455a` - Tag LKG at the correct commit SHA
- `ad0e368b` - Always run tasks

## v0.69.1

📝 **Miscellaneous**

- `d9068a2c` - Fix release and changelog script

## v0.69.0

📝 **Miscellaneous**

- `63ab70d7` - Remove automatic update of documentation notebooks

## v0.68.0

🌟 **Feature Release**

- `47b48e14` - Allow Duplicates and Outliers detectors to take in multiple StatsOutput objects

📝 **Miscellaneous**

- `65d8f3de` - Combine classwise bias metric outputs with non-classwise
- `ccfd72ef` - Adding clustering/coverage tutorial
- `6d09d710` - Add CONTRIBUTING.md
- `72387d9c` - Updated version replacement script to include cache files
- `5285f01b` - Prototype Performance  Estimation
- `3ae16116` - concept pages for balance and diversity, rescale Simpson diversity
- `3e16a905` - Switching documentation themes to the pydata theme

## v0.67.0

🌟 **Feature Release**

- `a0b04800` - Refactor DataEval functions and classes and update documentation

  - Changes DataEval functions and classes to be more hierarchical in modules:
    - detectors
      - drift (DriftCVS, DriftKS, DriftMMD, DriftUncertainty)
      - linters (Clusterer, Duplicates, Outliers)
      - ood (OOD_AE, OOD_AEGMM, OOD_LLR, OOD_VAE, OOD_VAEGMM)
    - flags (ImageStat)
    - metrics
      - bias (balance, coverage, diversity, parity)
      - estimators (ber, divergence, uap)
      - stats (imagestats, channelstats)
    - workflows (Sufficiency)
  - Backends have been moved from `models` to `tensorflow` and `torch`
  - Renamed following classes:
    - `Linter` -> `Outliers`
    - `parity` -> `label_parity`
    - `parity_metadata` -> `parity`
    - `DriftOutput` -> `DriftBaseOutput`
    - `DriftUnivariateOutput` -> `DriftOutput`
  - Miscellaneous fixes:
    - Documentation updated
    - Streamlined optional import checks in the `__init__.py` tree
    - Fixed misspelling in glossary

👾 **Fixes**

- `84aae760` - balance test cleanup

📝 **Miscellaneous**

- `6d09d710` - Add CONTRIBUTING.md
- `72387d9c` - Updated version replacement script to include cache files
- `5285f01b` - Prototype Performance  Estimation
- `3ae16116` - concept pages for balance and diversity, rescale Simpson diversity
- `3e16a905` - Switching documentation themes to the pydata theme
- `d50d9cd1` - Update Landing Page
- `2fd7fa59` - Author drift detection tutorial
- `49b5af42` - Use uv instead of pyenv for python deployment
- `0f6eb6b0` - Pin notebooks on release to specific version
- `4f101a4e` - Adjust imagestats and channelstats reference guides to new format
- `0ee82ede` - Only build data image in main pipeline
- `7b84ceb5` - Improve test coverage
- `d3c5258a` - Add StatsOutput as input type for linter and duplicates
- `cf73393a` - Updates drift reference guides and concept page
- `4ce5cdf7` - Adjust model reference guides to new format
- `17195a2b` - Adjust parity reference guides to new format
- `e9761b4d` - Adjust out of distribution reference guides to new format
- `eaf707a7` - Adjust uap reference guide to new format
- `335ac3be` - Adjust sufficiency reference guide to new format
- `3a866f01` - Change Optional[Type] to Type | None per 3.10+ standards

## v0.66.0

🌟 **Feature Release**

- `a0b04800` - Refactor DataEval functions and classes and update documentation

  - Changes DataEval functions and classes to be more hierarchical in modules:
    - detectors
      - drift (DriftCVS, DriftKS, DriftMMD, DriftUncertainty)
      - linters (Clusterer, Duplicates, Outliers)
      - ood (OOD_AE, OOD_AEGMM, OOD_LLR, OOD_VAE, OOD_VAEGMM)
    - flags (ImageStat)
    - metrics
      - bias (balance, coverage, diversity, parity)
      - estimators (ber, divergence, uap)
      - stats (imagestats, channelstats)
    - workflows (Sufficiency)
  - Backends have been moved from `models` to `tensorflow` and `torch`
  - Renamed following classes:
    - `Linter` -> `Outliers`
    - `parity` -> `label_parity`
    - `parity_metadata` -> `parity`
    - `DriftOutput` -> `DriftBaseOutput`
    - `DriftUnivariateOutput` -> `DriftOutput`
  - Miscellaneous fixes:
    - Documentation updated
    - Streamlined optional import checks in the `__init__.py` tree
    - Fixed misspelling in glossary

🛠️ **Improvements and Enhancements**

- `5f730baa` - Refactor ImageStats and ChannelStats as metric functions

👾 **Fixes**

- `84aae760` - balance test cleanup
- `3ebd278c` - handle float-type categorical variables in balance metric
- `066b7153` - Fixes modzscore to account for division by 0

📝 **Miscellaneous**

- `d50d9cd1` - Update Landing Page
- `2fd7fa59` - Author drift detection tutorial
- `49b5af42` - Use uv instead of pyenv for python deployment
- `0f6eb6b0` - Pin notebooks on release to specific version
- `4f101a4e` - Adjust imagestats and channelstats reference guides to new format
- `0ee82ede` - Only build data image in main pipeline
- `7b84ceb5` - Improve test coverage
- `d3c5258a` - Add StatsOutput as input type for linter and duplicates
- `cf73393a` - Updates drift reference guides and concept page
- `4ce5cdf7` - Adjust model reference guides to new format
- `17195a2b` - Adjust parity reference guides to new format
- `e9761b4d` - Adjust out of distribution reference guides to new format
- `eaf707a7` - Adjust uap reference guide to new format
- `335ac3be` - Adjust sufficiency reference guide to new format
- `3a866f01` - Change Optional[Type] to Type | None per 3.10+ standards
- `fe1e292d` - Use output dataclass with metadata
- `b3f6a027` - Unify handling of image reshaping

## v0.65.0

🛠️ **Improvements and Enhancements**

- `5f730baa` - Refactor ImageStats and ChannelStats as metric functions

👾 **Fixes**

- `3ebd278c` - handle float-type categorical variables in balance metric
- `066b7153` - Fixes modzscore to account for division by 0

📝 **Miscellaneous**

- `fe1e292d` - Use output dataclass with metadata
- `b3f6a027` - Unify handling of image reshaping

## v0.64.0

🌟 **Feature Release**

- `bea0446c` - Torch Dataset Reader

🛠️ **Improvements and Enhancements**

- `eda88822` - Refactor metrics

📝 **Miscellaneous**

- `a4b8e919` - Created new documentation issue templates
- `1028d082` - Remove is_arraylike function
- `dbcecec6` - Refactored read_dataset to handle common dataset returns
- `61b1f854` - Updated Workflow Landing Page
- `cf96c7f2` - Run doctest in CI pipeline
- `ecfcf89b` - Adjusted notebooks to work on google colab and added environment requirements
- `5f863782` - Update remaining metric output to NamedTuple
- `e58f4dba` - Add metadata parity documentation
- `6319a1d4` - Adding Duplicates concept
- `787545f5` - Adding ImageStats and ChannelStats concept document
- `7826405c` - Update Data Cleaning concept
- `50047116` - Change to Semantic Versioning
- `9e43399c` - Bayes Error Rate - explanation documentation
- `266ad738` - Updated BER docstrings with NDArray, shapes, and examples

## v0.63.0

🛠️ **Improvements and Enhancements**

- `3225cf18` - Convert remaining metrics and detectors to ArrayLike
- `5d88b82a` - Add Torch and Tensorflow interop through ArrayLike protocol and to_numpy converter
- `d3342275` - Refactor linter and duplicates to call evaluate with data
- `65d5aaa8` - Refactor metrics to call evaluate with data

## v0.61.0

🛠️ **Improvements and Enhancements**

- `cd59debb` - Release DataEval v0.61.0!

    DAML is now officially rebranded as DataEval!  New name, same great camel flavor.

## v0.56.0

🌟 **Feature Release**

- `64416675` - Update clusterer class and documentation

  - `Clusterer` detector released

    This class assists in exploratory data analysis of unlabeled data by identifying duplicates and outliers. Additional information on usage is available in our documentation.

## v0.55.0

🌟 **Feature Release**

- `278b4dc1` - Release Linter, Duplicates, ImageStats, ChannelStats and Parity

    `Linter`, `Duplicates` detectors and `ImageStats`, `ChannelStats`, and `Parity` metrics are now released. The existing metrics available have also been moved into different modules (`detectors` and `workflows`) that better reflect their functionality.

  - `detectors`
    - Drift detectors: `DriftCVM`, `DriftKS`, `DriftMMD`, `DriftUncertainty` and supporting classes
    - Out-of-distribution detectors: `OOD_AE`, `OOD_AEGMM`, `OOD_LLR`, `OOD_VAE`, `OOD_VAEGMM` and supporting classes
    - `Linter`
    - `Duplicates`
  - `metrics`
    - `BER`
    - `Divergence`
    - `Parity`
    - `ImageStats`
    - `ChannelStats`
    - `UAP`
  - `workflows`
    - `Sufficiency`

## v0.54.0

🛠️ **Improvements and Enhancements**

- `58263ac7` - Move niter param to evaluate and calculate and retain curve coefficients in output dictionary

    This change enhances the output of the `Sufficiency` metric to provide the coefficients for the learning curve by measure/class when running the metric. These parameters were previously recalculated each call to project and plot. The parameters are provided as a `Dict[str, np.ndarray]` under the `_CURVE_PARAMS_` key in the output dictionary.

## v0.53.0

🌟 **Feature Release**

- `322fc830` - Add parameter `k` to BER estimator for KNN to enable `k>1` for better consistency with ground truth in certain cases

## v0.52.0

🛠️ **Improvements and Enhancements**

- `07b12ac2` - Fully integrate outlier detection

    Outlier Detection API has been changed.  Additional details are available in our documentation.

## v0.51.0

🌟 **Feature Release**

- `2ed88a07` - Implement Drift Detection Metrics

    This change adds 4 types of Drift Detection metrics which allow for the detection of potential drift in the dataset.

  - Kolmogorov-Smirnov
  - Cramér-von Mises
  - Maximum Mean Discrepancy
  - Classifier Uncertainty

    The conceptual source is derived from [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953) and the implementation is derived from [Alibi-Detect v0.11.4](https://github.com/SeldonIO/alibi-detect/tree/v0.11.4).

## v0.45.0

🚧 **Deprecations and Removals**

- `5cc48bec` - Divergence metric naming corrected to HP Divergence

    Divergence metric output now returns a dictionary of `{ "divergence": float, "error": int }` instead of `{ "dpdivergence": float, "error": int }`.  Code, documentation and tutorials have been updated to the correct nomenclature of HP (Henze-Penrose) divergence.

## v0.44.6

🌟 **Feature Release**

- `41b20d3a` - Add rules for release label pipeline workflow and merge request release template

🛠️ **Improvements and Enhancements**

- `7ee53c9c` - Update Divergence default to MST

## v0.44.2

🛠️ **Improvements and Enhancements**

- `1468aa5c` - Switch to markdown and updated docs

## v0.43.0

🛠️ **Improvements and Enhancements**

- `670a0db5` - Add support for classwise Sufficiency metrics
- `b96ee099` - Have sufficiency train and eval functions take indices and batch size instead of a DataLoader

## v0.42.2

🛠️ **Improvements and Enhancements**

- `5225c491` - Change output classes to dictionaries
- `45040682` - Make Sufficiency a stateful class and revise SufficiencyOutput
- `7c5fdcff` - Pass method as a parameter to determine metric algorithm to use
- `2e883f6d` - Add better optimizer to find global minimum
- `c3c78680` - Expose AETrainer to public API to use model multiple times after training

👾 **Fixes**

- `93564b95` - Updating pyproject.toml and lock file to set dependency less than numpy 2.0

## v0.42.0

🛠️ **Improvements and Enhancements**

- `601cfae8` - Sufficiency Plotting of Multiple Metrics during one run
- `3d68a6f1` - Add parameter to plot function for optional file output

🚧 **Deprecations and Removals**

- `a6ce3e72` - Remove UAP_MST metric

## v0.40.2

🛠️ **Improvements and Enhancements**

- `f3eddaed` - Flavor 2 - Remove models from metrics entirely

## v0.40.1

🚧 **Deprecations and Removals**

- `db888bb7` - Remove usage of DamlDataset for ARiA metrics

## v0.38.1

🛠️ **Improvements and Enhancements**

- `42617f43` - Enable GPU functionality in pytorch features

## v0.38.0

🌟 **Feature Release**

- `c9b5116e` - ARiA Autoencoder as PyTorch Model

🛠️ **Improvements and Enhancements**

- `8fe97232` - Add export_model functionality and improve test coverage
- `42cc77ea` - Add empirical upper bound to UAP metric output

👾 **Fixes**

- `636dfdaf` - update project with **version** metadata

## v0.36.1

🌟 **Feature Release**

- `7d1a599f` - Implement the uap class

## v0.36.0

🛠️ **Improvements and Enhancements**

- `0799523b` - Object detection model training

## v0.29.0

🌟 **Feature Release**

- `166df3b0` - Implement Dataset Sufficiency Metric

🛠️ **Improvements and Enhancements**

- `5c4e6e06` - Use convolutional autoencoder for BER and Divergence metrics

👾 **Fixes**

- `c78e5502` - Sufficiency typecheck bugfix

## v0.28.5

🛠️ **Improvements and Enhancements**

- `9d1c354c` - Add fit_dataset, format_dataset to DpDivergence & BER

## v0.28.4

👾 **Fixes**

- `c39e009e` - Fix typecheck issues found with pyright-1.1.333

## v0.26.13

🌟 **Feature Release**

- `949e09bd` - Add kNN BER implementation

## v0.26.10

🛠️ **Improvements and Enhancements**

- `dab0a8ff` - Handle MST edge cases

## v0.26.4

🛠️ **Improvements and Enhancements**

- `bf31996f` - BER lower bound capability

## v0.25.11

🛠️ **Improvements and Enhancements**

- `dfe0bddb` - Add support for python 3.11

## v0.25.4

🛠️ **Improvements and Enhancements**

- `2ca285cc` - update BER metric to return a dataclass instead of dict

## v0.25.3

👾 **Fixes**

- `67f08b27` - Fix: Alibi-detect-models-have-fixed-architecture-shapes

## v0.25.2

🛠️ **Improvements and Enhancements**

- `db4adaff` - 69 convert metric output dictionary to dataclass

## v0.24.8

🌟 **Feature Release**

- `79614577` - Implement Multiclass MST version of BER

## v0.24.6

🌟 **Feature Release**

- `2ad9fed5` - Implement BER estimate

## v0.23.1

🌟 **Feature Release**

- `99d2fd22` - Implement outlier detection metrics using the alibi-detect VAE method

## v0.23.0

🌟 **Feature Release**

- `85eb2c1f` - Implement outlier detection metrics using the alibi-detect auto-encoder method
