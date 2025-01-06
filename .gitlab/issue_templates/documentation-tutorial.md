<!--
Documentation Tutorial Issue Template

This template is to be used when a new feature has been completed. \
All code that will be released should be apart of a tutorial in the docs/tutorial folder.
All new features should have at least a tutorial, concept page, and reference page. How-tos are optional.


Tutorials should be used to **teach** a user a new skill.
Multiple features can be covered by a tutorial.

Currently we have the following tutorials:
- Data Cleaning
- Assessing Data Manifold
- Identifying Bias
- Monitoring
- Assessing Performance with Dataset

Please look through the tutorials and determine if your feature fits within an already existing tutorial or if a new tutorial is needed.

-->

## What is being learned

- [ ] What is the skill <!-- Note: Tutorials are not for explaining specific functions (code) - Explaining functions (code) is the docstring -->
- [ ] Who is this for?
- [ ] DataEval metrics, detectors, workflows included

## Definition of Done

- [ ] Informative Title <!-- user should have a good idea of what they are going to do in the tutorial from the title -->
- [ ] File in the docs/tutorial folder

Tutorial Headings
- [ ] What you'll do (Optional)
- [ ] What you'll learn
- [ ] What you'll need
- [ ] On your own (Optional)
- [ ] In practice... <!-- The fine print that you avoided can go in this section. -->
- [ ] Further Reading <!-- Link to explanations or external documents -->

Body of Tutorial
- [ ] Code does not error
- [ ] All links work properly
- [ ] Run docs - `nox -e docs` <!-- has been run in terminal before pushing to gitlab -->

/label ~ARiA ~issue::docs