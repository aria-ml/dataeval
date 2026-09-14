<!--
Release Merge Request Template

This template is to be used when a change will affect the functionality of DataEval.
This includes changes that add new features, fix existing features, update or
enhance existing features, or remove features that are no longer supported.

On merge, the title of the merge request becomes the commit subject and the changelog
entry for this change, so ensure the title is clear and concise.

The title MUST start with a [type] prefix, for example "[fix] Stop Parity rejecting
independence on replication alone". CI rejects titles without a known prefix. The
prefix selects the changelog section and the version bump:

  [major]                       breaking change    -> major release
  [feat]                        new functionality  -> minor release
  [depr]                        deprecate/remove   -> minor release
  [impr] [perf]                 enhancement        -> minor release
  [fix]                         bug fix            -> patch release
  [docs] [test] [deps] [type]   housekeeping       -> no bump
  [devops] [devsecops] [lint] [misc]

See BRANCHING.md#commit-prefixes for guidance on choosing one.
-->

## Definition of Done

- [ ] Functionality verified
- [ ] Test cases reviewed
- [ ] Documentation reviewed
- [ ] Title starts with the correct `[type]` prefix and reads well as a changelog entry
