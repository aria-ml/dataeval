<!--
Documentation Reference Page Issue Template

This template is to be used when a new feature has been completed. \
All code that will be released should have a refenece page in the docs/reference folder.
All new features should have at least a tutorial, concept page, and reference page. How-tos are optional.


The reference page should follow the following format:
<Template>
# Feature Name

```{testsetup}
import necessary modules
from dataeval.<type> import <feature>

create_variables = arbitrary variable code
```

```{eval-rst}
.. autoclass:: dataeval.<type>.<feature>
   :members:
   :inherited-members:
``` 
<End Template>

There should be **NO** text on this page, just code setup and the autoclass directive!!

If you create a varaible in the code setup, then add a text explanation in the docstring about the format, shape, content, etc.
-->

## Definition of Done

- [ ] File in the docs/reference folder
- [ ] Example(s) (at least 1 needed) included in docstring
- [ ] Passes doctest - `tox -e doctest` <!-- has been run in terminal before pushing to gitlab -->

/label ~ARiA ~issue::docs