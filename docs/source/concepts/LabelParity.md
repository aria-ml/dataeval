# Label Parity

## What is label parity?

{term}`Label parity<Label Parity>` is a means for assessing equivalence in
label frequency between datasets. This assessment helps a user understand
labels as a source of potential {term}`bias<Bias>`. Label parity informs the
user if the distribution of labels is different.

## Why is label statistical independence important?

Label frequency shift can be a very simple source of performance degradation in
operational data.

## What can be done with the label parity information?

If labels are approximately equivalent in frequency across datasets, then we
can assume that a source of {term}`diift<Drift>`/{term}`bias<Bias>` in
operational data is not {term}`label shift<Label Shift>`. If we do find bias,
then some rebalancing of the training data or retraining including operational
data may need to take place.

## See Also

- [Chi-Squared Test](https://en.wikipedia.org/wiki/Chi-squared_test)
- [Parity](Parity.md)
