# Algorithm Summary

The following tables summarize the advised use cases and technical
requirements for the algorithms provided by the DataEval library.
Each algorithm targets different types of data or problem domains.
Refer to the method-specific pages for more detailed information.

## DataEval Algorithms

|Algorithm|Description|Image Classification|Object Detection|Unsupervised|
|:---|:---|:---:|:---:|:---:|
|[Balance](Balance.md)|Assesses the metadata distribution across classes|✔|✔||
|[BER](BER.md)|Determines feasibility by estimating the error rate|✔|||
|[Clusterer](Clustering.md)|Groups data to detect outliers and duplicates|✔|✔|✔|
|[Coverage](Coverage.md)|Measures how well the dataset covers the input space|✔|✔|✔|
|[Divergence](Divergence.md)|Detects differences between dataset distributions|✔|✔||
|[Diversity](Diversity.md)|Assesses the spread of metadata factors|✔|✔||
|[Drift](Drift.md)|Detects data distribution shifts from training data|✔|✔||
|[Duplicates](DataCleaning.md#duplicate-detection)|Identifies duplicate data entries|✔|✔|✔|
|[Label Parity](LabelParity.md)|Detects differences between label distributions|✔|✔||
|[Out-of-Distribution](OOD.md)|Detects data points that fall outside training distribution|✔|✔||
|[Outliers](Outliers.md)|Identifies anomalous data points based on deviations from mean|✔|✔|✔|
|[Parity](Parity.md)|Detects differences between metadata distributions|✔|✔||
|[Stats](Stats.md)|Computes statistical summaries of datasets|✔|✔|✔|
|[Sufficiency](Sufficiency.md)|Determines data needs for performance standards|✔|✔||
|[UAP](UAP.md)|Determines feasibility by estimating upper bound on average precision||✔||

## Algorithm Requirements

A red checkmark means the algorithm accepts multiple data types.

|Algorithm|Images|Labels|Bounding Boxes|Metadata|Scores|
|:---|:---:|:---:|:---:|:---:|:---:|
|[Balance](Balance.md)||✔||✔||
|[BER](BER.md)|✔|✔||||
|[Clusterer](Clustering.md)|✔|||||
|[Coverage](Coverage.md)|✔|||||
|[Divergence](Divergence.md)|✔|||||
|[Diversity](Diversity.md)||✔||✔||
|[Drift](Drift.md)|✔|||||
|[Duplicates](DataCleaning.md#duplicate-detection)|✔|||||
|[Label Parity](LabelParity.md)||✔||||
|[Out-of-Distribution](OOD.md)|✔|||||
|[Outliers](Outliers.md)|[✔]{.red-text}||[✔]{.red-text}|||
|[Parity](Parity.md)||✔||✔||
|[Stats](Stats.md)|[✔]{.red-text}|[✔]{.red-text}|[✔]{.red-text}|[✔]{.red-text}||
|[Sufficiency](Sufficiency.md)|✔|✔||||
|[UAP](UAP.md)||✔|||✔|
