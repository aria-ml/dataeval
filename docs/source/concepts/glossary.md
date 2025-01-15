# Glossary

## A

<!-- markdownlint-disable MD013 -->
```{glossary}
Accuracy
    A metric for evaluating {term}`classification<Classification>` models based on the fraction of predictions our model got correct. Mathematically, accuracy has the following definition:
    - $Accuracy =
    \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}$

    For {term}`Binary Classification`, it is defined as the following:
    - $Accuracy =
    \frac{(TP + TN)}{(TP + TN + FP + FN)}$ \
    where:

    - TP = {term}`True Positive Rate (TP)`
    - TN = {term}`True Negative Rate (TN)`
    - FP = {term}`False Positive Rate (FP)`
    - FN = {term}`False Negative Rate (FN)`

    A binary example with benign and malignant tumor detection rates is shown in the following image:

    ![binary accuracy example](./images/binary_accuracy_example.png)

AUROC
    The Area Under the ROC Curve (AUROC) is a metric that measures the performance of a {term}`classification<Classification>` model at all possible classification thresholds. It's calculated by measuring the two-dimensional area underneath a ROC curve from (0,0) to (1,1). AUROC can range from 0 to 1, with higher values indicating better performance:

    An example scale might be:

    - 0: A perfectly inaccurate test
    - 0.1-0.4: Unacceptable. Inaccurate a majority of the time
    - 0.5: A random model or no discrimination
    - 0.6: Unacceptable. Low discrimination
    - 0.7–0.8: Acceptable
    - 0.8–0.9: Excellent
    - 1: A perfect model that can correctly distinguish between all positive and negative class points

Artificial Intelligence (AI)
    Artificial Intelligence, or AI, is technology that enables computers and machines to simulate human intelligence and problem-solving capabilities. It is modeled after the decision-making processes of the human brain that can ‘learn’ from available data and make increasingly more accurate classifications or predictions over time. For the applications in {term}`DataEval`, Neural Networks are the main modeling method.

    See {term}`Neural Network`

Aspect Ratio
    For Images, the ratio of the width (in pixels) over the height (in pixels)
    - $Aspect Ratio = \frac{width}{height}$

    See {term}`Image Size`

Autoencoder
    An autoencoder is a type of artificial {term}`neural network<Neural Network>` that learns efficient encodings of unlabeled data by doing {term}`unsupervised learning<Unsupervised Learning>`. An autoencoder learns two functions: an encoding function that transforms the input data into a {term}`latent space<Latent Space>`, and a decoding function that recreates the input data from the encoded representation. Typically used for {term}`dimensionality reduction<Dimensionality Reduction>`.

Average Pooling
    A type of {term}`pooling layer<Pooling Layer>` that calculates the average value from a group of pixel values produced by a {term}`convolutional layer<Convolutional Layer>`. Typically used in a {term}`convolutional neural network<Convolutional Neural Network (CNN)>` to reduce the dimensionality between layers.
```

## B

```{glossary}
Balance
    A measure of co-occurrence of metadata factors with class labels. Metadata factors that spuriously correlate with individual classes may allow a model to learn shortcut relationships rather than the salient properties of each class.

Bayes Error Rate (BER)
    In statistical classification, bayes error rate is the lowest possible error rate for any classifier of a random outcome (into, for example, one of two categories) and is analogous to the {term}`irreducible error<Irreducible Error>`. A number of approaches to the estimation of the bayes error rate exist. In general, it is impossible to compute the exact value of the bayes error.

Bias
    The systematic error or deviation in a model's predictions from the actual outcomes. Bias can arise from various sources, such as a skewed or imbalanced dataset, incomplete feature representation, or the use of biased algorithms.

Binary Classification
    Binary classification is a fundamental task in {term}`machine learning<Machine Learning (ML)>`, where the goal is to categorize data into one of two classes or categories.

Black-box Shift Estimation (BBSE)
    A method for measuring {term}`label shift<Label Shift>` on machine learning datasets. It is calculated using a {term}`confusion matrix<Confusion Matrix>` and only requires that the matrix be invertible. It calculates the probability of a label *l* in the target dataset over the probability of the same label in the test data set. $(W = \frac{q(y)}{p(y)})$  It is solved as a linear equation $Ax = b$ where *A* is the Confusion Matrix estimated on the training dataset and *b* is average output of the predictor function calculated on the target dataset.

Blur
    For Images, the loss of sharpness resulting from motion of the subject or the camera during exposure. Objects appear less clear or distinct. It is calculated using the {term}`variance<Variance>` on a {term}`Laplacian filter<Laplacian Filter>`. A low value on the variance indicates very few sharp edges which points toward a blurry image.

Bonferroni Correction
    The Bonferroni Correction is a multiple-comparison correction used when several dependent or independent statistical tests are being performed simultaneously. The reason is that while a given alpha value may be appropriate for each individual comparison, it is not appropriate for the set of all comparisons.

Brightness
    For images, brightness is a measure of how light or dark an image is overall, or its luminous intensity after it's been digitized or acquired by a camera. It can also be described as the relative intensity of a visible light source's energy output.
```

## C

```{glossary}
Categorical Variable
    In statistics, a categorical variable (also called qualitative variable) is a variable that can take on one of a limited, and usually fixed, number of possible values, assigning each individual or other unit of observation to a particular group or nominal category on the basis of some qualitative property.

Channel (Images)
    Descriptive component of an image. For example, gray-scale images have one channel describing {term}`brightness<Brightness>` levels, while red-green-blue (RGB) images have 3 channels describing the red, green, blue brightness levels.

Chi-Square Test of Independence
    The Chi-Square Test of Independence determines whether there is an association between Categorical Variables (i.e., whether the variables are independent or related). For more information on how to compute see the following explanation: [*Chi Square Test Of Independence*](https://libguides.library.kent.edu/SPSS/ChiSquare)

Classification
    Classification is a supervised machine learning method where the model tries to predict the correct label of a given input data value. For instance, an algorithm can learn to predict whether a given email is spam or ham (no spam) or whether an image contains a house or a car.

Cluster Analysis
    Cluster Analysis is a statistical method for processing data. It is primarily used in {term}`classification<Classification>` projects. It works by organizing items into groups, or clusters, based upon how closely associated they are. The objective of cluster analysis is to find similar groups of subjects, where the “similarity” between each pair of subjects represents a unique characteristic of the group vs. the larger population/sample. It is an {term}`unsupervised learning<Unsupervised Learning>` algorithm, meaning the number of clusters is unknown before running the model.

Concept Drift
    Concept drift is a specific kind of data {term}`drift<Drift>` where there is a change in the relationship between the input data and the model target. It reflects the evolution of the underlying problem statement or process over time.

Confidence Level
    In statistics, the confidence level indicates the probability with which the estimation of the location of a statistical parameter (for example an arithmetic mean) in a sample survey is also true for the actual population.

Confusion Matrix
    A matrix made up of the following measurements: True Positives, True Negatives, False Positives and False Negatives. An image is shown below.

    See {term}`True Positive Rate (TP)`, {term}`True Negative Rate (TN)`, {term}`False Positive Rate (FP)`, {term}`False Negative Rate (FN)`

    ![Confusion Matrix](./images/confusion_matrix.png)

Contractive Autoencoder (CAE)
    A type of {term}`autoencoder<Autoencoder>` which is designed to be sensitive to small changes in the training dataset. It attempts to increase the robustness of the model by emphasizing the accurate encoding of small changes in the training data.

Convolutional Layer
    Input layer to a {term}`convolutional neural network<Convolutional Neural Network (CNN)>`. Core building block of CNN where a majority of the computation occurs. It requires 3 components; input data, a filter or kernel (matrix), and a feature map. The kernel sweeps across the input data's feature map using a dot product operation to 'find' the features. This process is called convolution.

Convolutional Neural Network (CNN)
    A type of {term}`Deep Neural Network<Deep Neural Network (DNN)>` used in computer vision applications such as image classification, image segmentation and image and video recognition. It uses at a minimum of 3 layers; the convolutional layer, the pooling layer and the Fully Connected (FC) Layer. With each layer, the CNN increases its complexity, identifying greater portions of the input image.

Coverage
    A measure of the distribution of the images in a dataset. A covered dataset has at least one image for every distinguishing property of the data set.

Cramér-von Mises (CVM) Drift Detection
    The Cramér-von Mises algorithm tests the {term}`null hypothesis<Null Hypothesis>` that a data sample (i.e. operational dataset), comes from a pre-specified population distribution or a family of such distributions with the idea that if the operational dataset does not come from the same distribution or family of distributions as the training dataset then data drift may have occurred. It is similar to the {term}`Kolmogorov-Smirnov test<Kolmogorov-Smirnov (K-S) Test>`.
```

## D

```{glossary}
DataEval
    Name of the ARiA application for Test and Evaluation (T&E) of {term}`machine learning<Machine Learning (ML)>` and {term}`artificial intelligence<Artificial Intelligence (AI)>` applications.

Dataset Splits
    Dataset splits, also known as data splitting, is the process of dividing a data set into multiple subsets to help train, test and evaluate {term}`machine learning<Machine Learning (ML)>` models.

Deduplication
    The process of identifying and exact or extremely similar data or images from a data set. A near duplicate is defined as within one standard deviation of the relevant distance statistic within a cluster of a data point in a data set.

Deep Neural Network (DNN)
    A deep neural network is an artificial {term}`neural network<Neural Network>` with multiple layers between the input and output layers. There are different types of neural networks but they generally consist of the same components: weights, biases, and activation function (and corresponding locations in the network). These components, as a whole, mimic the functionality of the human brain.

Denoising Autoencoder (DAE)
    A type of {term}`autoencoder<Autoencoder>` which is designed to increase its robustness by stochastically training the model for the reconstruction phase rather than the encoding phase.

Developmental Dataset
    Dataset used in for {term}`machine learning<Machine Learning (ML)>` model development. It consists of several {term}`dataset splits<Dataset Splits>` that can include training, validation and testing splits.

Dimensionality Reduction
    Dimensionality reduction is a method for representing a given dataset using a lower number of features (i.e. dimensions) while still capturing the original data’s meaningful properties. This amounts to removing irrelevant or redundant features, or simply noisy data, or combining features into a reduced number of new features to create a model with a lower number of variables.

Divergence
    Divergence is a kind of statistical distance: a function which establishes the separation from one probability distribution to another on a statistical {term}`manifold<Manifold>`.

Diversity
    A measure of the distribution of metadata factors in the dataset. A balanced dataset has an even distribution of class labels and generative factors.

Drift
    In predictive analytics, data science, {term}`machine learning<Machine Learning (ML)>` and related fields, the phenomenon where the statistical properties of the data change over time. It occurs when the underlying distribution of the input features or the target variable (what the model is trying to predict) shifts, leading to a discrepancy between the training data and the real-world data the model encounters during deployment.

Duplicates
    Statistical duplicates, or duplicate data, are repeated records or observations in a dataset. They can be caused by human error, technical errors, and/or data manipulation. In the case of {term}`DataEval` for the image classification and/or detection tasks, exact matches are found using a byte hash of the image information, while near matches use a {term}`perception-based hash<Perception-Based Hash>`
```

## E

```{glossary}
Embeddings
    Embeddings are representations of values or objects like text, images, and audio that are designed to be consumed by {term}`machine learning<Machine Learning (ML)>` models and semantic search algorithms.

Epoch
    Each time a dataset passes through an algorithm, it is said to have completed an epoch. Therefore, epoch, in {term}`machine learning<Machine Learning (ML)>`, refers to one entire pass of training data through the algorithm.
```

## F

```{glossary}
F1-Score
    One way to capture the precision-recall curve in a single metric. F1-Score combines {term}`precision<Precision>` and {term}`recall<Recall>` scores with equal weight, and works also for cases where the datasets are imbalanced as it requires both precision and recall to have a reasonable value. It is the harmonic mean of precision and recall and has a range of [0-1]. It is defined as follows:

    - $F1 = 2* \frac{\text{( Precision * Recall)}}{\text{(Precision + Recall)}}$

False Discovery Rate (FDR)
    The FDR is defined as the ratio of the number of False Positive Rate (FP) classifications (false discoveries) to the total number of True Positive Rate (TP) classifications (rejections of the null).

    - $FDR = \frac{FP}{(FP + TP)}$ \
    where:

    - TP = {term}`True Positive Rate (TP)`
    - FP = {term}`False Positive Rate (FP)`

False Discovery Rate (FDR) Correction
    False Discovery Rate Correction is a statistical procedure for correcting for the problem caused by running multiple hypothesis tests at once. It is typically used in high-throughput experiments in order to correct for random events that falsely appear significant.

False Negative Rate (FN)
    A measure of {term}`machine leaning<Machine Learning (ML)>` model accuracy. It measures the frequency at which a negative prediction was found for a positive ground truth value.

    See {term}`Confusion Matrix`

False Positive Rate (FP)
    A measure of {term}`machine learning<Machine Learning (ML)>` model accuracy. It measures the frequency at which a positive prediction was found for a negative ground truth value.

    See {term}`Confusion Matrix`

FB Score
    F "Beta" Score: The F-beta score is the weighted harmonic mean of {term}`precision<Precision>` and {term}`recall<Recall>` with a range of 0 (worst) to 1 (best). The beta ($\beta$) parameter represents the ratio of the recall importance to precision importance. A value greater than 1 gives more weight to recall while a value less than 1 favors precision. The formula is shown below:
    - $F_\beta =\frac{(1 +\beta^2)TP}{(1 +\beta^2)TP + FP + \beta^2FN}$ \
    where:

    - TP = {term}`True Positive Rate (TP)`
    - FP = {term}`False Positive Rate (FP)`
    - FN = {term}`False Negative Rate (FN)`

Feasibility
    Feasibility is a measure of whether the available data (both quantity and quality) can be used to satisfy the necessary performance characteristics of the {term}`machine learing<Machine Learning (ML)>` model.

Fully-Connected Layer
    Layer in a {term}`neural network<Neural Network>`. Every node in the layer is connected to every node in the previous layer. It performs the task of classification by creating probabilities that input data contains the features filtered for by the neural network.
```

## G

```{glossary}
Generative Model
    A type of {term}`machine learning<Machine Learning (ML)>` model which can generate new data that is similar to the input/training data for the model. For example, large language models (LLMs) are able to generate new text from their large pool of language data.
```

## H

```{glossary}
Hamming Distance
    The Hamming Distance between two strings of equal length is the number of positions at which these strings vary. In more technical terms, it is a measure of the minimum number of changes required to turn one string into the other. In {term}`DataEval`, images are turned (hashed) into strings in order to compute their Hamming Distance.

Hilbert Space
    A  Hilbert Space is an inner product space that is a complete space where distance between instances can be measured with respect to the norm or distance function induced by the inner product. The Hilbert Space generalizes the Euclidean space to a finite or infinite dimensional space. Usually, the Hilbert Space is high dimensional. By convention in {term}`machine learning<Machine Learning (ML)>`, unless otherwise stated, Hilbert space is also referred to as the *feature space*.
```

## I

```{glossary}
Image Size
    The total number of pixels in an image. It is the product of the width (in pixels) and height (in pixels).
    - $Image Size = width*height$

Inference
    Statistical inference is a method of making decisions about the parameters of a population, based on random sampling.

Irreducible Error
    The irreducible error, also known as noise, represents the variability or randomness in the data that cannot be explained by a regression model. The irreducible error arises from various sources such as unmeasured variables, measurement errors, or natural variability in the target variable. It is independent of the predictor variables X and cannot be reduced by improving the model. The only way to reduce irreducible error is by improving the quality of the data or acquiring additional information that captures the unexplained variability.

    See {term}`Bayes Error Rate (BER)`
```

## J

```{glossary}
Joint Sample
    A joint sample is a draw from the underlying distribution of both input data and its (potentially unknown) corresponding label. The correlation between input data and matching label is what a model attempts to capture. The joint distribution from which the joint sample is drawn characterizes this correlation.
```

## K

```{glossary}
Kolmogorov-Smirnov (K-S) test
    In statistics, the Kolmogorov–Smirnov Test is a nonparametric test of the equality of continuous, one-dimensional probability distributions that can be used to test whether a sample came from a given reference probability distribution (one-sample K–S test), or to test whether two samples came from the same distribution (two-sample K–S test). Intuitively, the test provides a method to qualitatively answer the question "How likely is it that we would see a collection of samples like this if they were drawn from that probability distribution?" or, in the second case, "How likely is it that we would see two sets of samples like this if they were drawn from the same (but unknown) probability distribution?". It is named after Andrey Kolmogorov and Nikolai Smirnov.
```

## L

```{glossary}
Label Parity
    Label parity is a means for assessing equivalence in label frequency between datasets. Label parity informs the user if the distribution of labels is different.

Label Shift
    In many real-world applications, the target (testing) distribution of a model can differ from the source (training) distribution. Label shift arises when class proportions differ between the source and target, but the feature distributions of each class do not. For example, the problems of bird identification in San Francisco (SF) versus New York (NY) exhibit label shift. While the likelihood of observing a snowy owl may differ, snowy owls should look similar in New York and San Francisco.

Laplacian Filter
    The Laplacian filter is an edge detection filter. It uses the second derivatives of an image to find regions of rapid intensity change.

Latent Space
    Also known as a Latent Feature Space or Encoded Space, is an embedding of a set of items within a {term}`manifold<Manifold>` in which items resembling each other under some encoding are positioned close to one another. Positions are defined by a set of *latent* variables that emerge from the properties of the objects. For example, placing images within a Gaussian Distribution based upon their color properties defines the parameters of the Gaussian and reduces the number of parameters for the images.

Linter
    The data linter identifies potential issues (lints) in the ML training data. The term "linter" stems from the origins of a tool known as "lint," which was initially developed by Stephen C. Johnson in 1978 at Bell Labs. For {term}`DataEval` and imagery, it identifies issues such as image quality (overly bright/dark, overly blurry, lacking information), unusual image properties (shape,size, **channels*), as well as duplicates and outliers.
```

## M

```{glossary}
Machine Learning (ML)
    Machine learning (ML) is a branch of {term}`artificial intelligence<Artificial Intelligence (AI)>` and computer science that focuses on the using data and algorithms to enable AI to imitate the way that humans learn, gradually improving its accuracy. In general, machine learning algorithms are used to make a prediction or {term}`classification<Classification>`. Based on some input data, which can be labeled or unlabeled, your algorithm will produce an estimate about a pattern in the data.

Manifold
    In mathematics, a manifold is a topological space that locally resembles Euclidean space near each point. One dimensional manifolds include lines and circles. Two dimensional manifolds are also called surfaces. One example is the family of Gaussian or Normal Functions. They form a manifold parameterized by the expected value and {term}`variance<Variance>` of the Gaussian functions.

Maximum Pooling
    Method used in the {term}`pooling layer<Pooling Layer>` of a {term}`convolutional neural netwok<Convolutional Neural Network (CNN)>`. It uses the maximum value in a group of pixel values (typically a 2 x 2 or 3 x 3 area) produced from the {term}`convolutional layer<Convolutional Layer>` to reduce the dimensionality of the result. An image of the operation is shown below.

    ![Maximum Pooling Example](./images/max_pooling.png)

Maximum Mean Discrepancy (MMD) Drift Detection
    MMD Drift Detection is a method which compares the mean {term}`embeddings<Embeddings>` of each sample: A reference sample and a target sample. For example, one might use the mean image embedding for a set of training data, and compare that to the mean embedding of an operational dataset. If the two means are significantly different from one another, drift is detected. What constitutes a "significant" difference stems from underlying assumptions about the distribution of embeddings. More info can be found here:
    [MMD Reference Paper](https://www.stat.berkeley.edu/~ryantibs/papers/radonks.pdf).

    See {term}`Drift`

Mean Average Precision
    A measure to evaluate the performance of object detection and segmentation systems over all classes covered by the system. It is calculated using the {term}`confusion matrix<Confusion Matrix>`, {term}`precision<Precision>` and {term}`recall<Recall>` metrics and the Area Under the Curve (AUC) metric. It is the average precision for each class averaged over the number of classes.

Minimum Pooling
    Method used in the {term}`pooling layer<Pooling Layer>` of a {term}`convolutinal neural network<Convolutional Neural Network (CNN)>`. It uses the minimum value in a group of pixel values (typically a 2 x 2 or 3 x 3 area) produced from the {term}`convolutional layer<Convolutional Layer>` to reduce the dimensionality of the result.

Modular AI Trustworthy Engineering (MAITE)
    A toolbox of common types, protocols (a.k.a. structural subtypes) and tooling to support AI test and evaluation (T&E) workflows. Part of the Joint AI T&E Infrastructure Capability (JATIC) program. Its goal is to streamline the development of JATIC Python projects by ensuring seamless, synergistic workflows when working with MAITE-conforming Python packages for different T&E tasks.

Mutual Information (MI)
    In probability theory and information theory, the mutual information of two random variables is a measure of the mutual dependence between the two variables. More specifically, it quantifies the "amount of information" obtained about one random variable by observing the other random variable.
```

## N

```{glossary}
Neural Network
    A neural network is a method in {term}`artificial intelligence<Artificial Intelligence (AI)>` that teaches computers to process data in a way that is inspired by the human brain. An example depiction is shown below:

    ![Example Neural Network](./images/neural_network_example.png)

Null Hypothesis
    In scientific research, the null hypothesis (often denoted H0) is the claim that the effect being studied does not exist. The null hypothesis can also be described as the hypothesis in which no relationship exists between the two sets of data or variables being analyzed. If the null hypothesis is true, any experimentally observed effect is due to chance alone, hence the term "null".

NumPy
    NumPy is a Python Library used for working with arrays. It also has functions for working in linear algebra, fourier transforms and matrices (used in convolution). NumPy was created in 2005 by Travis Oliphant.
```

## O

```{glossary}
Object Detection
    Object Detection is a computer vision technique for locating instances of objects in images or videos. Object detection algorithms typically leverage {term}`machine learning<Machine Learning (ML)>` or *deep learning* to produce meaningful results.

Operational Dataset
    Dataset used while a {term}`machine learning<Machine Learning (ML)>` model is in operation and not used or seen during training. It may or may not match the characteristics of the {term}`developmental dataset<Developmental Dataset>`.

    See {term}`Drift`.

Operational Drift
    Operational drift is specific type of {term}`drift<Drift>` defined as data drift during data model operations. It occurs when the data used in operation is not like data used during training of a {term}`machine learning<Machine Learning (ML)>` model. It can make the model less accurate in its predictions/classifications.

Outlier Detection
    Outlier Detection is process of detecting data/images that significantly deviate from the rest of the data/images. {term}`DataEval` uses the measure of two standard deviations of the  average of the relevant distance measure to identify outliers.

Outliers (Images)
    Images which differ significantly from all or most of the other images in a dataset.

Out-of-distribution (OOD)
    Out-of-distribution (OOD) data refers to data that is different from the data used to train the {term}`machine learning<Machine Learning (Ml)>` model. For example, data collected in a different way, at a different time, under different conditions, or for a different task than the data on which the model was originally trained. An illustration is shown below:

    ![Out Of Distribution Illustration](./images/out_of_distribution.png)

Overfitting
    Overfitting occurs when the {term}`machine learning<Machine Learning (ML)>` model cannot generalize. The model begins to predict heavily towards the distribution of the {term}`developmental dataset<Developmental Dataset>` which can hurt the model's predictions on the {term}`operational dataset<Operational Dataset>`. Overfitting happens due to several reasons, such as:

    - The training data size is too small and does not contain enough data samples to accurately represent all possible input data values.
    - The training data contains large amounts of irrelevant information, called noisy data.
    - The model trains for too long on a single sample set of data.
    - The model complexity is high, so it learns the noise within the training data.
```

## P

```{glossary}
P-Value
    A p-value, or probability value, is a number describing how likely it is that your data would have occurred under the {term}`null hypothesis<Null Hypothesis>` of your statistical test.

Parity
    Parity, in data analysis, measures the {term}`statistical independence<Statistical Independence>` between class labels and metadata factors using a {term}`Chi-Square Test of Independence`. A lack of independence indicates bias. See {term}`bias<Bias>`.

Perception-based Hash
    Hashing algorithms designed to remain unchanged on very similar inputs. They are designed to detect similar characteristics such as images that have only changed in color, brightness or compression method. So-called *pHashes*, allow for the comparison of two images by looking at the number of different bits between the input and a second image. This difference is known as the Hamming Distance.

    See {term}`Hamming Distance`

Pooling Layer
    Layers within a {term}`convolutional neural network<Convolutional Neural Network (CNN)>`, also known as *downsampling layers*. Similar to the {term}`convolutinal layer<Convolutional Layer>`, they typically sweep across the output of a convolutional layer and apply aggregation functions using a similar kernel calculation to the output. A lot of information is lost in the pooling layer but they reduce complexity, improve efficiency and limit the risk of {term}`overfitting<Overfitting>` to the {term}`developmental dataset<Developmental Dataset>`. The functions include {term}`minimul pooling<Minimum Pooling>`, {term}`maximum pooling<Maximum Pooling>` and {term}`average pooling<Average Pooling>`.

Precision
    A measure of how often a {term}`machine learning<Machine Learning (ML)>` model correctly predicts the positive class. It is defined as the True Positive Rate (TP) rate divided by the sum of the True Positive Rate (TP) and False Positive Rate (FP) rates.

    - $Precision = \frac{TP}{( TP + FP )}$ \
    where:

    - TP = {term}`True Positive Rate (TP)`
    - FP = {term}`False Positive Rate (FP)`

Precision Recall Curve
    A plot which shows {term}`precision<Precision>` (Y Axis) vs. {term}`recall<Recall>` (X axis) scores for one or more {term}`machine learning<Machine Learning (ML)>` models as a function of the {term}`confidence level<Confidence Level>` to make a prediction. In general, as confidence levels drop, precision decreases as recall increases. The curves are used to visualize the accuracy of a model using various measures including {term}`F1-score<F1-Score>`, *Area Under the Curve (AUROC)* and *Average Precision (AP)*. An example is shown below:

    ![Example Precision Recall Curve](./images/precision_recall_curve.png)

Principal Component Analysis (PCA)
    Principal component analysis (PCA) is a linear {term}`dimensionality reduction<Dimensionality Reduction>` technique with applications in exploratory data analysis, visualization and data preprocessing. The data is linearly transformed onto a new coordinate system such that the directions (principal components) capturing the largest variation in the data can be easily identified.

Probability Distribution
    A Probability Distribution is a mathematical function that gives the probabilities of occurrence of different possible random outcomes for an experiment. It can be defined for both discrete and continuous variables.
```

## Q

## R

```{glossary}
Recall
    A measure of the {term}`accuracy<Accuracy>` of a {term}`machine learning<Machine Learning (ML)>` model. It is the ability of a model to find all of the relevant cases within a data set. It is defined as the True Positive Rate (TP) over the sum of the *True Positive Rate* and False Negative Rate (FN).
    - $Recall = \frac{TP}{( TP + FN)}$ \
    where:

    - TP = {term}`True Positive Rate (TP)`
    - FN = {term}`False Negative Rate (FN)`

Receiver Operating Characteristic Curve
    A curve frequently used to evaluate the performance of {term}`binary classification<Binary Classification>` algorithms. It provides a graphical representation of a classifier's performance. It plots the {term}`true positive rate<True Positive Rate (TP)>` vs the {term}`false positive rate<False Positive Rate (FP)>` at a variety of thresholds. An example image is shown below:

    ![Example ROC Curve](./images/ROC_curve.png)

Regularized Autoencoder
    A type of {term}`autoencoder<Autoencoder>` used mostly in {term}`classification<Classification>` tasks. Types include *Sparse*, *Denoising* and *Contractive*. Regularization is a method for constraining the model in order to prevent {term}`overfitting<Overfitting>` and improve its ability to generalize to new data.

Riemannian Manifold
    A manifold where distances and angles can be measured. Type of manifold where {term}`divergence<Divergence>` can be measured.

ROC Curve
    See {term}`Receiver Operating Characteristic Curve`
```

## S

```{glossary}
Sparse Autoencoder (SAE)
    A type of {term}`autoencoder<Autoencoder>` inspired by the sparse coding hypothesis in neuroscience with the idea that more relevant information will be encoded in a machine learning network if fewer nodes are activated during any single input. By penalizing activation of multiple nodes, it is hoped that more relevant information is encoded rather than passing redundant information.

Statistical Independence
    Two events are statistically independent of one another if the probability of one or either event occurring is not affected by the occurrence or nonoccurrence of the other event.

Statistical Manifold
    An abstract space where each point is a probability distribution. An illustration of the concept is shown below:

    ![statistical manifold](./images/statistical_manifold.png)

Statistics
    Statistics is a branch of applied mathematics that involves the collection, description, analysis, and inference of conclusions from quantitative data.

Sufficiency
    Sufficiency, in the context of data analysis and the {term}`DataEval` tool, is the notion that the model and/or dataset are capable of satisfying the operational requirements. See {term}`Bayes Error Rate (BER)` and {term}`Upper-bound Average Precision (UAP)`.

Supervised Learning
    Supervised Learning is a category of {term}`machine learning<Machine Learning (ML)>` that uses labeled datasets to train algorithms to predict outcomes and recognize patterns. Unlike {term}`unsupervised learning<Unsupervised Learning>`, supervised learning algorithms are given labeled training data to learn the relationship between the inputs and the outputs.
```

## T

```{glossary}
TensorFlow
    TensorFlow is a free and open-source software library for {term}`machine learning<Machine Learning (ML)>` and {term}`artificial intelligence<Artificial Intelligence (AI)>`. It can be used across a range of tasks but focuses on training and inference of Deep Neural Networks (DNNs)`.

    See {term}`Deep Neural Network (DNN)`

Torch (PyTorch)
    Torch (or Pytorch) is an open-source {term}`machine learing<Machine Learning (ML)>` library. One of its core packages provides a flexible N-dimensional array (also called a Tensor) data structure used by many machine learning algorithms. It supports routines for manipulation and calculation using Tensors. PyTorch is a library written for the Python programming language.

True Negative Rate (TN)
    A measure of {term}`machine learning<Machine Learning (ML)>` model accuracy. It measures the frequency a negative prediction was found for a negative ground truth value.

    See {term}`Confusion Matrix`

True Positive Rate (TP)
    A measure of {term}`machine learning<Machine Learning (ML)>` model accuracy. It measures the frequency a positive prediction was found for a positive ground truth value.

    See {term}`Confusion Matrix`
```

## U

```{glossary}
Unsupervised Learning
    Unsupervised learning is a branch of {term}`machine learning<Machine Learning (ML)>` that learns from data without human supervision. Unlike {term}`Supervised Learning`, unsupervised machine learning models are given unlabeled data and allowed to discover patterns and insights without any explicit guidance or instruction.

Upper-bound Average Precision (UAP)
    Object detection equivalent of {term}`Bayes Error Rate<Bayes Error Rate (BER)>`. An estimate of an upper bound on performance for an object detection model on a dataset.
```

## V

```{glossary}
Variance
    In probability theory and statistics, variance is the expected value of the squared deviation from the mean of a random variable. Variance is a measure of dispersion from the average value.

Variational Autoencoder: (VAE)
    A type of {term}`autoencoder<Autoencoder>` used extensively in generative models (for example Large Language Models (LLMs)) because of its ability to generate new content. The encoder maps each data point (such as an image) from a large complex dataset into a distribution (for example Gaussian) within a {term}`latent space<Latent Space>` or encoded space, rather than a single point.
```

## W

## X

## Y

## Z
