# Machine-Learning Lifecycle and Workflows

## Operational Machine-Learning Lifecycle

Unlike a competition lifecycle of model development, the operational lifecycle
is an iterative process and has workflows and metrics associated with each
stage, and also cuts across multiple stages. The competition is between
near-peer nations for superior operational capability, not between model
developers for a better metric in a Kaggle-style AI competition.

## Stages/Steps of the Lifecycle

```{graphviz}
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [xref="{ref}`Scope And Objectives <scope-and-objectives>`",style="rounded,filled",pos="1.7,2.5!"]
      2 [xref="{ref}`Data Engineering <data-engineering>`",style="rounded,filled",pos="3.4,1.8!"]
      3 [xref="{ref}`Model Development<model-development>`",style="rounded,filled",pos="3.4,0.9!"]
      4 [xref="{ref}`Deployment <deployment>`",style="rounded,filled",pos="1.7,0.2!"]
      5 [xref="{ref}`Monitoring <monitoring>`",style="rounded,filled",pos="0.0,0.9!"]
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!"]
      
      1:e->2:n; 2:s->3:n; 3:s->4:e; 4:w->5:s; 5:n->6:s; 6:n->1:w

      1:s->3:w [dir=both,style=dashed]; 1:s->4:n [dir=both,style=dashed]
      1:s->5:e [dir=both,style=dashed]; 1:s->2:w [dir=both,style=dashed]
      2:w->5:e [dir=both,style=dashed]; 2:w->6:e [dir=both,style=dashed]
      3:w->5:e [dir=both,style=dashed]; 3:w->6:e [dir=both,style=dashed]
   }
```

*Consensus machine-learning lifecycle derivation* [^1] [^2] [^3]

* not a linear, sequential process
* roles and responsibilities across stages and personnel are dynamic
* "mind" (models) and "data" are both important

(scope-and-objectives)=

## Scope And Objectives

```{graphviz}
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [xref="{ref}`Scope And Objectives <scope-and-objectives>`",style="rounded,filled",pos="1.7,2.5!",fillcolor="#4151B0",fontcolor="white"]
      2 [xref="{ref}`Data Engineering <data-engineering>`",style="rounded,filled",pos="3.4,1.8!"]
      3 [xref="{ref}`Model Development<model-development>`",style="rounded,filled",pos="3.4,0.9!"]
      4 [xref="{ref}`Deployment <deployment>`",style="rounded,filled",pos="1.7,0.2!"]
      5 [xref="{ref}`Monitoring <monitoring>`",style="rounded,filled",pos="0.0,0.9!"]
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      
      1:e->2:n
      2:s->3:n [color="#C0C0C030",fillcolor="#C0C0C030"]
      3:s->4:e [color="#C0C0C030",fillcolor="#C0C0C030"]
      4:w->5:s [color="#C0C0C030",fillcolor="#C0C0C030"]
      5:n->6:s [color="#C0C0C030",fillcolor="#C0C0C030"]
      6:n->1:w [color="#C0C0C030",fillcolor="#C0C0C030"]

      1:s->3:w [dir=both,style=dashed]
      1:s->4:n [dir=both,style=dashed]
      1:s->5:e [dir=both,style=dashed]
      1:s->2:w [dir=both,style=dashed]
      2:w->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      2:w->6:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      3:w->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      3:w->6:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
   }
```

* define the scope of the problem and goals for the solution
* specify operational requirements

  * material release
  * safety analysis
  * doctrine
  * human factors
  * ...

* specify operational constraints

  * restrictions on generative factors for evaluating completeness
  * access to labels/groundtruth
  * restrictions on lifetime learning
  * ...

(data-engineering)=

## Data Engineering

```{graphviz}
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [xref="{ref}`Scope And Objectives <scope-and-objectives>`",style="rounded,filled",pos="1.7,2.5!"]
      2 [xref="{ref}`Data Engineering <data-engineering>`",style="rounded,filled",pos="3.4,1.8!",fillcolor="#4151B0",fontcolor="white"]
      3 [xref="{ref}`Model Development<model-development>`",style="rounded,filled",pos="3.4,0.9!"]
      4 [xref="{ref}`Deployment <deployment>`",style="rounded,filled",pos="1.7,0.2!",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      5 [xref="{ref}`Monitoring <monitoring>`",style="rounded,filled",pos="0.0,0.9!"]
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!"]
      
      1:e->2:n [color="#C0C0C030",fillcolor="#C0C0C030"]
      2:s->3:n
      3:s->4:e [color="#C0C0C030",fillcolor="#C0C0C030"]
      4:w->5:s [color="#C0C0C030",fillcolor="#C0C0C030"]
      5:n->6:s [color="#C0C0C030",fillcolor="#C0C0C030"]
      6:n->1:w [color="#C0C0C030",fillcolor="#C0C0C030"]

      1:s->3:w [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->4:n [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->2:w [dir=both,style=dashed]
      2:w->5:e [dir=both,style=dashed]
      2:w->6:e [dir=both,style=dashed]
      3:w->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      3:w->6:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
   }
```

* develop data pipelines
  * data linting
* develop labeling protocols and pipelines
  * label-error detection
* formulate sampling protocols
  * coverage assessment
  * ensure relevance, completeness, {term}`balance<Balance>`, and
    {term}`accuracy<Accuracy>`.
* curate static training and test datasets
  * assess leakage
  * train/test shift
* perform exploratory data analysis
  * data complexity / metafeatures to evaluate achievability of objectives

(model-development)=

## Model Development

```{graphviz}
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [xref="{ref}`Scope And Objectives <scope-and-objectives>`",style="rounded,filled",pos="1.7,2.5!"]
      2 [xref="{ref}`Data Engineering <data-engineering>`",style="rounded,filled",pos="3.4,1.8!",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      3 [xref="{ref}`Model Development<model-development>`",style="rounded,filled",pos="3.4,0.9!",fillcolor="#4151B0",fontcolor="white"]
      4 [xref="{ref}`Deployment <deployment>`",style="rounded,filled",pos="1.7,0.2!"]
      5 [xref="{ref}`Monitoring <monitoring>`",style="rounded,filled",pos="0.0,0.9!"]
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!"]
      
      1:e->2:n [color="#C0C0C030",fillcolor="#C0C0C030"]
      2:s->3:n [color="#C0C0C030",fillcolor="#C0C0C030"]
      3:s->4:e
      4:w->5:s [color="#C0C0C030",fillcolor="#C0C0C030"]
      5:n->6:s [color="#C0C0C030",fillcolor="#C0C0C030"]
      6:n->1:w [color="#C0C0C030",fillcolor="#C0C0C030"]

      1:s->3:w [dir=both,style=dashed]
      1:s->4:n [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->2:w [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      2:w->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      2:w->6:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      3:w->5:e [dir=both,style=dashed]
      3:w->6:e [dir=both,style=dashed]
   }
```

* model selection
  * metafeatures
  * model/data complexity matching
  * sufficiency assessment
* model training
  * training-data partitioning
  * training-data augmentation
  * leakage, bias and label errors
* model evaluation
  * performance
  * calibration
  * fairness and generalization
  * robustness and fault tolerance

(deployment)=

## Deployment

```{graphviz}
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [xref="{ref}`Scope And Objectives <scope-and-objectives>`",style="rounded,filled",pos="1.7,2.5!"]
      2 [xref="{ref}`Data Engineering <data-engineering>`",style="rounded,filled",pos="3.4,1.8!",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      3 [xref="{ref}`Model Development<model-development>`",style="rounded,filled",pos="3.4,0.9!",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      4 [xref="{ref}`Deployment <deployment>`",style="rounded,filled",pos="1.7,0.2!",fillcolor="#4151B0",fontcolor="white"]
      5 [xref="{ref}`Monitoring <monitoring>`",style="rounded,filled",pos="0.0,0.9!"]
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      
      1:e->2:n [color="#C0C0C030",fillcolor="#C0C0C030"]
      2:s->3:n [color="#C0C0C030",fillcolor="#C0C0C030"]
      3:s->4:e [color="#C0C0C030",fillcolor="#C0C0C030"]
      4:w->5:s
      5:n->6:s [color="#C0C0C030",fillcolor="#C0C0C030"]
      6:n->1:w [color="#C0C0C030",fillcolor="#C0C0C030"]

      1:s->3:w [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->4:n [dir=both,style=dashed]
      1:s->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->2:w [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      2:w->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      2:w->6:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      3:w->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      3:w->6:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
   }
```

* online or batch prediction?
* online or streaming features?
* model update cycle?
* model compression
* model optimization

```{note}
   Deployment decisions can impact model 
   performance metrics and these impacts need to 
   be assessed.
```

(monitoring)=

## Monitoring

```{graphviz}
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [xref="{ref}`Scope And Objectives <scope-and-objectives>`",style="rounded,filled",pos="1.7,2.5!"]
      2 [xref="{ref}`Data Engineering <data-engineering>`",style="rounded,filled",pos="3.4,1.8!"]
      3 [xref="{ref}`Model Development<model-development>`",style="rounded,filled",pos="3.4,0.9!"]
      4 [xref="{ref}`Deployment <deployment>`",style="rounded,filled",pos="1.7,0.2!",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      5 [xref="{ref}`Monitoring <monitoring>`",style="rounded,filled",pos="0.0,0.9!",fillcolor="#4151B0",fontcolor="white"]
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!"]
      
      1:e->2:n [color="#C0C0C030",fillcolor="#C0C0C030"]
      2:s->3:n [color="#C0C0C030",fillcolor="#C0C0C030"]
      3:s->4:e [color="#C0C0C030",fillcolor="#C0C0C030"]
      4:w->5:s [color="#C0C0C030",fillcolor="#C0C0C030"]
      5:n->6:s
      6:n->1:w [color="#C0C0C030",fillcolor="#C0C0C030"]

      1:s->3:w [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->4:n [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->5:e [dir=both,style=dashed]
      1:s->2:w [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      2:w->5:e [dir=both,style=dashed]
      2:w->6:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      3:w->5:e [dir=both,style=dashed]
      3:w->6:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
   }
```

* data shifts
  * covariate shift - data monitoring
  * label shift - prediction monitoring
  * concept drift
* data monitoring
  * data distribution-shift
* feature monitoring
  * feature distribution-shift
* model monitoring
  * prediction distribution-shift
  * uncertainty/confidence shifts
  * accuracy/performance metrics

(analysis)=

## Analysis

```{graphviz}
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [xref="{ref}`Scope And Objectives <scope-and-objectives>`",style="rounded,filled",pos="1.7,2.5!"]
      2 [xref="{ref}`Data Engineering <data-engineering>`",style="rounded,filled",pos="3.4,1.8!"]
      3 [xref="{ref}`Model Development<model-development>`",style="rounded,filled",pos="3.4,0.9!"]
      4 [xref="{ref}`Deployment <deployment>`",style="rounded,filled",pos="1.7,0.2!",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      5 [xref="{ref}`Monitoring <monitoring>`",style="rounded,filled",pos="0.0,0.9!",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      6 [xref="{ref}`Analysis <analysis>`",style="rounded,filled",pos="0.0,1.8!",fillcolor="#4151B0",fontcolor="white"]
      
      1:e->2:n [color="#C0C0C030",fillcolor="#C0C0C030"]
      2:s->3:n [color="#C0C0C030",fillcolor="#C0C0C030"]
      3:s->4:e [color="#C0C0C030",fillcolor="#C0C0C030"]
      4:w->5:s [color="#C0C0C030",fillcolor="#C0C0C030"]
      5:n->6:s [color="#C0C0C030",fillcolor="#C0C0C030"]
      6:n->1:w

      1:s->3:w [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->4:n [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      1:s->2:w [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      2:w->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      2:w->6:e [dir=both,style=dashed]
      3:w->5:e [dir=both,style=dashed,color="#C0C0C030",fillcolor="#C0C0C030"]
      3:w->6:e [dir=both,style=dashed]
   }
```

* determine whether model achieves specified goal and objective requirements
* refine data engineering and model development stages as needed to achieve
  objective requirements
* perform analysis on model predictions to generate operational insight that
  drive refinement of scope and objectives for future iterations

## *Always we begin again...*

```{epigraph}

*People tend to ask me: ‘How often should I update my models?’... The right
question to ask should be: ‘How often can I update my models?’*

-- Chip Huyen
```

* developing and deploying an ML system is a never-ending cyclical process
* the world changes and models must change to adapt to the changing world
* modern ML deployment is approaching DevOps timelines
  * Weibo, Alibaba, and ByteDance deploy new ML models on a **10 minute update
    cycle**

## References

[^1]: Chip Huyen, "Designing Machine Learning Systems", O’Reilly, (2022)
[^2]: ["Well-Architected machine learning lifecycle" (accessed 31 JUL 2023)][r2]
[^3]: [H. Veeradhi and K. Abdo. "Your guide to the Red Hat Data Science Model Lifecycle" (accessed 09 MAY 2022)][r3]

[r2]: https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/well-architected-machine-learning-lifecycle.html
[r3]: https://cloud.redhat.com/blog/your-guide-to-the-red-hat-data-science-model-lifecycle
