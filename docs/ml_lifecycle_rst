========================================
Machine-Learning Lifecycle and Workflows
========================================

--------------------------------------
Operational Machine-Learning Lifecycle
--------------------------------------

Unlike a competition lifecycle of model development, the operational lifecycle is an iterative process and has
workflows and metrics associated with each stage, and also cuts across multiple stages.  The competition is
between near-peer nations for superior operational capability, not between model developers for a better metric
in a Kaggle-style AI competition.

-----------------------------
Stages/Steps of the Lifecycle
-----------------------------

.. graphviz::
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [label="I: Scope And\nObjectives",style="rounded,filled",pos="1.7,2.5!",xref=":ref:`scope-objectives`"]
      2 [label="II: Data\nEngineering",style="rounded,filled",pos="3.4,1.8!",xref=":ref:`data-engineering`"]
      3 [label="III: Model\nDevelopment",style="rounded,filled",pos="3.4,0.9!",xref=":ref:`model-development`"]
      4 [label="IV: Deployment",style="rounded,filled",pos="1.7,0.2!",xref=":ref:`deployment`"]
      5 [label="V: Monitoring",style="rounded,filled",pos="0.0,0.9!",xref=":ref:`monitoring`"]
      6 [label="VI: Analysis",style="rounded,filled",pos="0.0,1.8!",xref=":ref:`analysis`"]
      
      1:e->2:n; 2:s->3:n; 3:s->4:e; 4:w->5:s; 5:n->6:s; 6:n->1:w

      1:s->3:w [dir=both,style=dashed]; 1:s->4:n [dir=both,style=dashed]
      1:s->5:e [dir=both,style=dashed]; 1:s->2:w [dir=both,style=dashed]
      2:w->5:e [dir=both,style=dashed]; 2:w->6:e [dir=both,style=dashed]
      3:w->5:e [dir=both,style=dashed]; 3:w->6:e [dir=both,style=dashed]
   }

*Consensus machine-learning lifecycle derivation*  [#f1]_ [#f2]_ [#f3]_

* not a linear, sequential process
* roles and responsibilities across stages and personnel are dynamic
* "mind" (models) and "data" are both important

.. _scope-objectives:

--------------------
Scope And Objectives
--------------------

.. graphviz::
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [label="I: Scope And\nObjectives",style="rounded,filled",pos="1.7,2.5!",fillcolor="#4151B0",fontcolor="white"]
      2 [label="II: Data\nEngineering",style="rounded,filled",pos="3.4,1.8!",xref=":ref:`data-engineering`"]
      3 [label="III: Model\nDevelopment",style="rounded,filled",pos="3.4,0.9!",xref=":ref:`model-development`"]
      4 [label="IV: Deployment",style="rounded,filled",pos="1.7,0.2!",xref=":ref:`deployment`"]
      5 [label="V: Monitoring",style="rounded,filled",pos="0.0,0.9!",xref=":ref:`monitoring`"]
      6 [label="VI: Analysis",style="rounded,filled",pos="0.0,1.8!",xref=":ref:`analysis`",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      
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

.. _data-engineering:

----------------
Data Engineering
----------------

.. graphviz::
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [label="I: Scope And\nObjectives",style="rounded,filled",pos="1.7,2.5!",xref=":ref:`scope-objectives`"]
      2 [label="II: Data\nEngineering",style="rounded,filled",pos="3.4,1.8!",fillcolor="#4151B0",fontcolor="white"]
      3 [label="III: Model\nDevelopment",style="rounded,filled",pos="3.4,0.9!",xref=":ref:`model-development`"]
      4 [label="IV: Deployment",style="rounded,filled",pos="1.7,0.2!",xref=":ref:`deployment`",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      5 [label="V: Monitoring",style="rounded,filled",pos="0.0,0.9!",xref=":ref:`monitoring`"]
      6 [label="VI: Analysis",style="rounded,filled",pos="0.0,1.8!",xref=":ref:`analysis`"]
      
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

* develop data pipelines

  * data linting

* develop labeling protocols and pipelines

  * label-error detection

* formulate sampling protocols

  * coverage assessment
  * ensure relevance, completeness, balance, and accuracy

* curate static training and test datasets

  * assess leakage
  * train/test shift

* perform exploratory data analysis

  * data complexity / metafeatures to evaluate achievability of objectives

.. _model-development:

-----------------
Model Development
-----------------

.. graphviz::
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [label="I: Scope And\nObjectives",style="rounded,filled",pos="1.7,2.5!",xref=":ref:`scope-objectives`"]
      2 [label="II: Data\nEngineering",style="rounded,filled",pos="3.4,1.8!",xref=":ref:`data-engineering`",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      3 [label="III: Model\nDevelopment",style="rounded,filled",pos="3.4,0.9!",fillcolor="#4151B0",fontcolor="white"]
      4 [label="IV: Deployment",style="rounded,filled",pos="1.7,0.2!",xref=":ref:`deployment`"]
      5 [label="V: Monitoring",style="rounded,filled",pos="0.0,0.9!",xref=":ref:`monitoring`"]
      6 [label="VI: Analysis",style="rounded,filled",pos="0.0,1.8!",xref=":ref:`analysis`"]
      
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

.. _deployment:

----------
Deployment
----------

.. graphviz::
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [label="I: Scope And\nObjectives",style="rounded,filled",pos="1.7,2.5!",xref=":ref:`scope-objectives`"]
      2 [label="II: Data\nEngineering",style="rounded,filled",pos="3.4,1.8!",xref=":ref:`data-engineering`",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      3 [label="III: Model\nDevelopment",style="rounded,filled",pos="3.4,0.9!",xref=":ref:`model-development`",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      4 [label="IV: Deployment",style="rounded,filled",pos="1.7,0.2!",fillcolor="#4151B0",fontcolor="white"]
      5 [label="V: Monitoring",style="rounded,filled",pos="0.0,0.9!",xref=":ref:`monitoring`"]
      6 [label="VI: Analysis",style="rounded,filled",pos="0.0,1.8!",xref=":ref:`analysis`",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      
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

* online or batch prediction?
* online or streaming features?
* model update cycle?
* model compression
* model optimization

.. note::
   Deployment decisions can impact model 
   performance metrics and these impacts need to 
   be assessed.

.. _monitoring:

----------
Monitoring
----------

.. graphviz::
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [label="I: Scope And\nObjectives",style="rounded,filled",pos="1.7,2.5!",xref=":ref:`scope-objectives`"]
      2 [label="II: Data\nEngineering",style="rounded,filled",pos="3.4,1.8!",xref=":ref:`data-engineering`"]
      3 [label="III: Model\nDevelopment",style="rounded,filled",pos="3.4,0.9!",xref=":ref:`model-development`"]
      4 [label="IV: Deployment",style="rounded,filled",pos="1.7,0.2!",xref=":ref:`deployment`",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      5 [label="V: Monitoring",style="rounded,filled",pos="0.0,0.9!",fillcolor="#4151B0",fontcolor="white"]
      6 [label="VI: Analysis",style="rounded,filled",pos="0.0,1.8!",xref=":ref:`analysis`"]
      
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

.. _analysis:

--------
Analysis
--------

.. graphviz::
   
   digraph flowchart {
      node [shape=box,width=1.7,height=0.6]
      edge [arrowsize=0.6]
      layout="neato"

      1 [label="I: Scope And\nObjectives",style="rounded,filled",pos="1.7,2.5!",xref=":ref:`scope-objectives`"]
      2 [label="II: Data\nEngineering",style="rounded,filled",pos="3.4,1.8!",xref=":ref:`data-engineering`"]
      3 [label="III: Model\nDevelopment",style="rounded,filled",pos="3.4,0.9!",xref=":ref:`model-development`"]
      4 [label="IV: Deployment",style="rounded,filled",pos="1.7,0.2!",xref=":ref:`deployment`",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      5 [label="V: Monitoring",style="rounded,filled",pos="0.0,0.9!",xref=":ref:`monitoring`",fontcolor="gray",color="#C0C0C030",fillcolor="#C0C0C030"]
      6 [label="VI: Analysis",style="rounded,filled",pos="0.0,1.8!",fillcolor="#4151B0",fontcolor="white"]
      
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

* determine whether model achieves specified goal and objective requirements
* refine data engineering and model development stages as needed to achieve objective requirements
* perform analysis on model predictions to generate operational insight that drive refinement of scope and objectives for future iterations

--------------------------
*Always we begin again...*
--------------------------

* developing and deploying an ML system is a never-ending cyclical process

* the world changes and models must change to adapt to the changing world

* modern ML deployment is approaching DevOps timelines

  * Weibo, Alibaba, and ByteDance deploy new ML models on a **10 minute update cycle**

.. epigraph::
   
   *People tend to ask me: ‘How often should I update my models?’... The right question to ask should be: ‘How often can I update my models?’*
   
   -- Chip Huyen

.. rubric:: Footnotes
.. [#] Chip Huyen, *Designing Machine Learning Systems*, O’Reilly, 2022
.. [#] *Well-Architected machine learning lifecycle* `(link) <https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/well-architected-machine-learning-lifecycle.html>`_ accessed 31 JUL 2023 (other elements of this resource, but not this page, cited in original CDAO persona documents)
.. [#] H\. Veeradhi and K. Abdo, *Your guide to the Red Hat Data Science Model Lifecycle* `(link) <https://cloud.redhat.com/blog/your-guide-to-the-red-hat-data-science-model-lifecycle>`_ accessed 09 MAY 2022 (cited in original CDAO persona documents)
