{% macro auto_summary(objs, title='') %}

.. list-table:: {{ title }}
   :class: autosummary longtable
   :widths: auto
   :header-rows: 0

{% for obj in objs %}
   {% set module_names = obj.name.split('.') %}
   * - :py:obj:`{{ module_names[-1] }}<{{ obj.id }}>`
     - {{ obj.summary }} 
{% endfor %}
{% endmacro %}
