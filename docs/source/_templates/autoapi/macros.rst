{% macro auto_summary(objs, title='') %}

.. list-table:: {{ title }}
   :widths: auto
   :header-rows: 0

{% for obj in objs %}
   * - :py:obj:`{{  obj.id }}`
     - {{ obj.summary }} 
{% endfor %}
{% endmacro %}
