{#
   We show all the class's methods and attributes on the same page. By default, we document
   all methods, including those defined by parent classes.
-#}

{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :show-inheritance:

{% block attributes_summary %}
  {% set wanted_attributes = (attributes | reject('in', inherited_members) | list) %}
  {% if wanted_attributes %}
   .. rubric:: Attributes
    {% for item in wanted_attributes %}
   .. autoattribute:: {{ item }}
    {%- endfor %}
  {% endif %}
  {% set inherited_attributes = (attributes | select('in', inherited_members) | list) %}
  {% if inherited_attributes %}
   .. rubric:: Inherited Attributes
    {% for item in inherited_attributes %}
   .. autoattribute:: {{ item }}
    {%- endfor %}
  {% endif %}
{% endblock -%}

{% block methods_summary %}
  {% set wanted_methods = (methods | reject('==', '__init__') | reject('in', inherited_members) | list) %}
  {% if wanted_methods %}
   .. rubric:: Methods
    {% for item in wanted_methods %}
   .. automethod:: {{ item }}
    {%- endfor %}
  {% endif %}
  {% set inherited_methods = (methods | reject('==', '__init__') | select('in', inherited_members) | list) %}
  {% if inherited_methods %}
   .. rubric:: Inherited Methods
    {% for item in inherited_methods %}
   .. automethod:: {{ item }}
    {%- endfor %}
  {% endif %}
{% endblock %}
