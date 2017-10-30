{%- extends 'null.tpl' -%}

{#  
Template to be used to convert Python notebook to Python script:
jupyter nbconvert --to python test.ipynb --template=mytemplate.tpl
#}

{% block header %}
# coding: utf-8

{% endblock header %}

{% block in_prompt %}
{# 
{% if resources.global_content_filter.include_input_prompt -%}
    # In[{{ cell.execution_count if cell.execution_count else ' ' }}]:
{%- endif %}
#}
# <codecell>
{% endblock in_prompt %}

{% block input %}
{#  {{ cell.source | ipython2python }}  -#}
{{ cell.source }}   {# Keep ipython markers #}
{% endblock input %}

{% block markdowncell scoped %}

# <markdowncell>
{%- if cell.metadata.slideshow %}
{%- if cell.metadata.slideshow.slide_type %}
# <{{ cell.metadata.slideshow.slide_type }}>
{#-
  # <slide_type={{ cell.metadata.slideshow.slide_type }}>
  # <slideshow slide_type="{{ cell.metadata.slideshow.slide_type }}">
-#}
{%- endif %}
{%- endif %}
{{ cell.source | wrap_text(width=78) | comment_lines }}
{% endblock markdowncell %}
