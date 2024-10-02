#!/usr/bin/env python
# coding: utf-8

# ## Splink Comparisons
# 
# 
# 

# In[ ]:


pip install splink


# In[ ]:


import splink.spark.comparison_library as cl
import splink.spark.comparison_level_library as cll
import splink.spark.comparison_template_library as ctl
from splink.comparison import Comparison

import pprint


# ## Exact Match
# Create an exact match comparison on a single field.
# 
# We can use human_readable_description to get a simle description of the levels involved.
# 
# Note that exact match consists of three levels, null handling, exact matches and all other outcomes (in this case non matches).

# In[ ]:


MyExactComparison = cl.exact_match("Email")
print(MyExactComparison.human_readable_description)


# the comparison functions are helpers that build dictionaries in the base comparison format.
# 
# .as_dict() allows us to see the raw format (useful if you want to modify it further).

# In[ ]:


pprint.pprint(MyExactComparison.as_dict())


# ## More Complex Comparisons
# Comparison templates are provided for common field types that need more complex sets of comparisons such as date fields.

# In[ ]:


MyDateComparison = ctl.date_comparison("date_of_birth")
print( MyDateComparison.human_readable_description)


# Comparison templates can often be customised with additional parameters.

# In[ ]:


MyDateComparison2 = ctl.date_comparison("date_of_birth",
                    damerau_levenshtein_thresholds=[],
                    levenshtein_thresholds=[1,2],
                    datediff_thresholds=[1, 2, 1],
                    datediff_metrics=["month","month", "year"])
print( MyDateComparison2.human_readable_description)


# ## Comparison Levels
# For a finer level of control comparison levels can be used to build a comparison.
# 
# Here we re-create the exact match comparison.

# In[ ]:


comparison_exact_name = {
    "output_column_name": "first_name",
    "comparison_description": "First name exact match",
    "comparison_levels": [
        cll.null_level("first_name"),
        cll.exact_match_level("first_name"),
        cll.else_level(),
    ],
}

print(Comparison(comparison_exact_name).human_readable_description)


# In[ ]:


comparison_names = {
    "output_column_name": "full_name",
    "comparison_description": "Full name matches",
    "comparison_levels": [
        cll.and_(
            cll.null_level("first_name"),
            cll.null_level("last_name"),
            label_for_charts = "Null Level",
            is_null_level = True
        ),
        cll.and_(
            cll.exact_match_level("first_name"),
            cll.exact_match_level("last_name"),
            label_for_charts = "first AND last name"
        ),
        cll.and_(
            cll.exact_match_level("left(first_name,1)"),
            cll.exact_match_level("last_name"),
            label_for_charts = "first initial AND last name"
        ),
        cll.or_(
            cll.exact_match_level("first_name"),
            cll.exact_match_level("last_name"),
            label_for_charts = "first OR last name"
        ),
        cll.else_level(),
    ],
}

print(Comparison(comparison_names).human_readable_description)

