#!/usr/bin/env python
# coding: utf-8

# ## Splink Demo - Probabilistic
# 
# 
# 

# In[72]:


pip install splink


# ### Splink Setup

# In[73]:


from splink.spark.jar_location import similarity_jar_location

from pyspark  import SparkContext, SparkConf
from pyspark.sql import SparkSession, types
from pyspark.sql.types import DoubleType, StringType
from pyspark.sql.functions import substring
import splink.spark.comparison_library as cl
import splink.spark.comparison_level_library as cll
import splink.spark.comparison_template_library as ctl
import splink.spark.blocking_rule_library as brl
from splink.comparison import Comparison
from splink.spark.linker import SparkLinker
import os
import pprint

conf = SparkConf()


conf.set("spark.driver.memory", "64g")
conf.set("spark.kryoserializer.buffer.max","1024m")
conf.set("spark.default.parallelism", "40")
spark.conf.set("spark.sql.shuffle.partitions", "40")


# Adds custom similarity functions, which are bundled with Splink
# documented here: https://github.com/moj-analytical-services/splink_scalaudfs
# The jar file needs to be downloaded from the above and uploaded to the synapse workspace
path = similarity_jar_location()

#conf.set('spark.driver.extraClassPath', path) #added by ash
conf.set("spark.jars", path)

sc = SparkContext.getOrCreate(conf=conf)

spark = SparkSession(sc)
spark.sparkContext.setCheckpointDir("./tmp_checkpoints")

os.makedirs("/tmp/Temp", exist_ok=True)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[ ]:


print(path)


# ### Data Matching Function Setup

# In[74]:


spark.udf.registerJavaFunction('jaro_winkler', 'uk.gov.moj.dash.linkage.JaroWinklerSimilarity',DoubleType())                              
spark.udf.registerJavaFunction('jaccard_sim', 'uk.gov.moj.dash.linkage.JaccardSimilarity',DoubleType())                          
spark.udf.registerJavaFunction('cosine_distance', 'uk.gov.moj.dash.linkage.CosineDistance',DoubleType())
spark.udf.registerJavaFunction('sqlEscape', 'uk.gov.moj.dash.linkage.sqlEscape',StringType())                        
spark.udf.registerJavaFunction('levdamerau_distance', 'uk.gov.moj.dash.linkage.LevDamerauDistance',DoubleType())   
spark.udf.registerJavaFunction('jaro_sim', 'uk.gov.moj.dash.linkage.JaroSimilarity',DoubleType())   


# ### Import Demo Dataset

# In[75]:


#https://moj-analytical-services.github.io/splink/datasets.html

from splink.datasets import splink_datasets

#df = splink_datasets.fake_1000
#columns: ['unique_id', 'first_name', 'surname', 'dob', 'city', 'email', 'cluster']

pdf = splink_datasets.historical_50k
df = spark.createDataFrame(pdf)
#columns: ['unique_id','cluster','full_name','first_and_surname','first_name','surname','dob','birth_place','postcode_fake','gender','occupation']

#list(df.columns.values)
#pdf.head(5)
#df.limit(5).show()


# ### Linkage settings

# In[ ]:


settings = {
    "link_type": "dedupe_only",
    "blocking_rules_to_generate_predictions": [
        brl.block_on(["substr(dob,1,4)","postcode_fake"]),
        brl.block_on(["dob","lower(substr(first_name,1,1))"]),
        brl.block_on(["lower(surname)","postcode_fake"]),
        brl.block_on(["lower(surname)","lower(first_name)"]),
    ],
    "comparisons": [
        ctl.name_comparison(
            "first_name",
            set_to_lowercase = True,
            include_exact_match_level=True,
            damerau_levenshtein_thresholds=[1,2],
            jaro_winkler_thresholds=[0.9, 0.8],
            term_frequency_adjustments=False
            ),
        ctl.name_comparison(
            "surname",
            set_to_lowercase = True,
            include_exact_match_level=True,
            damerau_levenshtein_thresholds=[1,2],
            jaro_winkler_thresholds=[0.9, 0.8],
            term_frequency_adjustments=False
            ),
        ctl.date_comparison("dob", 
            cast_strings_to_date=True,
            levenshtein_thresholds=[2],
            damerau_levenshtein_thresholds=[],
            datediff_thresholds=[1, 1],
            datediff_metrics=["month", "year"],
            ),
        cl.exact_match("birth_place"),
        ctl.postcode_comparison("postcode_fake", set_to_lowercase = True),
        cl.exact_match("gender"),
        cl.exact_match("occupation")
    ],
    "retain_matching_columns": True,
    "retain_intermediate_calculation_columns": True,
    #"additional_columns_to_retain": ["Street","Locality","Town","County"]
    "em_convergence": 0.001
}


# ### Create the linkage model

# In[76]:


linker = SparkLinker(df, settings)

linker.profile_columns(
    ["first_name", "surname", "postcode_fake", "substr(dob, 1,4)"], top_n=10, bottom_n=5
)


# ### Calculate probability two random records match
# 

# In[77]:


deterministic_rules = [
    "l.first_name = r.first_name and l.surname = r.surname and l.dob = r.dob",
    "l.first_name = r.first_name and l.surname = r.surname and l.postcode_fake = r.postcode_fake"
]

linker.estimate_probability_two_random_records_match(deterministic_rules, recall=0.70)


# ### Number of comparisons generated by blocking rules

# In[78]:


linker.cumulative_num_comparisons_from_blocking_rules_chart()


# ### Random sampling to calculate u values (known non-matches)

# In[79]:


linker.estimate_u_using_random_sampling(max_pairs=1e7)


# ### Deterministic rules to calculate m values (known matches)

# In[80]:


training_blocking_rule = brl.block_on(["surname","first_name","dob"])
training_session_1 = linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)

training_blocking_rule = brl.block_on(["surname","first_name","postcode_fake"])
training_session_2 = linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)

training_blocking_rule = brl.block_on(["first_name","postcode_fake","dob"])
training_session_3 = linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)

training_blocking_rule = brl.block_on(["surname","postcode_fake","dob"])
training_session_4 = linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)



# In[81]:


training_blocking_rule = brl.block_on(["surname","birth_place","occupation"])
training_session_5 = linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)


# In[82]:


training_blocking_rule = brl.block_on(["first_name","birth_place","occupation"])
training_session_6 = linker.estimate_parameters_using_expectation_maximisation(training_blocking_rule)


# In[83]:


linker.match_weights_chart()
#linker.m_u_parameters_chart()


# In[84]:


linker.unlinkables_chart()


# In[85]:


results = linker.predict(threshold_match_probability=0.9)


# In[86]:


rdf = results.as_pandas_dataframe(limit=100)
rdf


# ### Cluster to generate distinct output list

# In[87]:


clusters = linker.cluster_pairwise_predictions_at_threshold(results, threshold_match_probability=0.9)
#cdf = clusters.as_pandas_dataframe()
#clusters.as_pandas_dataframe(limit=5)
spark.createDataFrame(clusters.as_pandas_dataframe()).createOrReplaceTempView("clusters")


# In[88]:


get_ipython().run_cell_magic('sql', '', 'select count(1)\r\nfrom\r\n(select distinct cluster_id\r\nfrom clusters)\r\n--limit 10\n')


# ### M & U Analysis

# In[89]:


from splink.charts import waterfall_chart
records_to_plot = rdf.to_dict(orient="records")
linker.waterfall_chart(records_to_plot, filter_nulls=False)

