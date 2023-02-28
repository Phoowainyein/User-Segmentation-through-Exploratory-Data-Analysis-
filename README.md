# User-Segmentation-through-Exploratory-Data-Analysis-

---
jupyter:
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.7.12
  nbformat: 4
  nbformat_minor: 5
  papermill:
    default_parameters: {}
    duration: 23.162013
    end_time: "2023-02-20T14:16:35.712121"
    environment_variables: {}
    input_path: \_\_notebook\_\_.ipynb
    output_path: \_\_notebook\_\_.ipynb
    parameters: {}
    start_time: "2023-02-20T14:16:12.550108"
    version: 2.3.4
---

::: {#321c549a .cell .code execution_count="1" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:20.346678Z\",\"iopub.status.busy\":\"2023-02-20T14:16:20.345842Z\",\"iopub.status.idle\":\"2023-02-20T14:16:23.269680Z\",\"shell.execute_reply\":\"2023-02-20T14:16:23.268688Z\"}" papermill="{\"duration\":2.938551,\"end_time\":\"2023-02-20T14:16:23.272094\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:20.333543\",\"status\":\"completed\"}" tags="[]"}
``` python
# Imports and settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno 
import plotly.express as px
#settings
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)  
```

::: {.output .display_data}
```{=html}
        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-2.18.0.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        
```
:::
:::

::: {#c897806c .cell .markdown papermill="{\"duration\":6.635e-3,\"end_time\":\"2023-02-20T14:16:23.285844\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:23.279209\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family:newtimeroman;font-size:30px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Assignment for Data Analyst
Intern, Wolt Market team </p>
```
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family: cursive;font-size:30px; padding : 10px ;border-radius: 15px ; display: inline-block;">
WOLT </p>
```
:::

::: {#2cec1aa9 .cell .markdown papermill="{\"duration\":6.456e-3,\"end_time\":\"2023-02-20T14:16:23.298996\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:23.292540\",\"status\":\"completed\"}" tags="[]"}
![title](vertopal_952abda08627499e997a89b2f4aae7b1/d0c612c80558e4124678eb19caadcd7acf03b136.jpg)
:::

::: {#fdc24ef9 .cell .markdown papermill="{\"duration\":6.373e-3,\"end_time\":\"2023-02-20T14:16:23.312198\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:23.305825\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family: Gill Sans, sans-serif;font-size: 25px; padding : 10px ;border-radius: 15px ; display: block;">
Your assignment is to create a user segmentation that helps Wolt understand what type of
users we have. While doing this, please familiarize yourself with the dataset and show us your
excellent exploratory data analysis skills. Remember to justify your segmentation approach so
that we understand why the way you did it is better than an arbitrary solution by a non-data
analyst who can do some slicing-and-dicing with the data.
</p>
```
:::

::: {#65b59033 .cell .code execution_count="2" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:23.327262Z\",\"iopub.status.busy\":\"2023-02-20T14:16:23.326338Z\",\"iopub.status.idle\":\"2023-02-20T14:16:23.523063Z\",\"shell.execute_reply\":\"2023-02-20T14:16:23.522036Z\"}" papermill="{\"duration\":0.206845,\"end_time\":\"2023-02-20T14:16:23.525416\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:23.318571\",\"status\":\"completed\"}" tags="[]"}
``` python
data = pd.read_csv('/kaggle/input/woltassignmentfordataanalystintern/dataset_for_analyst_assignment_20201120.csv')
data.sample(5)
```

::: {.output .execute_result execution_count="2"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>REGISTRATION_DATE</th>
      <th>REGISTRATION_COUNTRY</th>
      <th>PURCHASE_COUNT</th>
      <th>PURCHASE_COUNT_DELIVERY</th>
      <th>PURCHASE_COUNT_TAKEAWAY</th>
      <th>FIRST_PURCHASE_DAY</th>
      <th>LAST_PURCHASE_DAY</th>
      <th>USER_ID</th>
      <th>BREAKFAST_PURCHASES</th>
      <th>LUNCH_PURCHASES</th>
      <th>EVENING_PURCHASES</th>
      <th>DINNER_PURCHASES</th>
      <th>LATE_NIGHT_PURCHASES</th>
      <th>TOTAL_PURCHASES_EUR</th>
      <th>DISTINCT_PURCHASE_VENUE_COUNT</th>
      <th>MIN_PURCHASE_VALUE_EUR</th>
      <th>MAX_PURCHASE_VALUE_EUR</th>
      <th>AVG_PURCHASE_VALUE_EUR</th>
      <th>PREFERRED_DEVICE</th>
      <th>IOS_PURCHASES</th>
      <th>WEB_PURCHASES</th>
      <th>ANDROID_PURCHASES</th>
      <th>PREFERRED_RESTAURANT_TYPES</th>
      <th>USER_HAS_VALID_PAYMENT_METHOD</th>
      <th>MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE</th>
      <th>MOST_COMMON_WEEKDAY_TO_PURCHASE</th>
      <th>AVG_DAYS_BETWEEN_PURCHASES</th>
      <th>MEDIAN_DAYS_BETWEEN_PURCHASES</th>
      <th>AVERAGE_DELIVERY_DISTANCE_KMS</th>
      <th>PURCHASE_COUNT_BY_STORE_TYPE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21448</th>
      <td>2019-09-29 00:00:00.000</td>
      <td>DNK</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21449</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ios</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>{\n  "General merchandise": 0,\n  "Grocery": 0,\n  "Pet supplies": 0,\n  "Restaurant": 0,\n  "Retail store": 0\n}</td>
    </tr>
    <tr>
      <th>14030</th>
      <td>2019-09-20 00:00:00.000</td>
      <td>FIN</td>
      <td>15</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>2020-04-06 00:00:00.000</td>
      <td>2020-10-03 00:00:00.000</td>
      <td>14031</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>379.500</td>
      <td>6.0</td>
      <td>16.224</td>
      <td>41.656</td>
      <td>25.300</td>
      <td>ios</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>False</td>
      <td>16.0</td>
      <td>7.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>1.558</td>
      <td>{\n  "General merchandise": 1,\n  "Grocery": 7,\n  "Pet supplies": 0,\n  "Restaurant": 7,\n  "Retail store": 0\n}</td>
    </tr>
    <tr>
      <th>8894</th>
      <td>2019-09-13 00:00:00.000</td>
      <td>DNK</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8895</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ios</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>False</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>{\n  "General merchandise": 0,\n  "Grocery": 0,\n  "Pet supplies": 0,\n  "Restaurant": 0,\n  "Retail store": 0\n}</td>
    </tr>
    <tr>
      <th>21383</th>
      <td>2019-09-29 00:00:00.000</td>
      <td>GRC</td>
      <td>2</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2019-09-29 00:00:00.000</td>
      <td>2019-10-21 00:00:00.000</td>
      <td>21384</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>33.396</td>
      <td>2.0</td>
      <td>10.140</td>
      <td>23.368</td>
      <td>16.192</td>
      <td>ios</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>True</td>
      <td>19.0</td>
      <td>7.0</td>
      <td>22.0</td>
      <td>22.0</td>
      <td>2.330</td>
      <td>{\n  "General merchandise": 0,\n  "Grocery": 0,\n  "Pet supplies": 0,\n  "Restaurant": 2,\n  "Retail store": 0\n}</td>
    </tr>
    <tr>
      <th>19941</th>
      <td>2019-09-28 00:00:00.000</td>
      <td>DNK</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2019-09-28 00:00:00.000</td>
      <td>2019-09-28 00:00:00.000</td>
      <td>19942</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>30.360</td>
      <td>1.0</td>
      <td>30.420</td>
      <td>30.480</td>
      <td>30.360</td>
      <td>web</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>True</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.520</td>
      <td>{\n  "General merchandise": 0,\n  "Grocery": 0,\n  "Pet supplies": 0,\n  "Restaurant": 1,\n  "Retail store": 0\n}</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#f8a1ae9c .cell .markdown papermill="{\"duration\":6.968e-3,\"end_time\":\"2023-02-20T14:16:23.539672\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:23.532704\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Dataset Exploration: Uncovering the Insights
</p>
```
:::

::: {#14bd63ae .cell .code execution_count="3" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:23.555349Z\",\"iopub.status.busy\":\"2023-02-20T14:16:23.554886Z\",\"iopub.status.idle\":\"2023-02-20T14:16:23.584760Z\",\"shell.execute_reply\":\"2023-02-20T14:16:23.583505Z\"}" papermill="{\"duration\":4.2378e-2,\"end_time\":\"2023-02-20T14:16:23.589294\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:23.546916\",\"status\":\"completed\"}" tags="[]"}
``` python
data.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 21983 entries, 0 to 21982
    Data columns (total 30 columns):
     #   Column                                   Non-Null Count  Dtype  
    ---  ------                                   --------------  -----  
     0   REGISTRATION_DATE                        21983 non-null  object 
     1   REGISTRATION_COUNTRY                     21983 non-null  object 
     2   PURCHASE_COUNT                           21983 non-null  int64  
     3   PURCHASE_COUNT_DELIVERY                  12028 non-null  float64
     4   PURCHASE_COUNT_TAKEAWAY                  12028 non-null  float64
     5   FIRST_PURCHASE_DAY                       11964 non-null  object 
     6   LAST_PURCHASE_DAY                        12027 non-null  object 
     7   USER_ID                                  21983 non-null  int64  
     8   BREAKFAST_PURCHASES                      12028 non-null  float64
     9   LUNCH_PURCHASES                          12028 non-null  float64
     10  EVENING_PURCHASES                        12028 non-null  float64
     11  DINNER_PURCHASES                         12028 non-null  float64
     12  LATE_NIGHT_PURCHASES                     12028 non-null  float64
     13  TOTAL_PURCHASES_EUR                      12028 non-null  float64
     14  DISTINCT_PURCHASE_VENUE_COUNT            12028 non-null  float64
     15  MIN_PURCHASE_VALUE_EUR                   12028 non-null  float64
     16  MAX_PURCHASE_VALUE_EUR                   12028 non-null  float64
     17  AVG_PURCHASE_VALUE_EUR                   12028 non-null  float64
     18  PREFERRED_DEVICE                         21910 non-null  object 
     19  IOS_PURCHASES                            12028 non-null  float64
     20  WEB_PURCHASES                            12028 non-null  float64
     21  ANDROID_PURCHASES                        12028 non-null  float64
     22  PREFERRED_RESTAURANT_TYPES               2694 non-null   object 
     23  USER_HAS_VALID_PAYMENT_METHOD            21983 non-null  bool   
     24  MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE  12028 non-null  float64
     25  MOST_COMMON_WEEKDAY_TO_PURCHASE          12028 non-null  float64
     26  AVG_DAYS_BETWEEN_PURCHASES               7832 non-null   float64
     27  MEDIAN_DAYS_BETWEEN_PURCHASES            7832 non-null   float64
     28  AVERAGE_DELIVERY_DISTANCE_KMS            12028 non-null  float64
     29  PURCHASE_COUNT_BY_STORE_TYPE             21983 non-null  object 
    dtypes: bool(1), float64(20), int64(2), object(7)
    memory usage: 4.9+ MB
:::
:::

::: {#789d0391 .cell .markdown papermill="{\"duration\":6.564e-3,\"end_time\":\"2023-02-20T14:16:23.604102\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:23.597538\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Identifying Missing Values
</p>
```
:::

::: {#897e391d .cell .code execution_count="4" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:23.619849Z\",\"iopub.status.busy\":\"2023-02-20T14:16:23.619136Z\",\"iopub.status.idle\":\"2023-02-20T14:16:27.060198Z\",\"shell.execute_reply\":\"2023-02-20T14:16:27.059190Z\"}" papermill="{\"duration\":3.45189,\"end_time\":\"2023-02-20T14:16:27.062978\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:23.611088\",\"status\":\"completed\"}" tags="[]"}
``` python
msno.bar(data,color='#00c2e8',figsize=(10,5), fontsize=12)
```

::: {.output .execute_result execution_count="4"}
    <AxesSubplot:>
:::

::: {.output .display_data}
![](vertopal_952abda08627499e997a89b2f4aae7b1/3779bbf0be92aeac5e4b39e5f38b4c22ac30771a.png)
:::
:::

::: {#86158480 .cell .markdown papermill="{\"duration\":1.3667e-2,\"end_time\":\"2023-02-20T14:16:27.091535\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.077868\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:white; color :#00c2e8 ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
This bar chart  displays the number of non-missing values (in blue) and missing values (in white) for each column in our dataset. This allows us  to quickly identify which columns have missing values and the extent of the missing data.
</p>
```
:::

::: {#27e22fca .cell .code execution_count="5" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:27.121374Z\",\"iopub.status.busy\":\"2023-02-20T14:16:27.120912Z\",\"iopub.status.idle\":\"2023-02-20T14:16:27.131004Z\",\"shell.execute_reply\":\"2023-02-20T14:16:27.129480Z\"}" papermill="{\"duration\":3.0812e-2,\"end_time\":\"2023-02-20T14:16:27.133934\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.103122\",\"status\":\"completed\"}" tags="[]"}
``` python
print(data.columns)
```

::: {.output .stream .stdout}
    Index(['REGISTRATION_DATE', 'REGISTRATION_COUNTRY', 'PURCHASE_COUNT', 'PURCHASE_COUNT_DELIVERY', 'PURCHASE_COUNT_TAKEAWAY', 'FIRST_PURCHASE_DAY', 'LAST_PURCHASE_DAY', 'USER_ID', 'BREAKFAST_PURCHASES', 'LUNCH_PURCHASES', 'EVENING_PURCHASES', 'DINNER_PURCHASES', 'LATE_NIGHT_PURCHASES', 'TOTAL_PURCHASES_EUR', 'DISTINCT_PURCHASE_VENUE_COUNT', 'MIN_PURCHASE_VALUE_EUR', 'MAX_PURCHASE_VALUE_EUR', 'AVG_PURCHASE_VALUE_EUR', 'PREFERRED_DEVICE', 'IOS_PURCHASES', 'WEB_PURCHASES', 'ANDROID_PURCHASES', 'PREFERRED_RESTAURANT_TYPES', 'USER_HAS_VALID_PAYMENT_METHOD', 'MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE', 'MOST_COMMON_WEEKDAY_TO_PURCHASE', 'AVG_DAYS_BETWEEN_PURCHASES', 'MEDIAN_DAYS_BETWEEN_PURCHASES', 'AVERAGE_DELIVERY_DISTANCE_KMS', 'PURCHASE_COUNT_BY_STORE_TYPE'], dtype='object')
:::
:::

::: {#5555b639 .cell .code execution_count="6" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:27.174636Z\",\"iopub.status.busy\":\"2023-02-20T14:16:27.173921Z\",\"iopub.status.idle\":\"2023-02-20T14:16:27.215268Z\",\"shell.execute_reply\":\"2023-02-20T14:16:27.209189Z\"}" papermill="{\"duration\":6.1598e-2,\"end_time\":\"2023-02-20T14:16:27.219282\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.157684\",\"status\":\"completed\"}" tags="[]"}
``` python
print("-"*50)
print("Missing values of each column")
print("-"*50)
print(data.isna().sum())
print("-"*50)
print("Total missing values:",data.isna().sum().sum())
```

::: {.output .stream .stdout}
    --------------------------------------------------
    Missing values of each column
    --------------------------------------------------
    REGISTRATION_DATE                              0
    REGISTRATION_COUNTRY                           0
    PURCHASE_COUNT                                 0
    PURCHASE_COUNT_DELIVERY                     9955
    PURCHASE_COUNT_TAKEAWAY                     9955
    FIRST_PURCHASE_DAY                         10019
    LAST_PURCHASE_DAY                           9956
    USER_ID                                        0
    BREAKFAST_PURCHASES                         9955
    LUNCH_PURCHASES                             9955
    EVENING_PURCHASES                           9955
    DINNER_PURCHASES                            9955
    LATE_NIGHT_PURCHASES                        9955
    TOTAL_PURCHASES_EUR                         9955
    DISTINCT_PURCHASE_VENUE_COUNT               9955
    MIN_PURCHASE_VALUE_EUR                      9955
    MAX_PURCHASE_VALUE_EUR                      9955
    AVG_PURCHASE_VALUE_EUR                      9955
    PREFERRED_DEVICE                              73
    IOS_PURCHASES                               9955
    WEB_PURCHASES                               9955
    ANDROID_PURCHASES                           9955
    PREFERRED_RESTAURANT_TYPES                 19289
    USER_HAS_VALID_PAYMENT_METHOD                  0
    MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE     9955
    MOST_COMMON_WEEKDAY_TO_PURCHASE             9955
    AVG_DAYS_BETWEEN_PURCHASES                 14151
    MEDIAN_DAYS_BETWEEN_PURCHASES              14151
    AVERAGE_DELIVERY_DISTANCE_KMS               9955
    PURCHASE_COUNT_BY_STORE_TYPE                   0
    dtype: int64
    --------------------------------------------------
    Total missing values: 246829
:::
:::

::: {#ee24473b .cell .markdown papermill="{\"duration\":1.6439e-2,\"end_time\":\"2023-02-20T14:16:27.252417\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.235978\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Handling Missing Values
</p>
```
:::

::: {#9c3a5f2a .cell .markdown papermill="{\"duration\":1.5516e-2,\"end_time\":\"2023-02-20T14:16:27.284317\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.268801\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:white; color :#00c2e8 ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
What to do with missing values ? <br> My approach is to remove excessive amounts of missing values from most columns  and then using imputation to fill  in the remaining missing values .Based on the information above,we can clearly identifed 9955 values are missing in  most column,I'll be carefully taking out those rows.
</p>
```
:::

::: {#5b7111cf .cell .code execution_count="7" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:27.316813Z\",\"iopub.status.busy\":\"2023-02-20T14:16:27.316379Z\",\"iopub.status.idle\":\"2023-02-20T14:16:27.345013Z\",\"shell.execute_reply\":\"2023-02-20T14:16:27.344118Z\"}" papermill="{\"duration\":4.7929e-2,\"end_time\":\"2023-02-20T14:16:27.347747\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.299818\",\"status\":\"completed\"}" tags="[]"}
``` python

data.dropna(subset=['REGISTRATION_DATE', 'REGISTRATION_COUNTRY', 'PURCHASE_COUNT', 'PURCHASE_COUNT_DELIVERY', 'PURCHASE_COUNT_TAKEAWAY', 'FIRST_PURCHASE_DAY', 'LAST_PURCHASE_DAY', 'USER_ID',
                    'BREAKFAST_PURCHASES', 'LUNCH_PURCHASES', 'EVENING_PURCHASES', 'DINNER_PURCHASES', 'LATE_NIGHT_PURCHASES', 'TOTAL_PURCHASES_EUR', 'DISTINCT_PURCHASE_VENUE_COUNT', 
                    'MIN_PURCHASE_VALUE_EUR', 'MAX_PURCHASE_VALUE_EUR', 'AVG_PURCHASE_VALUE_EUR', 'PREFERRED_DEVICE', 'IOS_PURCHASES', 'WEB_PURCHASES', 'ANDROID_PURCHASES', 'USER_HAS_VALID_PAYMENT_METHOD',
                    'MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE', 'MOST_COMMON_WEEKDAY_TO_PURCHASE', 'AVERAGE_DELIVERY_DISTANCE_KMS', 'PURCHASE_COUNT_BY_STORE_TYPE'],inplace=True)
```
:::

::: {#7b75f2d8 .cell .code execution_count="8" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:27.382375Z\",\"iopub.status.busy\":\"2023-02-20T14:16:27.381892Z\",\"iopub.status.idle\":\"2023-02-20T14:16:27.403432Z\",\"shell.execute_reply\":\"2023-02-20T14:16:27.402589Z\"}" papermill="{\"duration\":4.1232e-2,\"end_time\":\"2023-02-20T14:16:27.406006\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.364774\",\"status\":\"completed\"}" tags="[]"}
``` python
data['total_days']= pd.to_datetime(data['LAST_PURCHASE_DAY'])- pd.to_datetime( data['FIRST_PURCHASE_DAY'])
data['TOTAL_DAYS_BETWEEN_PURCHASES']=data['total_days'].dt.days
data.drop(columns=['total_days'],inplace=True)
#data.head(1)
```
:::

::: {#5f94dbf1 .cell .markdown papermill="{\"duration\":1.6093e-2,\"end_time\":\"2023-02-20T14:16:27.438728\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.422635\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:white; color :#00c2e8 ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Based on the available information, it appears that the rows containing null values for columns <b>'AVG_DAYS_BETWEEN_PURCHASES','MEDIAN_DAYS_BETWEEN_PURCHASES' </b> are a result of  no difference between<b>'FIRST_PURCHASE_DAY'</b> and <b>'LAST_PURCHASE_DAY'</b> .This indicates that the first and last purchase dates happen to be identical, which in turn is resulting in a null value when attempting to calculate the average.As a result, I will replace null values with 0.
</p>
```
:::

::: {#9167175d .cell .code execution_count="9" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:27.472378Z\",\"iopub.status.busy\":\"2023-02-20T14:16:27.471908Z\",\"iopub.status.idle\":\"2023-02-20T14:16:27.478244Z\",\"shell.execute_reply\":\"2023-02-20T14:16:27.477354Z\"}" papermill="{\"duration\":2.7512e-2,\"end_time\":\"2023-02-20T14:16:27.482401\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.454889\",\"status\":\"completed\"}" tags="[]"}
``` python
data['AVG_DAYS_BETWEEN_PURCHASES'].fillna(0, inplace=True)
data['MEDIAN_DAYS_BETWEEN_PURCHASES'].fillna(0, inplace=True)
```
:::

::: {#2f060fff .cell .markdown papermill="{\"duration\":1.5762e-2,\"end_time\":\"2023-02-20T14:16:27.514260\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.498498\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:white; color :#00c2e8 ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Lastly, we are left with a categorical column where a large percentage of values are missing.In general, imputing missing values for categorical data is more challenging than for numerical data, and it's important to carefully consider when dealing with this type of categorial column.<br><br>
When significant amount of data are missing from this categorial column ,the least desirable action we can take is deleting all the rows.<br>For example, if a categorical column has a missing value, one common approach is to impute the most frequent category and  fill up the missing value. However, this can be problematic if the mode is not representative of the true underlying distribution of the data.<br><br>
Overall, imputing for categorical columns with a large amount of missing data is not a good idea because there may not be enough information available to accurately fill in the missing values.Therefore , my approach is to add more columns (boolean values)  With each restaurant type that user indicated their preference.
You can find my solution in the cells below.</p>
```
:::

::: {#cecf27f7 .cell .code execution_count="10" _kg_hide-input="false" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:27.548947Z\",\"iopub.status.busy\":\"2023-02-20T14:16:27.548398Z\",\"iopub.status.idle\":\"2023-02-20T14:16:27.733909Z\",\"shell.execute_reply\":\"2023-02-20T14:16:27.732905Z\"}" papermill="{\"duration\":0.206212,\"end_time\":\"2023-02-20T14:16:27.737056\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.530844\",\"status\":\"completed\"}" tags="[]"}
``` python
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('\n', '')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace(' ', '')

data["preferred_American_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "american" in str(row).lower() else False)
data["preferred_Japanese_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "japanese" in str(row).lower() else False)
data["preferred_Italian_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "italian" in str(row).lower() else False)
data["preferred_Mexican_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "mexican" in str(row).lower() else False)
data["preferred_Indian_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "indian" in str(row).lower() else False)
data["preferred_Middleeastern_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "middleeastern" in str(row).lower() else False)
data["preferred_Korean_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "korean" in str(row).lower() else False)
data["preferred_Thai_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "thai" in str(row).lower() else False)
data["preferred_Vietnamese_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "vietnamese" in str(row).lower() else False)
data["preferred_Hawaiian_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "hawaiian" in str(row).lower() else False)
data["preferred_Greek_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "greek" in str(row).lower() else False)
data["preferred_Spanish_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "spanish" in str(row).lower() else False)
data["preferred_Nepalese_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "nepalese" in str(row).lower() else False)
data["preferred_Chinese_restaurant"] = data["PREFERRED_RESTAURANT_TYPES"].apply(lambda row: True if "chinese" in str(row).lower() else False)

```
:::

::: {#ecc1d642 .cell .code execution_count="11" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:27.773824Z\",\"iopub.status.busy\":\"2023-02-20T14:16:27.773302Z\",\"iopub.status.idle\":\"2023-02-20T14:16:27.879059Z\",\"shell.execute_reply\":\"2023-02-20T14:16:27.878166Z\"}" papermill="{\"duration\":0.126085,\"end_time\":\"2023-02-20T14:16:27.881752\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.755667\",\"status\":\"completed\"}" tags="[]"}
``` python
#I would like to validate that any preffered resturatnt types  are  not left by changing values 
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('american', '0')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('japanese', '1')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('italian', '2')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('mexican', '3')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('indian', '4')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('middleeastern', '5')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('korean', '6')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('thai', '7')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('vietnamese', '8')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('hawaiian', '9')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('greek', '10')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('spanish', '11')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('nepalese', '12')
data['PREFERRED_RESTAURANT_TYPES'] = data['PREFERRED_RESTAURANT_TYPES'].str.replace('chinese', '13')
#data.PREFERRED_RESTAURANT_TYPES.value_counts()
"""This commented code  confirms that there are precisely 14 preferred restaurant types in this dataset, and that each one has been accurately labeled."
he above comment confirms that there are precisely 14 preferred restaurant types in this dataset, and that each one has been accurately labeled.
"""
#droping the column 
data.drop('PREFERRED_RESTAURANT_TYPES', axis=1,inplace=True)
```
:::

::: {#3fff76f2 .cell .code execution_count="12" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:27.914897Z\",\"iopub.status.busy\":\"2023-02-20T14:16:27.914097Z\",\"iopub.status.idle\":\"2023-02-20T14:16:32.678972Z\",\"shell.execute_reply\":\"2023-02-20T14:16:32.678174Z\"}" papermill="{\"duration\":4.786051,\"end_time\":\"2023-02-20T14:16:32.683829\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:27.897778\",\"status\":\"completed\"}" tags="[]"}
``` python
msno.bar(data,color='#00c2e8',figsize=(10,5), fontsize=12)
```

::: {.output .execute_result execution_count="12"}
    <AxesSubplot:>
:::

::: {.output .display_data}
![](vertopal_952abda08627499e997a89b2f4aae7b1/788f6facbfc0cc8ab29da5c87c04cb1cf469dd7c.png)
:::
:::

::: {#0fc5d7b6 .cell .markdown papermill="{\"duration\":1.5402e-2,\"end_time\":\"2023-02-20T14:16:32.714891\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:32.699489\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:white; color :#00c2e8 ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Finally,There is no missing data in our dataset,we can now proceed with our analysis.
</p>
```
:::

::: {#bb977e26 .cell .markdown papermill="{\"duration\":1.3631e-2,\"end_time\":\"2023-02-20T14:16:32.742744\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:32.729113\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Stastical Summary</p>
```
:::

::: {#8d82c2f4 .cell .markdown papermill="{\"duration\":1.362e-2,\"end_time\":\"2023-02-20T14:16:32.770108\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:32.756488\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family: Arial;font-size: 18px; padding : 10px ;border-radius: 15px ; display: inline-block;">
 •  count: the number of non-null values in each column<br>
 •  mean: the average value of each column<br>
 •  std: the standard deviation of each column<br>
 •  min: the minimum value of each column<br>
 •  25%: the first quartile value (25th percentile) of each column<br>
 •  50%: the second quartile value (50th percentile, or median) of each column<br>
 •  75%: the third quartile value (75th percentile) of each column<br>
 •  max: the maximum value of each column<br>
</p>
```
:::

::: {#792a8fd3 .cell .code execution_count="13" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:32.798897Z\",\"iopub.status.busy\":\"2023-02-20T14:16:32.798539Z\",\"iopub.status.idle\":\"2023-02-20T14:16:32.952488Z\",\"shell.execute_reply\":\"2023-02-20T14:16:32.951354Z\"}" papermill="{\"duration\":0.171746,\"end_time\":\"2023-02-20T14:16:32.955616\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:32.783870\",\"status\":\"completed\"}" tags="[]"}
``` python
data.describe()
```

::: {.output .execute_result execution_count="13"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PURCHASE_COUNT</th>
      <th>PURCHASE_COUNT_DELIVERY</th>
      <th>PURCHASE_COUNT_TAKEAWAY</th>
      <th>USER_ID</th>
      <th>BREAKFAST_PURCHASES</th>
      <th>LUNCH_PURCHASES</th>
      <th>EVENING_PURCHASES</th>
      <th>DINNER_PURCHASES</th>
      <th>LATE_NIGHT_PURCHASES</th>
      <th>TOTAL_PURCHASES_EUR</th>
      <th>DISTINCT_PURCHASE_VENUE_COUNT</th>
      <th>MIN_PURCHASE_VALUE_EUR</th>
      <th>MAX_PURCHASE_VALUE_EUR</th>
      <th>AVG_PURCHASE_VALUE_EUR</th>
      <th>IOS_PURCHASES</th>
      <th>WEB_PURCHASES</th>
      <th>ANDROID_PURCHASES</th>
      <th>MOST_COMMON_HOUR_OF_THE_DAY_TO_PURCHASE</th>
      <th>MOST_COMMON_WEEKDAY_TO_PURCHASE</th>
      <th>AVG_DAYS_BETWEEN_PURCHASES</th>
      <th>MEDIAN_DAYS_BETWEEN_PURCHASES</th>
      <th>AVERAGE_DELIVERY_DISTANCE_KMS</th>
      <th>TOTAL_DAYS_BETWEEN_PURCHASES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.0</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
      <td>11963.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.131489</td>
      <td>5.758673</td>
      <td>0.372816</td>
      <td>11034.527543</td>
      <td>0.194516</td>
      <td>2.383014</td>
      <td>0.495361</td>
      <td>3.035777</td>
      <td>0.0</td>
      <td>176.485423</td>
      <td>3.324668</td>
      <td>24.328795</td>
      <td>41.972444</td>
      <td>31.095800</td>
      <td>2.909053</td>
      <td>1.057260</td>
      <td>2.165176</td>
      <td>11.502466</td>
      <td>4.011368</td>
      <td>41.947839</td>
      <td>36.398813</td>
      <td>5.967768</td>
      <td>148.268495</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.786432</td>
      <td>10.559979</td>
      <td>1.417792</td>
      <td>6378.993651</td>
      <td>1.106832</td>
      <td>5.639530</td>
      <td>1.832122</td>
      <td>5.246879</td>
      <td>0.0</td>
      <td>299.577080</td>
      <td>3.770203</td>
      <td>18.000283</td>
      <td>40.843205</td>
      <td>19.913191</td>
      <td>7.473841</td>
      <td>4.785202</td>
      <td>6.893241</td>
      <td>6.927882</td>
      <td>2.010432</td>
      <td>66.105767</td>
      <td>66.093225</td>
      <td>3.465655</td>
      <td>155.504859</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.012000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.016000</td>
      <td>1.012000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>5534.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>32.384000</td>
      <td>1.000000</td>
      <td>14.196000</td>
      <td>23.368000</td>
      <td>19.228000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.963500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>11039.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>75.900000</td>
      <td>2.000000</td>
      <td>19.266000</td>
      <td>35.560000</td>
      <td>27.324000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>4.000000</td>
      <td>17.000000</td>
      <td>9.000000</td>
      <td>5.940000</td>
      <td>89.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>0.000000</td>
      <td>16510.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.0</td>
      <td>196.328000</td>
      <td>4.000000</td>
      <td>29.406000</td>
      <td>51.816000</td>
      <td>38.456000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>6.000000</td>
      <td>54.000000</td>
      <td>38.000000</td>
      <td>8.981500</td>
      <td>304.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>320.000000</td>
      <td>320.000000</td>
      <td>44.000000</td>
      <td>21983.000000</td>
      <td>52.000000</td>
      <td>171.000000</td>
      <td>71.000000</td>
      <td>104.000000</td>
      <td>0.0</td>
      <td>7979.620000</td>
      <td>71.000000</td>
      <td>338.676000</td>
      <td>3048.000000</td>
      <td>569.756000</td>
      <td>200.000000</td>
      <td>196.000000</td>
      <td>221.000000</td>
      <td>23.000000</td>
      <td>7.000000</td>
      <td>421.000000</td>
      <td>421.000000</td>
      <td>11.999000</td>
      <td>426.000000</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#22814fc1 .cell .markdown papermill="{\"duration\":2.2017e-2,\"end_time\":\"2023-02-20T14:16:33.003503\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:32.981486\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Registrations classified by country</p>
```
:::

::: {#49624534 .cell .code execution_count="14" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:33.048353Z\",\"iopub.status.busy\":\"2023-02-20T14:16:33.047823Z\",\"iopub.status.idle\":\"2023-02-20T14:16:33.137572Z\",\"shell.execute_reply\":\"2023-02-20T14:16:33.136559Z\"}" papermill="{\"duration\":0.115677,\"end_time\":\"2023-02-20T14:16:33.140469\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:33.024792\",\"status\":\"completed\"}" tags="[]"}
``` python
countries=data.REGISTRATION_COUNTRY.value_counts()
countriesbypercentage=data.REGISTRATION_COUNTRY.value_counts(normalize=True).mul(100).round(1).astype(str) + '%' 
cbp=pd.DataFrame({"Numbers/Amount":countries,'Percent' :countriesbypercentage}).rename_axis('Country').reset_index()
cbp.style.hide_index()
```

::: {.output .execute_result execution_count="14"}
```{=html}
<style type="text/css">
</style>
<table id="T_37bde_">
  <thead>
    <tr>
      <th class="col_heading level0 col0" >Country</th>
      <th class="col_heading level0 col1" >Numbers/Amount</th>
      <th class="col_heading level0 col2" >Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_37bde_row0_col0" class="data row0 col0" >FIN</td>
      <td id="T_37bde_row0_col1" class="data row0 col1" >5435</td>
      <td id="T_37bde_row0_col2" class="data row0 col2" >45.4%</td>
    </tr>
    <tr>
      <td id="T_37bde_row1_col0" class="data row1 col0" >DNK</td>
      <td id="T_37bde_row1_col1" class="data row1 col1" >4938</td>
      <td id="T_37bde_row1_col2" class="data row1 col2" >41.3%</td>
    </tr>
    <tr>
      <td id="T_37bde_row2_col0" class="data row2 col0" >GRC</td>
      <td id="T_37bde_row2_col1" class="data row2 col1" >1530</td>
      <td id="T_37bde_row2_col2" class="data row2 col2" >12.8%</td>
    </tr>
    <tr>
      <td id="T_37bde_row3_col0" class="data row3 col0" >NOR</td>
      <td id="T_37bde_row3_col1" class="data row3 col1" >13</td>
      <td id="T_37bde_row3_col2" class="data row3 col2" >0.1%</td>
    </tr>
    <tr>
      <td id="T_37bde_row4_col0" class="data row4 col0" >EST</td>
      <td id="T_37bde_row4_col1" class="data row4 col1" >13</td>
      <td id="T_37bde_row4_col2" class="data row4 col2" >0.1%</td>
    </tr>
    <tr>
      <td id="T_37bde_row5_col0" class="data row5 col0" >HUN</td>
      <td id="T_37bde_row5_col1" class="data row5 col1" >5</td>
      <td id="T_37bde_row5_col2" class="data row5 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row6_col0" class="data row6 col0" >CZE</td>
      <td id="T_37bde_row6_col1" class="data row6 col1" >4</td>
      <td id="T_37bde_row6_col2" class="data row6 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row7_col0" class="data row7 col0" >SWE</td>
      <td id="T_37bde_row7_col1" class="data row7 col1" >4</td>
      <td id="T_37bde_row7_col2" class="data row7 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row8_col0" class="data row8 col0" >POL</td>
      <td id="T_37bde_row8_col1" class="data row8 col1" >4</td>
      <td id="T_37bde_row8_col2" class="data row8 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row9_col0" class="data row9 col0" >ISR</td>
      <td id="T_37bde_row9_col1" class="data row9 col1" >3</td>
      <td id="T_37bde_row9_col2" class="data row9 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row10_col0" class="data row10 col0" >LVA</td>
      <td id="T_37bde_row10_col1" class="data row10 col1" >3</td>
      <td id="T_37bde_row10_col2" class="data row10 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row11_col0" class="data row11 col0" >GBR</td>
      <td id="T_37bde_row11_col1" class="data row11 col1" >2</td>
      <td id="T_37bde_row11_col2" class="data row11 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row12_col0" class="data row12 col0" >FRA</td>
      <td id="T_37bde_row12_col1" class="data row12 col1" >2</td>
      <td id="T_37bde_row12_col2" class="data row12 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row13_col0" class="data row13 col0" >LTU</td>
      <td id="T_37bde_row13_col1" class="data row13 col1" >2</td>
      <td id="T_37bde_row13_col2" class="data row13 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row14_col0" class="data row14 col0" >CAN</td>
      <td id="T_37bde_row14_col1" class="data row14 col1" >1</td>
      <td id="T_37bde_row14_col2" class="data row14 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row15_col0" class="data row15 col0" >DEU</td>
      <td id="T_37bde_row15_col1" class="data row15 col1" >1</td>
      <td id="T_37bde_row15_col2" class="data row15 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row16_col0" class="data row16 col0" >HRV</td>
      <td id="T_37bde_row16_col1" class="data row16 col1" >1</td>
      <td id="T_37bde_row16_col2" class="data row16 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row17_col0" class="data row17 col0" >CYP</td>
      <td id="T_37bde_row17_col1" class="data row17 col1" >1</td>
      <td id="T_37bde_row17_col2" class="data row17 col2" >0.0%</td>
    </tr>
    <tr>
      <td id="T_37bde_row18_col0" class="data row18 col0" >ARE</td>
      <td id="T_37bde_row18_col1" class="data row18 col1" >1</td>
      <td id="T_37bde_row18_col2" class="data row18 col2" >0.0%</td>
    </tr>
  </tbody>
</table>
```
:::
:::

::: {#dafcf945 .cell .code execution_count="15" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:33.185466Z\",\"iopub.status.busy\":\"2023-02-20T14:16:33.185011Z\",\"iopub.status.idle\":\"2023-02-20T14:16:33.888261Z\",\"shell.execute_reply\":\"2023-02-20T14:16:33.887084Z\"}" papermill="{\"duration\":0.727953,\"end_time\":\"2023-02-20T14:16:33.890568\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:33.162615\",\"status\":\"completed\"}" tags="[]"}
``` python
countries= data.REGISTRATION_COUNTRY.value_counts().rename_axis('Countries').reset_index(name='Numbers')
fig = px.bar(countries, x="Countries",y="Numbers",title="Registration Numbers by Country")
fig.update_traces(marker_color='#00c2e8')
fig.show()
```

::: {.output .display_data}
``` json
{"config":{"plotlyServerURL":"https://plot.ly"},"data":[{"alignmentgroup":"True","hovertemplate":"Countries=%{x}<br>Numbers=%{y}<extra></extra>","legendgroup":"","marker":{"color":"#00c2e8","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"textposition":"auto","type":"bar","x":["FIN","DNK","GRC","NOR","EST","HUN","CZE","SWE","POL","ISR","LVA","GBR","FRA","LTU","CAN","DEU","HRV","CYP","ARE"],"xaxis":"x","y":[5435,4938,1530,13,13,5,4,4,4,3,3,2,2,2,1,1,1,1,1],"yaxis":"y"}],"layout":{"barmode":"relative","legend":{"tracegroupgap":0},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":5.0e-2},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Registration Numbers by Country"},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Countries"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Numbers"}}}}
```
:::
:::

::: {#a286d11e .cell .markdown papermill="{\"duration\":1.9308e-2,\"end_time\":\"2023-02-20T14:16:33.930708\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:33.911400\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:white; color :#00c2e8 ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
 Based on the data, the three countries with the highest number of customers are Finland, Denmark, and Greece.</p>
```
:::

::: {#25a5da92 .cell .code execution_count="16" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:33.970846Z\",\"iopub.status.busy\":\"2023-02-20T14:16:33.970532Z\",\"iopub.status.idle\":\"2023-02-20T14:16:33.980253Z\",\"shell.execute_reply\":\"2023-02-20T14:16:33.979077Z\"}" papermill="{\"duration\":3.2177e-2,\"end_time\":\"2023-02-20T14:16:33.982599\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:33.950422\",\"status\":\"completed\"}" tags="[]"}
``` python
data.preferred_American_restaurant.value_counts()
```

::: {.output .execute_result execution_count="16"}
    False    10671
    True      1292
    Name: preferred_American_restaurant, dtype: int64
:::
:::

::: {#17160992 .cell .markdown papermill="{\"duration\":1.9471e-2,\"end_time\":\"2023-02-20T14:16:34.021363\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:34.001892\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
The number of users and their preferred restaurant type</p>
```
:::

::: {#4822c048 .cell .code execution_count="17" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:34.061837Z\",\"iopub.status.busy\":\"2023-02-20T14:16:34.061546Z\",\"iopub.status.idle\":\"2023-02-20T14:16:34.067040Z\",\"shell.execute_reply\":\"2023-02-20T14:16:34.066015Z\"}" papermill="{\"duration\":2.8851e-2,\"end_time\":\"2023-02-20T14:16:34.069332\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:34.040481\",\"status\":\"completed\"}" tags="[]"}
``` python
def generate_value_counts_for_restaurant_style_preferance(df,column_list):
    pieces=[]
    for col in column_list:
        tmp_series = df[col].value_counts()
        tmp_series.name = col
        pieces.append(tmp_series)
    df_value_counts = pd.concat(pieces, axis=1)
    df_value_counts = df_value_counts .rename_axis('yes_I_prefer')
    return df_value_counts
```
:::

::: {#9eaf6484 .cell .code execution_count="18" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:34.109245Z\",\"iopub.status.busy\":\"2023-02-20T14:16:34.108932Z\",\"iopub.status.idle\":\"2023-02-20T14:16:34.130381Z\",\"shell.execute_reply\":\"2023-02-20T14:16:34.129407Z\"}" papermill="{\"duration\":4.3889e-2,\"end_time\":\"2023-02-20T14:16:34.132355\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:34.088466\",\"status\":\"completed\"}" tags="[]"}
``` python

preferred_restaurants=["preferred_American_restaurant", "preferred_Italian_restaurant", "preferred_Japanese_restaurant","preferred_Mexican_restaurant",
                                "preferred_Indian_restaurant","preferred_Middleeastern_restaurant","preferred_Korean_restaurant","preferred_Thai_restaurant"
                                 ,"preferred_Vietnamese_restaurant","preferred_Hawaiian_restaurant","preferred_Greek_restaurant","preferred_Spanish_restaurant",
                                 "preferred_Nepalese_restaurant","preferred_Chinese_restaurant"]

value_counts_res = generate_value_counts_for_restaurant_style_preferance(data,preferred_restaurants)
#value_counts_res = value_counts_res.rename_axis('yes_I_prefer')

value_counts_res 
```

::: {.output .execute_result execution_count="18"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>preferred_American_restaurant</th>
      <th>preferred_Italian_restaurant</th>
      <th>preferred_Japanese_restaurant</th>
      <th>preferred_Mexican_restaurant</th>
      <th>preferred_Indian_restaurant</th>
      <th>preferred_Middleeastern_restaurant</th>
      <th>preferred_Korean_restaurant</th>
      <th>preferred_Thai_restaurant</th>
      <th>preferred_Vietnamese_restaurant</th>
      <th>preferred_Hawaiian_restaurant</th>
      <th>preferred_Greek_restaurant</th>
      <th>preferred_Spanish_restaurant</th>
      <th>preferred_Nepalese_restaurant</th>
      <th>preferred_Chinese_restaurant</th>
    </tr>
    <tr>
      <th>yes_I_prefer</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>False</th>
      <td>10671</td>
      <td>11045</td>
      <td>11311</td>
      <td>11498</td>
      <td>11734</td>
      <td>11739</td>
      <td>11915</td>
      <td>11896</td>
      <td>11910</td>
      <td>11939</td>
      <td>11944</td>
      <td>11962</td>
      <td>11957</td>
      <td>11934</td>
    </tr>
    <tr>
      <th>True</th>
      <td>1292</td>
      <td>918</td>
      <td>652</td>
      <td>465</td>
      <td>229</td>
      <td>224</td>
      <td>48</td>
      <td>67</td>
      <td>53</td>
      <td>24</td>
      <td>19</td>
      <td>1</td>
      <td>6</td>
      <td>29</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#9c610d81 .cell .markdown papermill="{\"duration\":1.9842e-2,\"end_time\":\"2023-02-20T14:16:34.172047\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:34.152205\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:white; color :#00c2e8 ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Based on this data, it appears that 1,292 people have indicated a preference for American-style restaurants, while 912 people prefer Italian restaurants, and so on for other restaurant types."
</p>
```
:::

::: {#5a4f148d .cell .code execution_count="19" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:34.212880Z\",\"iopub.status.busy\":\"2023-02-20T14:16:34.211949Z\",\"iopub.status.idle\":\"2023-02-20T14:16:34.336756Z\",\"shell.execute_reply\":\"2023-02-20T14:16:34.335815Z\"}" papermill="{\"duration\":0.148791,\"end_time\":\"2023-02-20T14:16:34.340075\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:34.191284\",\"status\":\"completed\"}" tags="[]"}
``` python

fig = px.bar(value_counts_res.iloc[1:, :],  y=["preferred_American_restaurant", "preferred_Italian_restaurant", "preferred_Japanese_restaurant","preferred_Mexican_restaurant",
                                "preferred_Indian_restaurant","preferred_Middleeastern_restaurant","preferred_Korean_restaurant","preferred_Thai_restaurant"
                                 ,"preferred_Vietnamese_restaurant","preferred_Hawaiian_restaurant","preferred_Greek_restaurant","preferred_Spanish_restaurant",
                                 "preferred_Nepalese_restaurant","preferred_Chinese_restaurant"] ,title="Restaurant Style Preferences of Users",text_auto=True)
fig.update_layout(barmode='group')
fig.show()
```

::: {.output .display_data}
``` json
{"config":{"plotlyServerURL":"https://plot.ly"},"data":[{"alignmentgroup":"True","hovertemplate":"variable=preferred_American_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_American_restaurant","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"preferred_American_restaurant","offsetgroup":"preferred_American_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[1292],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Italian_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Italian_restaurant","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"preferred_Italian_restaurant","offsetgroup":"preferred_Italian_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[918],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Japanese_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Japanese_restaurant","marker":{"color":"#00cc96","pattern":{"shape":""}},"name":"preferred_Japanese_restaurant","offsetgroup":"preferred_Japanese_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[652],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Mexican_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Mexican_restaurant","marker":{"color":"#ab63fa","pattern":{"shape":""}},"name":"preferred_Mexican_restaurant","offsetgroup":"preferred_Mexican_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[465],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Indian_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Indian_restaurant","marker":{"color":"#FFA15A","pattern":{"shape":""}},"name":"preferred_Indian_restaurant","offsetgroup":"preferred_Indian_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[229],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Middleeastern_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Middleeastern_restaurant","marker":{"color":"#19d3f3","pattern":{"shape":""}},"name":"preferred_Middleeastern_restaurant","offsetgroup":"preferred_Middleeastern_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[224],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Korean_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Korean_restaurant","marker":{"color":"#FF6692","pattern":{"shape":""}},"name":"preferred_Korean_restaurant","offsetgroup":"preferred_Korean_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[48],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Thai_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Thai_restaurant","marker":{"color":"#B6E880","pattern":{"shape":""}},"name":"preferred_Thai_restaurant","offsetgroup":"preferred_Thai_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[67],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Vietnamese_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Vietnamese_restaurant","marker":{"color":"#FF97FF","pattern":{"shape":""}},"name":"preferred_Vietnamese_restaurant","offsetgroup":"preferred_Vietnamese_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[53],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Hawaiian_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Hawaiian_restaurant","marker":{"color":"#FECB52","pattern":{"shape":""}},"name":"preferred_Hawaiian_restaurant","offsetgroup":"preferred_Hawaiian_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[24],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Greek_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Greek_restaurant","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"preferred_Greek_restaurant","offsetgroup":"preferred_Greek_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[19],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Spanish_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Spanish_restaurant","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"preferred_Spanish_restaurant","offsetgroup":"preferred_Spanish_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[1],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Nepalese_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Nepalese_restaurant","marker":{"color":"#00cc96","pattern":{"shape":""}},"name":"preferred_Nepalese_restaurant","offsetgroup":"preferred_Nepalese_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[6],"yaxis":"y"},{"alignmentgroup":"True","hovertemplate":"variable=preferred_Chinese_restaurant<br>yes_I_prefer=%{x}<br>value=%{y}<extra></extra>","legendgroup":"preferred_Chinese_restaurant","marker":{"color":"#ab63fa","pattern":{"shape":""}},"name":"preferred_Chinese_restaurant","offsetgroup":"preferred_Chinese_restaurant","orientation":"v","showlegend":true,"textposition":"auto","texttemplate":"%{y}","type":"bar","x":[true],"xaxis":"x","y":[29],"yaxis":"y"}],"layout":{"barmode":"group","legend":{"title":{"text":"variable"},"tracegroupgap":0},"template":{"data":{"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"choropleth":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"choropleth"}],"contour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"contour"}],"contourcarpet":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"contourcarpet"}],"heatmap":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmap"}],"heatmapgl":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"heatmapgl"}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"histogram2d":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2d"}],"histogram2dcontour":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"histogram2dcontour"}],"mesh3d":[{"colorbar":{"outlinewidth":0,"ticks":""},"type":"mesh3d"}],"parcoords":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"parcoords"}],"pie":[{"automargin":true,"type":"pie"}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"scatter3d":[{"line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatter3d"}],"scattercarpet":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattercarpet"}],"scattergeo":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergeo"}],"scattergl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattergl"}],"scattermapbox":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scattermapbox"}],"scatterpolar":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolar"}],"scatterpolargl":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterpolargl"}],"scatterternary":[{"marker":{"colorbar":{"outlinewidth":0,"ticks":""}},"type":"scatterternary"}],"surface":[{"colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"type":"surface"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}]},"layout":{"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"autotypenumbers":"strict","coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]],"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"geo":{"bgcolor":"white","lakecolor":"white","landcolor":"#E5ECF6","showlakes":true,"showland":true,"subunitcolor":"white"},"hoverlabel":{"align":"left"},"hovermode":"closest","mapbox":{"style":"light"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","gridwidth":2,"linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white"}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"ternary":{"aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"bgcolor":"#E5ECF6","caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"title":{"x":5.0e-2},"xaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2},"yaxis":{"automargin":true,"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","zerolinewidth":2}}},"title":{"text":"Restaurant Style Preferences of Users"},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"yes_I_prefer"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"value"}}}}
```
:::
:::

::: {#e103728a .cell .markdown papermill="{\"duration\":2.6724e-2,\"end_time\":\"2023-02-20T14:16:34.394741\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:34.368017\",\"status\":\"completed\"}" tags="[]"}
Since Finland, Denmark, and Greece have the highest number of customers,
I would also like to display their preferred types of restaurants.
:::

::: {#786773d1 .cell .code execution_count="20" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:34.449735Z\",\"iopub.status.busy\":\"2023-02-20T14:16:34.448986Z\",\"iopub.status.idle\":\"2023-02-20T14:16:34.471231Z\",\"shell.execute_reply\":\"2023-02-20T14:16:34.470253Z\"}" papermill="{\"duration\":5.2541e-2,\"end_time\":\"2023-02-20T14:16:34.473363\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:34.420822\",\"status\":\"completed\"}" tags="[]"}
``` python
finland=data.loc[(data.REGISTRATION_COUNTRY=='FIN')]
denmark=data.loc[(data.REGISTRATION_COUNTRY=='DNK')]
greece=data.loc[(data.REGISTRATION_COUNTRY=='GRC')]

#only_preferred_restaurant = lambda x: x.iloc[:,-14:]
#fin_users=only_preferred_restaurant(finland)
#dnk_users=only_preferred_restaurant(denmark)
#grc_users=only_preferred_restaurant(greece)

countries =[finland,denmark,greece]
top3countries = pd.concat(countries)
```
:::

::: {#de86ba43 .cell .markdown papermill="{\"duration\":2.6578e-2,\"end_time\":\"2023-02-20T14:16:34.526908\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:34.500330\",\"status\":\"completed\"}" tags="[]"}
```{=html}
<p style="background-color:#00c2e8; color :white ;font-family: Arial;font-size: 20px; padding : 10px ;border-radius: 15px ; display: inline-block;">
Purchase Count by Country</p>
```
:::

::: {#a1f1fb86 .cell .code execution_count="21" execution="{\"iopub.execute_input\":\"2023-02-20T14:16:34.583521Z\",\"iopub.status.busy\":\"2023-02-20T14:16:34.581880Z\",\"iopub.status.idle\":\"2023-02-20T14:16:34.607919Z\",\"shell.execute_reply\":\"2023-02-20T14:16:34.606906Z\"}" papermill="{\"duration\":5.6203e-2,\"end_time\":\"2023-02-20T14:16:34.610120\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:34.553917\",\"status\":\"completed\"}" tags="[]"}
``` python
data.groupby(["REGISTRATION_COUNTRY"]).agg({"PURCHASE_COUNT" : ['count','mean', 'median', 'min', 'max']})
```

::: {.output .execute_result execution_count="21"}
```{=html}
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="5" halign="left">PURCHASE_COUNT</th>
    </tr>
    <tr>
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>median</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>REGISTRATION_COUNTRY</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ARE</th>
      <td>1</td>
      <td>2.000000</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>CAN</th>
      <td>1</td>
      <td>5.000000</td>
      <td>5.0</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>CYP</th>
      <td>1</td>
      <td>3.000000</td>
      <td>3.0</td>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>CZE</th>
      <td>4</td>
      <td>11.500000</td>
      <td>5.0</td>
      <td>1</td>
      <td>35</td>
    </tr>
    <tr>
      <th>DEU</th>
      <td>1</td>
      <td>2.000000</td>
      <td>2.0</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>DNK</th>
      <td>4938</td>
      <td>5.686715</td>
      <td>3.0</td>
      <td>1</td>
      <td>114</td>
    </tr>
    <tr>
      <th>EST</th>
      <td>13</td>
      <td>6.538462</td>
      <td>4.0</td>
      <td>1</td>
      <td>24</td>
    </tr>
    <tr>
      <th>FIN</th>
      <td>5435</td>
      <td>6.364489</td>
      <td>3.0</td>
      <td>1</td>
      <td>221</td>
    </tr>
    <tr>
      <th>FRA</th>
      <td>2</td>
      <td>1.500000</td>
      <td>1.5</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>GBR</th>
      <td>2</td>
      <td>8.500000</td>
      <td>8.5</td>
      <td>4</td>
      <td>13</td>
    </tr>
    <tr>
      <th>GRC</th>
      <td>1530</td>
      <td>6.780392</td>
      <td>2.0</td>
      <td>1</td>
      <td>320</td>
    </tr>
    <tr>
      <th>HRV</th>
      <td>1</td>
      <td>6.000000</td>
      <td>6.0</td>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>HUN</th>
      <td>5</td>
      <td>2.400000</td>
      <td>2.0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>ISR</th>
      <td>3</td>
      <td>5.333333</td>
      <td>6.0</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>LTU</th>
      <td>2</td>
      <td>12.000000</td>
      <td>12.0</td>
      <td>6</td>
      <td>18</td>
    </tr>
    <tr>
      <th>LVA</th>
      <td>3</td>
      <td>13.666667</td>
      <td>6.0</td>
      <td>2</td>
      <td>33</td>
    </tr>
    <tr>
      <th>NOR</th>
      <td>13</td>
      <td>2.230769</td>
      <td>1.0</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>POL</th>
      <td>4</td>
      <td>2.000000</td>
      <td>1.5</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>SWE</th>
      <td>4</td>
      <td>1.500000</td>
      <td>1.5</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>
```
:::
:::

::: {#d73ec5af .cell .markdown papermill="{\"duration\":2.6851e-2,\"end_time\":\"2023-02-20T14:16:34.664019\",\"exception\":false,\"start_time\":\"2023-02-20T14:16:34.637168\",\"status\":\"completed\"}" tags="[]"}
:::
