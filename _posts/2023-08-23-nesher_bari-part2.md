---
title: Nesher Bari - Exploratory Data Analysis
author: raul
date: 2023-08-23 9:00:00 +0800
categories: [portfolio]
tags: [ai/ml, data-science]
pin: true
---

In the previous part of this series, we introduced the Nesher Bari project, which aims to build an ML solution to accelerate vulture conservation (see [more details in Wildlife.ai's website](https://wildlife.ai/projects/nesher-bari/)).
In this post, we dive into more technical details and explore one of the main datasets in the project: the Ornitela dataset.

## import libraries

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jupyterthemes import jtplot

COLORS = sns.color_palette("deep", 12).as_hex()
pd.set_option('display.max_columns', 100)

darkmode_on = True
if darkmode_on:
    jtplot.style(theme='grade3', ticks=True, grid=True)
```

## Load & Extract Data: Quick Overview

```python
df_ornitela_raw = pd.read_csv('./../data/Ornitela_Vultures_Gyps_fulvus_TAU_UCLA_Israel_newer.csv')
df_ornitela_raw.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event-id</th>
      <th>visible</th>
      <th>timestamp</th>
      <th>location-long</th>
      <th>location-lat</th>
      <th>acceleration-raw-x</th>
      <th>acceleration-raw-y</th>
      <th>acceleration-raw-z</th>
      <th>bar:barometric-height</th>
      <th>battery-charge-percent</th>
      <th>battery-charging-current</th>
      <th>external-temperature</th>
      <th>gps:hdop</th>
      <th>gps:satellite-count</th>
      <th>gps-time-to-fix</th>
      <th>ground-speed</th>
      <th>heading</th>
      <th>height-above-msl</th>
      <th>import-marked-outlier</th>
      <th>gls:light-level</th>
      <th>mag:magnetic-field-raw-x</th>
      <th>mag:magnetic-field-raw-y</th>
      <th>mag:magnetic-field-raw-z</th>
      <th>orn:transmission-protocol</th>
      <th>tag-voltage</th>
      <th>sensor-type</th>
      <th>individual-taxon-canonical-name</th>
      <th>tag-local-identifier</th>
      <th>individual-local-identifier</th>
      <th>study-name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16422103004</td>
      <td>True</td>
      <td>2020-08-28 04:27:58.000</td>
      <td>35.013573</td>
      <td>32.753487</td>
      <td>-65.0</td>
      <td>10.0</td>
      <td>-1058.0</td>
      <td>0.0</td>
      <td>92</td>
      <td>0.0</td>
      <td>28.0</td>
      <td>1.7</td>
      <td>5</td>
      <td>159.69</td>
      <td>0.277778</td>
      <td>87.0</td>
      <td>368.0</td>
      <td>False</td>
      <td>1046.0</td>
      <td>-0.621</td>
      <td>0.036</td>
      <td>0.014</td>
      <td>GPRS</td>
      <td>4100.0</td>
      <td>gps</td>
      <td>Gyps fulvus</td>
      <td>202382</td>
      <td>T59w</td>
      <td>Ornitela_Vultures_Gyps_fulvus_TAU_UCLA_Israel</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16422103005</td>
      <td>True</td>
      <td>2020-08-28 04:30:33.000</td>
      <td>35.013290</td>
      <td>32.753368</td>
      <td>-33.0</td>
      <td>-638.0</td>
      <td>815.0</td>
      <td>0.0</td>
      <td>92</td>
      <td>15.0</td>
      <td>28.0</td>
      <td>1.7</td>
      <td>5</td>
      <td>16.04</td>
      <td>0.277778</td>
      <td>47.0</td>
      <td>368.0</td>
      <td>False</td>
      <td>1386.0</td>
      <td>-0.603</td>
      <td>-0.330</td>
      <td>-0.495</td>
      <td>GPRS</td>
      <td>4103.0</td>
      <td>gps</td>
      <td>Gyps fulvus</td>
      <td>202382</td>
      <td>T59w</td>
      <td>Ornitela_Vultures_Gyps_fulvus_TAU_UCLA_Israel</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16422103006</td>
      <td>True</td>
      <td>2020-08-28 04:35:28.000</td>
      <td>35.013302</td>
      <td>32.753448</td>
      <td>-17.0</td>
      <td>-635.0</td>
      <td>824.0</td>
      <td>0.0</td>
      <td>93</td>
      <td>15.0</td>
      <td>29.0</td>
      <td>1.8</td>
      <td>5</td>
      <td>11.44</td>
      <td>0.000000</td>
      <td>113.0</td>
      <td>368.0</td>
      <td>False</td>
      <td>2047.0</td>
      <td>-0.575</td>
      <td>-0.367</td>
      <td>-0.493</td>
      <td>GPRS</td>
      <td>4108.0</td>
      <td>gps</td>
      <td>Gyps fulvus</td>
      <td>202382</td>
      <td>T59w</td>
      <td>Ornitela_Vultures_Gyps_fulvus_TAU_UCLA_Israel</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16422103007</td>
      <td>True</td>
      <td>2020-08-28 04:40:28.000</td>
      <td>35.013493</td>
      <td>32.753475</td>
      <td>108.0</td>
      <td>4.0</td>
      <td>1044.0</td>
      <td>0.0</td>
      <td>93</td>
      <td>0.0</td>
      <td>31.0</td>
      <td>1.8</td>
      <td>5</td>
      <td>11.53</td>
      <td>0.000000</td>
      <td>52.0</td>
      <td>368.0</td>
      <td>False</td>
      <td>1928.0</td>
      <td>0.040</td>
      <td>-0.045</td>
      <td>-0.659</td>
      <td>GPRS</td>
      <td>4108.0</td>
      <td>gps</td>
      <td>Gyps fulvus</td>
      <td>202382</td>
      <td>T59w</td>
      <td>Ornitela_Vultures_Gyps_fulvus_TAU_UCLA_Israel</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16422103008</td>
      <td>True</td>
      <td>2020-08-28 04:45:37.000</td>
      <td>35.013519</td>
      <td>32.753521</td>
      <td>60.0</td>
      <td>-432.0</td>
      <td>-1147.0</td>
      <td>0.0</td>
      <td>93</td>
      <td>0.0</td>
      <td>31.0</td>
      <td>2.0</td>
      <td>4</td>
      <td>20.52</td>
      <td>0.277778</td>
      <td>290.0</td>
      <td>368.0</td>
      <td>False</td>
      <td>496.0</td>
      <td>-0.314</td>
      <td>0.111</td>
      <td>-0.113</td>
      <td>GPRS</td>
      <td>4106.0</td>
      <td>gps</td>
      <td>Gyps fulvus</td>
      <td>202382</td>
      <td>T59w</td>
      <td>Ornitela_Vultures_Gyps_fulvus_TAU_UCLA_Israel</td>
    </tr>
  </tbody>
</table>

Let's explore some basic information about the Ornitela dataset, mainly shape and schema:

```python
print(f" n_rows: {df_ornitela_raw.shape[0]} \n n_columns: {df_ornitela_raw.shape[-1]}")
```

```
 n_rows: 2374007 
 n_columns: 30
```

```python
print(f"dataset schema:")
print("="*60)
df_ornitela_raw.info()
```

```
dataset schema:
============================================================
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2374007 entries, 0 to 2374006
Data columns (total 30 columns):
 #   Column                           Dtype  
---  ------                           -----  
 0   event-id                         int64  
 1   visible                          bool   
 2   timestamp                        object 
 3   location-long                    float64
 4   location-lat                     float64
 5   acceleration-raw-x               float64
 6   acceleration-raw-y               float64
 7   acceleration-raw-z               float64
 8   bar:barometric-height            float64
 9   battery-charge-percent           int64  
 10  battery-charging-current         float64
 11  external-temperature             float64
 12  gps:hdop                         float64
 13  gps:satellite-count              int64  
 14  gps-time-to-fix                  float64
 15  ground-speed                     float64
 16  heading                          float64
 17  height-above-msl                 float64
 18  import-marked-outlier            bool   
 19  gls:light-level                  float64
 20  mag:magnetic-field-raw-x         float64
 21  mag:magnetic-field-raw-y         float64
 22  mag:magnetic-field-raw-z         float64
 23  orn:transmission-protocol        object 
 24  tag-voltage                      float64
 25  sensor-type                      object 
 26  individual-taxon-canonical-name  object 
 27  tag-local-identifier             int64  
 28  individual-local-identifier      object 
 29  study-name                       object 
dtypes: bool(2), float64(18), int64(4), object(6)
memory usage: 511.7+ MB
```

Let's explore if there are any duplicated event entries or any entries with null values:

```python
print(f"duplicated rows in the Ornitela dataset:")
print("="*60)
print(df_ornitela_raw.duplicated().sum())

print(f"duplicated rows in the Ornitela dataset:")
print("="*60)
print(df_ornitela_raw.isna().sum())
```

```
duplicated rows in the Ornitela dataset:
============================================================
0
duplicated rows in the Ornitela dataset:
============================================================
event-id                           0
visible                            0
timestamp                          0
location-long                      1
location-lat                       1
acceleration-raw-x                 0
acceleration-raw-y                 0
acceleration-raw-z                 0
bar:barometric-height              0
battery-charge-percent             0
battery-charging-current           0
external-temperature               0
gps:hdop                           0
gps:satellite-count                0
gps-time-to-fix                    0
ground-speed                       0
heading                            0
height-above-msl                   0
import-marked-outlier              0
gls:light-level                    0
mag:magnetic-field-raw-x           0
mag:magnetic-field-raw-y           0
mag:magnetic-field-raw-z           0
orn:transmission-protocol          0
tag-voltage                        0
sensor-type                        0
individual-taxon-canonical-name    0
tag-local-identifier               0
individual-local-identifier        0
study-name                         0
dtype: int64
```

We can see that the Ornitela dataset contains no duplicates but 1 entry with null latitude and longitude.

Let's make sure we drop any null values and even duplicates (even if there are non) and then extract some basic stats from each of the attributes:

```python
df_ornitela = (
    df_ornitela_raw
    .copy()
    .drop_duplicates()
    .dropna()
)

df_ornitela.describe().apply(lambda s: s.apply('{0:.5f}'.format))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event-id</th>
      <th>location-long</th>
      <th>location-lat</th>
      <th>acceleration-raw-x</th>
      <th>acceleration-raw-y</th>
      <th>acceleration-raw-z</th>
      <th>bar:barometric-height</th>
      <th>battery-charge-percent</th>
      <th>battery-charging-current</th>
      <th>external-temperature</th>
      <th>gps:hdop</th>
      <th>gps:satellite-count</th>
      <th>gps-time-to-fix</th>
      <th>ground-speed</th>
      <th>heading</th>
      <th>height-above-msl</th>
      <th>gls:light-level</th>
      <th>mag:magnetic-field-raw-x</th>
      <th>mag:magnetic-field-raw-y</th>
      <th>mag:magnetic-field-raw-z</th>
      <th>tag-voltage</th>
      <th>tag-local-identifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>20552869208.05338</td>
      <td>35.41146</td>
      <td>29.08398</td>
      <td>25.38844</td>
      <td>519.36946</td>
      <td>800.00706</td>
      <td>0.00000</td>
      <td>92.90503</td>
      <td>7.60352</td>
      <td>34.75356</td>
      <td>1.51000</td>
      <td>7.60744</td>
      <td>29.60435</td>
      <td>4.39986</td>
      <td>178.27466</td>
      <td>705.50741</td>
      <td>688.54746</td>
      <td>0.12841</td>
      <td>-0.10202</td>
      <td>-0.24087</td>
      <td>4121.36826</td>
      <td>206449.77052</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2181978669.49922</td>
      <td>2.80387</td>
      <td>4.93453</td>
      <td>101.53411</td>
      <td>301.16978</td>
      <td>259.57626</td>
      <td>0.00000</td>
      <td>13.07223</td>
      <td>12.04568</td>
      <td>4.85817</td>
      <td>1.06649</td>
      <td>2.30207</td>
      <td>25.71204</td>
      <td>6.77396</td>
      <td>105.65555</td>
      <td>529.72801</td>
      <td>841.40669</td>
      <td>1.18763</td>
      <td>0.78688</td>
      <td>0.86235</td>
      <td>84.77994</td>
      <td>5381.99189</td>
    </tr>
    <tr>
      <th>min</th>
      <td>16105780011.00000</td>
      <td>0.00000</td>
      <td>2.00003</td>
      <td>-1606.00000</td>
      <td>-1329.00000</td>
      <td>-1315.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>3.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>-1998.00000</td>
      <td>0.00000</td>
      <td>-6.71800</td>
      <td>-13.78400</td>
      <td>-6.31800</td>
      <td>0.00000</td>
      <td>202359.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18884479400.25000</td>
      <td>34.82033</td>
      <td>30.75541</td>
      <td>-17.00000</td>
      <td>234.00000</td>
      <td>604.00000</td>
      <td>0.00000</td>
      <td>91.00000</td>
      <td>0.00000</td>
      <td>33.00000</td>
      <td>1.00000</td>
      <td>6.00000</td>
      <td>12.76000</td>
      <td>0.00000</td>
      <td>85.00000</td>
      <td>431.00000</td>
      <td>17.00000</td>
      <td>-0.06700</td>
      <td>-0.37700</td>
      <td>-0.56200</td>
      <td>4095.00000</td>
      <td>202377.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>20991679678.50000</td>
      <td>35.00437</td>
      <td>30.83160</td>
      <td>27.00000</td>
      <td>568.00000</td>
      <td>817.00000</td>
      <td>0.00000</td>
      <td>100.00000</td>
      <td>0.00000</td>
      <td>35.00000</td>
      <td>1.30000</td>
      <td>7.00000</td>
      <td>16.16000</td>
      <td>0.27778</td>
      <td>179.00000</td>
      <td>515.00000</td>
      <td>131.00000</td>
      <td>0.25500</td>
      <td>-0.00500</td>
      <td>-0.17100</td>
      <td>4155.00000</td>
      <td>202398.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>22502683086.75000</td>
      <td>35.22010</td>
      <td>30.95092</td>
      <td>73.00000</td>
      <td>799.00000</td>
      <td>992.00000</td>
      <td>0.00000</td>
      <td>100.00000</td>
      <td>15.00000</td>
      <td>38.00000</td>
      <td>1.70000</td>
      <td>9.00000</td>
      <td>35.86000</td>
      <td>9.44444</td>
      <td>270.00000</td>
      <td>884.00000</td>
      <td>1676.00000</td>
      <td>0.55200</td>
      <td>0.34500</td>
      <td>0.12800</td>
      <td>4178.00000</td>
      <td>213563.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>23854497939.00000</td>
      <td>45.30416</td>
      <td>40.02653</td>
      <td>1798.00000</td>
      <td>1810.00000</td>
      <td>2040.00000</td>
      <td>0.00000</td>
      <td>100.00000</td>
      <td>57.00000</td>
      <td>68.00000</td>
      <td>15.90000</td>
      <td>22.00000</td>
      <td>272.37000</td>
      <td>783.33333</td>
      <td>360.00000</td>
      <td>9992.00000</td>
      <td>2047.00000</td>
      <td>28.17200</td>
      <td>4.78000</td>
      <td>14.32900</td>
      <td>4203.00000</td>
      <td>213596.00000</td>
    </tr>
  </tbody>
</table>

Also, let's take a quick look at the frequency of values for some key categorical attributes:

```python
cols_categorical = [
    'orn:transmission-protocol', 
    'individual-taxon-canonical-name', 
    'individual-local-identifier', 
    'study-name'
]

# print frequency tables for each categorical feature
for column in cols_categorical:
    display(pd.crosstab(
        index=df_ornitela[column], 
        columns='% observations',
        normalize='columns'
    )*100
           )
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>% observations</th>
    </tr>
    <tr>
      <th>orn:transmission-protocol</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GPRS</th>
      <td>99.58252</td>
    </tr>
    <tr>
      <th>SMS</th>
      <td>0.41748</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>% observations</th>
    </tr>
    <tr>
      <th>individual-taxon-canonical-name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gyps</th>
      <td>1.144563</td>
    </tr>
    <tr>
      <th>Gyps fulvus</th>
      <td>98.855437</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>% observations</th>
    </tr>
    <tr>
      <th>individual-local-identifier</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A00w</th>
      <td>1.603534</td>
    </tr>
    <tr>
      <th>A01w</th>
      <td>0.366259</td>
    </tr>
    <tr>
      <th>A02w</th>
      <td>0.079781</td>
    </tr>
    <tr>
      <th>A03w</th>
      <td>1.865749</td>
    </tr>
    <tr>
      <th>A04w</th>
      <td>0.320934</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>T91b</th>
      <td>0.832938</td>
    </tr>
    <tr>
      <th>T99b</th>
      <td>1.806609</td>
    </tr>
    <tr>
      <th>Y26</th>
      <td>0.006108</td>
    </tr>
    <tr>
      <th>Y26b</th>
      <td>1.052061</td>
    </tr>
    <tr>
      <th>Y27b</th>
      <td>1.870551</td>
    </tr>
  </tbody>
</table>
<p>110 rows × 1 columns</p>


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>% observations</th>
    </tr>
    <tr>
      <th>study-name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ornitela_Vultures_Gyps_fulvus_TAU_UCLA_Israel</th>
      <td>100.0</td>
    </tr>
  </tbody>
</table>

From this we notice a few remarks:

- Most recorded events come from GPS sensors instead of SMS.
  
- Most vultures are Griffon vultures (~99.9%), and the rest are only tagged as 'Gyps' (vultures): because of the small fraction of the latter, we will safely assume that 'Gyps' also refer to Griffon vultures.
  
- All the data entries correspond to the 'Ornitela Vultures Gyps fulvus' project (as expected), in collaboration between UCLA and TAU: this means that we don't have to filter the dataframe for that study.
  

It's more difficult to observe the frequency of records for each identifier (corresponding to each unique vulture) in a tabular form. To appreciate that, let's plot the percentage of observations for each vulture (based on tag-local identifier):

```python
vultures_rate = pd.crosstab(
    index=df_ornitela['individual-local-identifier'], 
    columns='% observations', 
    normalize='columns')*100

plt.figure(figsize=(32,8))
plt.xticks(rotation = 70)                           # rotates X-Axis Ticks by 70-degrees
plt.ylim([0,6])
sns.set(font_scale=1.5)
sns.barplot(
    data=vultures_rate, 
    x=vultures_rate.index, 
    y=vultures_rate['% observations']
)
plt.show()
```

![png](../../assets/img/nesher_bari/output_15_0.png?msec=1698559689962)

In the figure above, we can see that the frequency of observations (sampling size) across different vultures is pretty even. Although there are differences of roughly ~1% across difference vultures, the sampling is not dominated by one or a group of vultures.

## Data Distribution

In this section, we look more in detail at the distribution of different attributes. Studying these attributes might help us understand if there are any systematic trends or biases in the data. It also helps us better understand the attributes from the dataset, thus facilitating feature selection too.

First, we define a reusable function to plot a distribution for each attribute:

```python
# define an auxiliary function to draw several plots in a tight layout
def plot_distributon(
    cols, 
    stat='count', 
    bins=100,
    log_transformation=False,
):
    plt.figure(figsize=(25, 7))
    sns.set(font_scale=2)
    for i, feature in enumerate(cols):
        ax = plt.subplot(1, len(cols), i+1)
        if log_transformation:
            sns.histplot(
                data=df_ornitela[cols], 
                x=feature, 
                stat=stat, 
                bins=bins,
                log_scale=(False,True)
            )
        else:
            sns.histplot(
                data=df_ornitela[cols], 
                x=feature, 
                stat=stat, 
                bins=bins,
            )
```

### Location

We explore the probability distribution of *latitude* and *longitude*. This is important because the INPA only operates in Israel, therefore we wouldn't like the dataset to contain many griffon vultures that flew away from the area.

```python
loc_cols = ['location-long', 'location-lat']
df_ornitela[loc_cols].describe().apply(lambda s: s.apply('{0:.5f}'.format))

ISRAEL_LAT_RANGE = (29.55805, 33.20733)
ISRAEL_LONG_RANGE = (34.57149, 35.57212)

# plot lat and long  distribution
fig, axs  = plt.subplots(1,2, figsize=(25, 7))
sns.set(font_scale=2)

for i, col in enumerate(loc_cols):
    sns.histplot(
        data=df_ornitela,
        x=col, 
        bins=25,
        stat='probability',
        ax=axs[i]
    )
    axs[0].axvline(ISRAEL_LONG_RANGE[i], color=COLORS[1], linewidth=2)
    axs[1].axvline(ISRAEL_LAT_RANGE[i], color=COLORS[1], linewidth=2)

plt.tight_layout()
plt.show()
```

![png](../../assets/img/nesher_bari/output_21_0.png?msec=1698559689963)

Although some griffon vultures flew out of Israel, we can see that the vast majority of records occurred in Israel.

In the future, we could have a more fine-grained analysis to potentially improve the performance of an ML model. This would mainly involve getting rid of events that didn't occur in the Negev desert in Israel, where INPA focuses their conversation efforts.

### Height

Height might be a potential telling feature as outliers might correspond to a death related event, although change in height for a particular vulture would be more telling (this might be encoded in acceleration). Let's plot the probability distribution of height:

```python
plt.figure(figsize=(14,7))
sns.histplot(
    data=df_ornitela, 
    x=df_ornitela['height-above-msl'], 
    stat='probability',
    bins=80
)

plt.axvline(
    df_ornitela['height-above-msl'].median(), 
    color=COLORS[1],
    label=f"median = {df_ornitela['height-above-msl'].median()} m"
)
plt.legend(loc=0, prop={'size': 20})
plt.xlim(-500,3000)
plt.show()
```

![png](../../assets/img/nesher_bari/output_24_0.png?msec=1698559689991)

We get a tail-end distribution with a median altitude of roughly 515m. This reflects the known fact that griffon vultures tend to stick around higher altitudes.
We will also print the percentile distribution for reference:

### Ground Speed

Ground speeds will tend to be very specific decimal numbers. Therefore, the challenge with looking at its distritbution is that it'll be very sparse. To make up for this sparness, we can bin the data and plot the probability distribution:

```python
plt.figure(figsize=(14,7))
sns.histplot(
    data=df_ornitela,
    x=df_ornitela['ground-speed'], 
    stat='probability',
    bins=75,
)
plt.xlim([0, 200])
plt.show()
```

![png](../../assets/img/nesher_bari/output_28_0.png?msec=1698559689964)

We see that it's very much a tail-end distribution with the tail just around 25m/s. Therefore, most likely the speed of a vulture will be below 20 m/s. However, there's some interesting behaviour at play. We binned the data in 75 bins but we can only see 3 in the plot above. I did bound the x-axis between speeds of 0 to 200, but still this doesn't really explain all the missing bins. Let'e see if a log-plot of the count distribution without x-axis boundaries can give us any insights:

```python
plt.figure(figsize=(14,7))
sns.histplot(
    data=df_ornitela[df_ornitela['ground-speed']>40], 
    x=df_ornitela['ground-speed'], 
    stat='count',
    bins=75,
    log_scale=(False,True),
)
plt.show()
```

![png](../../assets/img/nesher_bari/output_30_0.png?msec=1698559689964)

```python
df_ornitela[['ground-speed']].describe().apply(lambda s: s.apply('{0:.5f}'.format))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ground-speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2374006.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.39986</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.77396</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.27778</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.44444</td>
    </tr>
    <tr>
      <th>max</th>
      <td>783.33333</td>
    </tr>
  </tbody>
</table>

Now we can see a few outliers with ground-speed values of slightly more than 400 [m/s] and slightly less than 800 [m/s]. This seems excesively large ground speeds for Griffon vultures, and it still so it must a malfunctioning of the tracking device.

The summary statistics also doesn't seem to indicate why we see these behaviour with missing bins. Let's try to filter the dataset for speed values less than 80 m/s and replot the probability distribution of speed:

```python
df_ornitela_filtered = df_ornitela[df_ornitela['ground-speed'] < 80]
print(
    f"Number of records where vultures are moving at speeds greater than 80m/s:",
    f"{1 - (len(df_ornitela_filtered)/len(df_ornitela)):.2e}"
)
print("="*100)

plt.figure(figsize=(14,7))
sns.histplot(
    data=df_ornitela_filtered, 
    x=df_ornitela_filtered['ground-speed'], 
    stat='probability',
    bins=100,
)
plt.xlim([0, 25])
plt.show()
```

```
Number of records where vultures are moving at speeds greater than 80m/s: 1.68e-06
====================================================================================================
```

![png](../../assets/img/nesher_bari/output_33_1.png?msec=1698559689964)

This distribution is a much more telling picture of speed. It tells us that vultures are most likely to be static or moving at speeds below 2 m/s, which probably means they're moving on the ground. Moving forward, we probably want to implement this filter in our dataset unless those outliers really represent dying vultures according to experts. In any case, these only represent 1.5 million of a fraction of the Ornitela sample.

### Acceleration

Now we take a look at a very interesting attribute: acceleration which is measured in the 3 dimensional x-, y- and z-xis. First, let's have an overall look at the desntiy distribution for acceleration in each direction:

```python
cols_acceleration = [
    'acceleration-raw-x', 
    'acceleration-raw-y',
    'acceleration-raw-z'
]

plot_distributon(cols_acceleration, stat='density', bins=120)
plt.tight_layout()
plt.show()
```

![png](../../assets/img/nesher_bari/output_36_0.png?msec=1698559689964)

These a very interesting and diverse distributions! Going through each one (left to right above):

- **x-axis:** It appears that the acceleration in this axis is mainly Gaussianly distributed around 0. Compared to the other directions, it's surprising how symmatricaly distributed the acceleration is.
  
- **y-axis:** This deviates more from a Gaussian shape, though one could argue it resembles a multi-modal Gaussian distribution. The surprising remark of this distribution is that the acceleration values are overall positive as opposed to the acceleration in the x-axis.
  
- **z-axis:** Finally, the acceleration in the z-axis has a Gaussian-like shape without a defined peak. Again, we see that there are no negative acceleration values in this direction (except for a tiny bump around `acceleration-raw-z= -500` ).
  

Moving to interpreting these results, first one should note that griffon vulture are gliders, meaning that they minimise flapping and aim to optimise air currents. 
This might explain why the x-axis acceleration is distributed around 0. That is, if the wind is moves at constant speeds (not direction) and the amount of flapping is minimal, the values around 0 will correspond to fluctuations in the wind speed that are not significant. This gliding might also explain why most acceleration values in the y- and z- directions are negative. Mainly, it is unlikely that the wind will decelerate vultures sideways in a significant manner and as glidders, they make good use of convective air currents to move upwards.

To see if we can unpack more insights, let's plot the three density distributions in a log-scale:

```python
# plot with log transformation on the y-axis
plot_distributon(
    cols_acceleration,
    stat='density',
    bins=120,
    log_transformation=True
)
plt.tight_layout()
```

![png](../../assets/img/nesher_bari/output_38_0.png?msec=1698559689964)

Apart from really highlighting the bump at 0 for the z-axis accleration, I don't think this log-scale distributions are very telling. We will also print the summary statistics for reference:

```python
df_ornitela[cols_acceleration].describe().apply(lambda s: s.apply('{0:.5f}'.format))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acceleration-raw-x</th>
      <th>acceleration-raw-y</th>
      <th>acceleration-raw-z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
      <td>2374006.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>25.38844</td>
      <td>519.36946</td>
      <td>800.00706</td>
    </tr>
    <tr>
      <th>std</th>
      <td>101.53411</td>
      <td>301.16978</td>
      <td>259.57626</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1606.00000</td>
      <td>-1329.00000</td>
      <td>-1315.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-17.00000</td>
      <td>234.00000</td>
      <td>604.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>27.00000</td>
      <td>568.00000</td>
      <td>817.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>73.00000</td>
      <td>799.00000</td>
      <td>992.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1798.00000</td>
      <td>1810.00000</td>
      <td>2040.00000</td>
    </tr>
  </tbody>
</table>

### Pressure

Let's have a look at the distribution of pressure as measured by barometric height ([see more details here](https://en.wikipedia.org/wiki/Barometric_formula))

```python
plt.figure(figsize=(14,7))
sns.histplot(
    data=df_ornitela, 
    x=df_ornitela['bar:barometric-height'], 
    bins=80,
)
plt.show()
```

![png](../../assets/img/nesher_bari/output_42_0.png?msec=1698559689991)


We see that the pressure is not very informative because it's always 0 so we will ignore this attribute from the analysis.

### Satellite Count

This counts how many satellites is used to produced each record. Plotting the count distribution:

```python
plt.figure(figsize=(14,7))
sns.histplot(
    data=df_ornitela, 
    x=df_ornitela['gps:satellite-count'], 
    bins=15,
)
plt.axvline(
    df_ornitela['gps:satellite-count'].median(), 
    color=COLORS[1], 
    label=f"median = {df_ornitela['gps:satellite-count'].median()}",
)
plt.legend(loc=0, prop={'size': 20})
plt.show()
```

![png](../../assets/img/nesher_bari/output_46_0.png?msec=1698559689991)

We can see that most frequently around 7 satellites are used to produce a data entry. However, this attribute might not be critical for the first version of an ML algorithm.

### Temperature

Now let's look at another very interesting attritbute: temperature. Assuming this corresponds to the temperature of the vulture, this can be a very good proxy for a death event. Especially if we look at the history of low temperature events associated with death, we might be able to distinguish between different types of death: lead poisoning, collision or hunting for example. Plotting the distribution of temperature:

```python
plt.figure(figsize=(14,7))
sns.histplot(
    data=df_ornitela, 
    x=df_ornitela['external-temperature'], 
    stat='percent',
    bins=50,
)
plt.axvline(
    df_ornitela['external-temperature'].median(), 
    color=COLORS[1], 
    label=f"median = {df_ornitela['external-temperature'].median()} ˚C"
)
plt.legend(loc=0, prop={'size': 12})
plt.show()
```

![png](../../assets/img/nesher_bari/output_50_0.png?msec=1698559689991)

Interestingly, there are quite a few low temperature values and a bump around 0 (around 1%). Given that death is a rare event, the latter is an outlier that may be hidden by the abundance of values around 35˙C. Let's see if the logged-scale count distribution can resolve that:

```python
# Logarithmic transformation on the y-axis
plt.figure(figsize=(14,7))
sns.histplot(
    data=df_ornitela, 
    x=df_ornitela['external-temperature'],
    stat='percent',
    bins=50, 
    log_scale=(False,True),
)
```

![png](../../assets/img/nesher_bari/output_52_1.png?msec=1698559689992)

When looking at at the plot above, now we really see the highlighted bump around 0 as a very interesting outlier, corresponding to more than 1,000 records of dead vultures.

## Correlation Overview

### Scatter Matrix Plot

Finally, a scatter matrix plot allows us to see if there are any correlations between numerical features, which will feed into feature selection. If there's a strong correlation (or various) between two attributes, this will tell us that one of the features is redundant to feed into an ML model.

Since plotting a scatter matrix is very computationally demanding, I'll subsample the data set:

```python
# extract numerical cols of interest
num_cols = [
    #'location-long', 
    #'location-lat',
    'acceleration-raw-x', 
    'acceleration-raw-y',
    'acceleration-raw-z',
    'ground-speed',
    'external-temperature',
    'height-above-msl',
]

df_num = df_ornitela[num_cols].sample(len(df_ornitela)//10)
g = sns.pairplot(df_num)
for ax in g.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), rotation = 90)                           # rotate x axis labels
    ax.set_ylabel(ax.get_ylabel(), rotation = 0)                            # rotate y axis labels
    ax.yaxis.get_label().set_horizontalalignment('right')                   # set y labels alignment
plt.tight_layout()
```

![png](../../assets/img/nesher_bari/output_55_0.png?msec=1698559690105)

There are no obvious correlations at first glance. We can note however that the deviation of the scatter is quite great and that we see quite a lot of clustering between the attributes. In any case, we don't have to remove attributes due to correlations.

## Conclusion

In this post, we performed an Exploratory Data Analysis (EDA) on the Ornitela dataset, which enables with a better understanding on how the dataset is characterised. Moreover, it gave us an overview of what attributes of the dataset are most important for developing an ML algorithm that can predict a likelihood of high-risk of death based on the attributes (e.g., latitude, longitude, speed, etc).

From a causal inference perspective, I think the most important attributes for the first version of an ML algorithm are the following:

- `event-id`
- `individual-local-identifier`
- `timestamp`
- `location-long`
- `location-lat`
- `height-above-msl`
- `ground-speed`
- `acceleration-raw-x`
- `acceleration-raw-y`
- `acceleration-raw-z`
- `external-temperature`

Once we have done a first feature selection, we can go ahead and prepare the dataset to be fed into an ML model. Stay tuned for the next steps!