---
title: Nesher Bari - Data Preparation
author: raul
date: 2023-10-15 9:00:00 +0800
categories: [portfolio]
tags: [ai/ml, data-science]
pin: 
---

In the previous two parts of this series, we introduced the Nesher Bari project and dived into an Exploratory Data Analysis (EDA) of one of the main datasets.
In this post, I will walk through how we can prepare and clean the dataset, so it is ready for ingestion by an AI/ML model.

## import libraries

```python
import pandas as pd
import numpy as np
import os
from functools import reduce
from IPython.display import display_html
from itertools import chain,cycle

pd.set_option('display.max_columns', 100)
```

## Challenges

The main challenge we have to address is data inconsistency. That is, different datasets have different conventions for unique identifiers (id's). To address these issues, we will have to transform the datasets using some logic:

```python
df_ornitela_raw = pd.read_csv('./../data/Ornitela_Vultures_Gyps_fulvus_TAU_UCLA_Israel_newer.csv')
df_movebank_raw = pd.read_csv("./../data/eda_movebank_dataset.csv")

# make all ids upper case in ornitela dataframes
df_ornitela = (
    df_ornitela_raw
    .copy()
    .dropna()
)
print(f"The raw Ornitela time series has {len(df_ornitela_raw)}, but {len(df_ornitela_raw)-len(df_ornitela)} are duplicates")

df_ornitela['individual-local-identifier'] = df_ornitela['individual-local-identifier'].str.upper() 

# transform 'white' stirng to just 'w' in whoswho and mortality dataframes
flag_color = lambda x: x.lower().split(' ')[1] in ['white', 'black', 'b', 'w'] if len(x.split(' ')) > 1 else False
shorten_color = lambda x: x.split(" ")[0]+x.split(" ")[1][0]

flag_separation = lambda x: lambda x: x.lower().split(' ')[1] in ['white', 'black', 'b', 'w'] if len(x.split(' ')) > 1 else False

list_updated_dfs = []
for df_i_raw in [df_mortality_raw, df_whoswho_raw]:
    df_i = df_i_raw.copy()
    df_i = df_i[df_i['Movebank_id'].notna()]
    df_i['Movebank_id'] = df_i['Movebank_id'].astype(str)

    df_i['is_colorful'] = df_i['Movebank_id'].apply(flag_color)
    df_i['Movebank_id'] = (
        df_i
        .apply(lambda row: shorten_color(row.Movebank_id) if row.is_colorful else row.Movebank_id, axis=1)
    )
    df_i['Movebank_id'] = df_i['Movebank_id'].str.upper()
    df_i = df_i.drop('is_colorful', axis=1)

    list_updated_dfs.append(df_i)

df_mortality, df_whoswho = list_updated_dfs
```

Now we have consistent keys that allow us to join these three datasets: the Ornitela dataset, the "Who's who" dataset and the mortality dataset, the latter being used to extract the target data. Note that the Who's who dataset has some strange tags e.g., `'S94>A99W'` and `'Y11>T98W'`. These tags might correspond to either multiple vultures or special tags, something that the rangers will be able to confirm.

Next, we can explore how many overlapping vultures each of the last two has with respect to the Ornitela dataset:

```python
ids_dict = {
    "ornitela": ids_ornitela,
    "whoswho": ids_whoswho,
    "mortality": ids_mortality
}

for dataset, arr in ids_dict.items():
    if dataset == "ornitela":
        print(f"The ornitela dataset has {len(arr)} unique vultures.")
        print("="*100)
    else:
        arr_intersect = list(set(ids_ornitela) & set(arr))
        print(f"The {dataset} dataset has {len(arr_intersect)} vulture in common with the Ornitela dataset:")
        print(arr_intersect)
        print("-"*80)
print("="*100)
```

```
The ornitela dataset has 110 unique vultures.
====================================================================================================
The whoswho dataset has 110 vulture in common with the Ornitela dataset:
['J33W', 'Y26', 'A57W', 'A73W', 'E32W', 'E01W', 'A20W', 'J39W', 'E16W', 'E33W', 'A31W', 'A10W', 'E11W', 'T13B', 'A35W', 'T71W', 'E37W', 'A03W', 'A15W', 'E09W', 'T61W', 'T91B', 'J31W', 'T17W', 'J66W', 'A38W', 'T77W', 'E19W', 'J28W', 'T14W', 'J38W', 'J12W', 'Y26B', 'A22W', 'T59W', 'E45W', 'J15W', 'E36W', 'E03', 'A18W', 'J34W', 'E15W', 'A13W', 'T86B', 'A16W', 'A01W', 'J11W', 'T70W', 'J17W', 'J35W', 'A08W', 'E10W', 'A33W', 'A29W', 'T90B', 'T53B', 'A04W', 'E14W', 'J30W', 'A02W', 'E17W', 'A76W', 'A09W', 'E13W', 'E12W', 'E38W', 'E07W', 'A32W', 'E04W', 'E02W', 'T69B', 'J19W', 'T76W', 'T19B', 'A19W', 'E34W', 'A39W', 'A53W', 'E03W', 'J36W', 'A75W', 'T50B', 'J00W', 'T85W', 'A55W', 'E05W', 'T25B', 'A78W', 'J16W', 'A52W', 'J05W', 'A05W', 'E41W', 'E39W', 'A00W', 'A58W', 'T99B', 'T79W', 'J53W', 'T66W', 'T56B', 'E30W', 'Y27B', 'J18W', 'A36W', 'T15W', 'A56W', 'J06W', 'J32W', 'E00W']
--------------------------------------------------------------------------------
The mortality dataset has 27 vulture in common with the Ornitela dataset:
['J28W', 'J36W', 'E14W', 'T85W', 'A20W', 'T59W', 'E33W', 'A76W', 'J15W', 'E03', 'A18W', 'E07W', 'T71W', 'J53W', 'T69B', 'T66W', 'T56B', 'J18W', 'T86B', 'E09W', 'J17W', 'T17W', 'E10W', 'J66W', 'A38W', 'T77W', 'E19W']
--------------------------------------------------------------------------------
====================================================================================================
```

So we see that all the Ornitela vultures are there in the who's who dataset, whereas only 27 Ornitela vultures are present in the mortality dataset. This discrepency is mainly due to the mortality dataset only having information about deceased vultures, whereas the who's who dataset has information about alive vultures too.

---

## Who's Who Dataset

In this section, we'll explore how we can use the 'Whos Who' dataset for getting more insights of the vultures. There are approx. empty 150 columns, which are also present in the raw data. We will get rid of these and also examine for what vultures we have information on whether they're alive or deceased:

```python
df_whoswho = df_whoswho.loc[:, ~df_whoswho.columns.str.contains('^Unnamed')]

ORNITELA_STUDY_NAME = "Ornitela_Vultures_Gyps_fulvus_TAU_UCLA_Israel"

df_whoswho = (
    df_whoswho
    .drop_duplicates(subset='Movebank_id')
    .dropna(subset=["is_alive"])
)    

df_whoswho_dead = (
    df_whoswho[
        (df_whoswho['is_alive'] == 0) & (df_whoswho['date_death'].notna())
    ]
)

df_whoswho_ornitela_dead = (
    df_whoswho[
        (df_whoswho['is_alive'] == 0) & (df_whoswho['date_death'].notna()) &
        (df_whoswho["Movebank_study"] == ORNITELA_STUDY_NAME)
    ]
)
```

So in the who's who dataset, we only have about 10% information about deceased Ornitela vultures. This corresponds to only 18 out of the 101 Ornitela vultures.
In the tables above, we can see that for both all records and the deceased vulture subset, the majority of vultures correspond to the *Gyps fulvus INPA Hatzofe* study, which explains why we have mortality information about only a few vultures.

However, we can still use the Who's Who dataset as either a look up table with vulture information, but more importantly, to construct our train dataset. That is, we can use this information to tag dead or alive Ornitela vultures in the time series data (ornitela dataset) granted they're missing in the mortality dataset. Let's construct this dataframe that we will use to construct an ML trainable Ornitela dataset - we just need to select the relevant columns for tagging mortality and transform the date of death into a datetime datatype. I will also rename the column `details_stop` to `reason`:

```python
df_whoswho_ornitela = df_whoswho[df_whoswho['Movebank_study'] == ORNITELA_STUDY_NAME]

cols_whoswho_clean = [
    'Nili_id',
    'Movebank_id',
    #'is_alive',
    'date_death',
    'details_stop'
]

df_whoswho_ornitela_clean = (
    df_whoswho_ornitela[cols_whoswho_clean]
    .reset_index(drop=True)
    .rename(columns={'details_stop': 'reason'})
)
df_whoswho_ornitela_clean['date_death'] = pd.to_datetime(df_whoswho_ornitela_clean['date_death'],
                                                         format='%Y-%m-%d', errors='coerce')

df_whoswho_ornitela_clean
```

## Mortality Dataset

To start, let's print an overview of the Mortality dataset (`dead_and_injured_vultures.xlsx`), and find out how many of the records correspond to death and injured vultures:

```python
print(f"There are {len(df_mortality_raw)} vultures in the original Whos who dataset.")
print(f"Out of those records, {len(df_mortality_raw.drop_duplicates(subset='Movebank_id'))} are unique records.")

df_mortality.tail()
```

```
There are 88 vultures in the original Whos who dataset.
Out of those records, 88 are unique records.
```


Next, let's find out how many common vultures this dataset has both with the Who's Who and the ornitela table, similar to what we did for the Ornitela dataset:

```python
print(f"The mortality dataset has {len(ids_mortality)} unique vultures.")
print("-"*80)
for dataset, arr in ids_dict.items():
    if dataset == "mortality":
        pass
    else:
        arr_intersect = list(set(ids_mortality) & set(arr))
        print(f"The {dataset} dataset has {len(arr_intersect)} vultures in common with the mortality dataset:")
        print(arr_intersect)
        print("-"*80)

arr_intersect_all = reduce(np.intersect1d, [ids_mortality, ids_ornitela, ids_whoswho])
print(f"The three datasets have {len(arr_intersect_all)} vultures in common, mainly:")        
print(arr_intersect_all)
print("-"*80)
print(
    f"Are the common vultures between the three datasets the same as the common vultures between the who's who and mortality datasets?",
    set(arr_intersect) == set(np.intersect1d(ids_mortality, ids_whoswho))
)
print("="*100)
```

By running this we find a few insights:

- Every vulture that's present in the mortality table is also present in the who's who dataset: this indicates that the mortality dataset also contains information
- The Ornitela and mortality datasets have 27 vultures in common, and the three datasets also have 27 vultures in common

Now let's examine the Ornitela vultures in the mortality dataset:

```python
df_mortality_ornitela = df_mortality[df_mortality['Movebank_id'].isin(df_ornitela['individual-local-identifier'].unique())].reset_index(drop=True)
df_mortality_ornitela
```

When running this we can see that only two Ornitela vultures are injured in the mortality dataset. For the sake of simplicity and because we're only considered alive or dead as a target flag, we'll consider those two records to be alive. That way we can rename the column `death or injury date` to `date_death` and set the value of this column and `reason` to a `NaT` and `NaN` respectively for those injured vultures.

After that, we can trim a few columns that won't be neccesary to get a clean mortality dataset that we will also use to make the ML trainable Ornitela dataset and compare it to the Who's who cleaned dataset above:

```python
ids_injured = ["J17W", "T71W"]

for id_i in ids_injured:
    df_mortality_ornitela.loc[df_mortality_ornitela['Movebank_id'] == id_i, 'death or injury date'] = pd.NaT
    df_mortality_ornitela.loc[df_mortality_ornitela['Movebank_id'] == id_i, 'reason'] = np.NaN

cols_mortality_clean = [
    'Nili_id', 
    'Movebank_id',
    'death or injury date',
#    'fate', 
    'reason', 
]

df_mortality_ornitela_clean = (
    df_mortality_ornitela[cols_mortality_clean]
    .reset_index(drop=True)
    .rename(columns={'death or injury date': 'date_death'})
)


display(df_mortality_ornitela_clean)
display(df_whoswho_ornitela_clean[~df_whoswho_ornitela_clean['date_death'].isna()])
```

Great, we see that these two datasets are taking shape to serve as targets for the Ornitela time series dataset! Because the have the same schema (structure) we can just join them and get rid of any duplicates (e.g., `taranaki` or `xena`):

```python
df_mortality_target = pd.concat([df_mortality_ornitela_clean, df_whoswho_ornitela_clean], ignore_index=True)
df_mortality_target.head()

print(f"There are {count_dup} deceased Ornitela vultures that are both in the who's who and mortality datasets.")
```

```
There are 26 deceased Ornitela vultures that are both in the who's who and mortality datasets.
```

If we look at the duplicates, two interesting patterns appear:

1. There are some vultures that have been marked as deceased in the mortality dataset but not in the who's who dataset
2. Some duplicates records have different dates of death - normally one or two days apart

In any case, this insight must indicates that the mortality dataset is a more recent source of truth, so for every duplicate we encounter we'll keep the mortality record (the first):

```python
df_mortality_target = df_mortality_target.drop_duplicates(subset=['Movebank_id'], keep='first')
df_mortality_target
```


So finally we have our target dataset (with tags about mortality) and we can proceed to merge it with the Ornitela time series dataset to create an ML-trainable dataset.

## Constructing an ML training-ready Ornitela dataset

Before we join the target dataset with the Ornitela dataset, in the Exploratory Data Analysis (EDA) of the Ornitela we decided to only use a subset of columns that the most sense from an inference point of view:

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

Let's filter our dataset with those columns and make sure that the timestamp is a timestamp datatype and not string:

```python
cols_ornitela_clean = [
    'event-id',                                   # primary key
    'individual-local-identifier',                # foreign key
    'timestamp', 
    'location-long', 
    'location-lat',
    'acceleration-raw-x', 
    'acceleration-raw-y', 
    'acceleration-raw-z',
    'external-temperature', 
    'ground-speed', 
    'height-above-msl',
]

df_ornitela = df_ornitela[cols_ornitela_clean]
df_ornitela['timestamp'] = pd.to_datetime(df_ornitela['timestamp'],
                                          format='%Y-%m-%d %H:%M:%S', errors='coerce')
df_ornitela
```



This is a pretty large dataset with over a quarter of a million rows! Before we go ahead with the join, let's see how many unique vultures this has in common with the mortality dataset we prepared above:

```python
ids_target = df_mortality_target['Movebank_id'].unique()
ids_ornitela = df_ornitela['individual-local-identifier'].unique()
ids_intersect = np.intersect1d(ids_target, ids_ornitela)

print(f'The Ornitela and target datasets have {len(ids_target)} and {len(ids_ornitela)} unique vultures respecitvely.')
print(f'Out of this number, they have {len(ids_intersect)} vultures in common:')
print("-"*80)
print(ids_intersect)
```

```
The Ornitela and target datasets have 102 and 110 unique vultures respecitvely.
Out of this number, they have 99 vultures in common:
--------------------------------------------------------------------------------
['A00W' 'A01W' 'A03W' 'A05W' 'A09W' 'A10W' 'A13W' 'A15W' 'A16W' 'A18W'
 'A19W' 'A20W' 'A22W' 'A29W' 'A31W' 'A32W' 'A33W' 'A35W' 'A36W' 'A38W'
 'A39W' 'A52W' 'A53W' 'A55W' 'A56W' 'A57W' 'A58W' 'A73W' 'A75W' 'A76W'
 'A78W' 'E00W' 'E01W' 'E03' 'E03W' 'E04W' 'E05W' 'E07W' 'E09W' 'E10W'
 'E11W' 'E12W' 'E13W' 'E14W' 'E15W' 'E16W' 'E17W' 'E19W' 'E32W' 'E33W'
 'E34W' 'E37W' 'E38W' 'E39W' 'E41W' 'E45W' 'J00W' 'J05W' 'J06W' 'J11W'
 'J12W' 'J15W' 'J16W' 'J17W' 'J18W' 'J19W' 'J28W' 'J31W' 'J32W' 'J33W'
 'J34W' 'J35W' 'J36W' 'J39W' 'J53W' 'J66W' 'T13B' 'T14W' 'T15W' 'T17W'
 'T19B' 'T25B' 'T50B' 'T53B' 'T56B' 'T59W' 'T66W' 'T69B' 'T71W' 'T76W'
 'T77W' 'T79W' 'T85W' 'T86B' 'T90B' 'T99B' 'Y26' 'Y26B' 'Y27B']
```

Great, almost 100 vultures in common! Now, we can do an inner join with the relevant mortality records:

```python
df_ornitela_joined = (
    df_ornitela
    .merge(
        df_mortality_target[['Movebank_id', 'date_death']],
        how="inner",
        left_on="individual-local-identifier",
        right_on="Movebank_id"
    )

)

print(f"The joined dataset has {len(df_ornitela) - len(df_ornitela_joined)} less records than the Ornitela dataset.")
print(f"Out of these records, there are {len(df_ornitela_joined) - len(df_ornitela_joined.drop_duplicates(subset=cols_ornitela_clean))} duplicates.")
print('-'*80)
df_ornitela_joined
```

```
The joined dataset has 122020 less records than the Ornitela dataset.
Out of these records, there are 0 duplicates.
--------------------------------------------------------------------------------
```

So we can see that we missed roughly 5% of the records in the join, which is not significant percentage. 

Next, we need to apply logic to create the mortality labels for each time series record. That is, for each record we compare if the timestamp of the event with the date death for that vulture (if it exists) and if `timestamp` >= `date_death`, then we set a flag for that record as `is_at_risk = "Y"`. The challenge we face is that for mortality we only have a date, whereas the events have a timestamp. We could subsample the timeseries data to only have one record per day, but for that we need to first check the sampling rate for each record to find out if the time separation between events is constant:

```python
print(f"There are {len(df_ornitela['timestamp'].diff().value_counts())} different sampling rates:")
print('-'*60)

df_ornitela['timestamp'].diff().value_counts()
```

```
There are 22549 different sampling rates:
------------------------------------------------------------



0 days 00:10:00    506955
0 days 00:10:01    149427
0 days 00:09:59    130773
0 days 00:10:02     79673
0 days 00:09:58     66405
                    ...  
0 days 12:33:59         1
0 days 08:32:11         1
0 days 12:13:52         1
0 days 13:10:43         1
0 days 03:30:13         1
Name: timestamp, Length: 22549, dtype: int64
```

```python
pd.crosstab(
    index=df_mortality_target[df_mortality_target["Movebank_id"].isin(ids_intersect)]["reason"],
    columns='% observations', 
    normalize='columns')*100
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>% observations</th>
    </tr>
    <tr>
      <th>reason</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>collision powerline</th>
      <td>3.703704</td>
    </tr>
    <tr>
      <th>electrocution</th>
      <td>3.703704</td>
    </tr>
    <tr>
      <th>poisoning</th>
      <td>55.555556</td>
    </tr>
    <tr>
      <th>unknown</th>
      <td>37.037037</td>
    </tr>
  </tbody>
</table>


We can see that the Ornitela timeseries has a very uneven sampling rate. Thus, it's not straight forward to subsample the dataset and be certain that we have one record per day per vulture.
Moreover, poisoning corresponds to over half of the vulture deaths in the Ornitela time series data. This implies that the risk of death might not be instant and vultures that poisoned vultures that die on a given day might have been at risk of death for at least the last 24 hours or more.
Thus, for a first iteration of the ML algorithm, I'll make the following assumption to implement the mortality labelling logic:

**ASSUMPTION:** For any vulture that has a non null `date_death` value, any timestamp within that date will be flagged as `is_at_risk = "Y"`:

```python
# create a labelling column for the time series dataset
risk_evaluator = lambda row: "N" if row.timestamp < row.date_death else ("N" if pd.isnull(row.date_death) else "Y")

df_ornitela_joined['is_at_risk'] = (
    df_ornitela_joined
    .apply(risk_evaluator, axis=1)
)

df_ornitela_target = df_ornitela_joined.copy()
df_ornitela_target
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event-id</th>
      <th>individual-local-identifier</th>
      <th>timestamp</th>
      <th>location-long</th>
      <th>location-lat</th>
      <th>acceleration-raw-x</th>
      <th>acceleration-raw-y</th>
      <th>acceleration-raw-z</th>
      <th>external-temperature</th>
      <th>ground-speed</th>
      <th>height-above-msl</th>
      <th>Movebank_id</th>
      <th>date_death</th>
      <th>is_at_risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16422103004</td>
      <td>T59W</td>
      <td>2020-08-28 04:27:58</td>
      <td>35.013573</td>
      <td>32.753487</td>
      <td>-65.0</td>
      <td>10.0</td>
      <td>-1058.0</td>
      <td>28.0</td>
      <td>0.277778</td>
      <td>368.0</td>
      <td>T59W</td>
      <td>2021-09-11</td>
      <td>N</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16422103005</td>
      <td>T59W</td>
      <td>2020-08-28 04:30:33</td>
      <td>35.013290</td>
      <td>32.753368</td>
      <td>-33.0</td>
      <td>-638.0</td>
      <td>815.0</td>
      <td>28.0</td>
      <td>0.277778</td>
      <td>368.0</td>
      <td>T59W</td>
      <td>2021-09-11</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16422103006</td>
      <td>T59W</td>
      <td>2020-08-28 04:35:28</td>
      <td>35.013302</td>
      <td>32.753448</td>
      <td>-17.0</td>
      <td>-635.0</td>
      <td>824.0</td>
      <td>29.0</td>
      <td>0.000000</td>
      <td>368.0</td>
      <td>T59W</td>
      <td>2021-09-11</td>
      <td>N</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16422103007</td>
      <td>T59W</td>
      <td>2020-08-28 04:40:28</td>
      <td>35.013493</td>
      <td>32.753475</td>
      <td>108.0</td>
      <td>4.0</td>
      <td>1044.0</td>
      <td>31.0</td>
      <td>0.000000</td>
      <td>368.0</td>
      <td>T59W</td>
      <td>2021-09-11</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16422103008</td>
      <td>T59W</td>
      <td>2020-08-28 04:45:37</td>
      <td>35.013519</td>
      <td>32.753521</td>
      <td>60.0</td>
      <td>-432.0</td>
      <td>-1147.0</td>
      <td>31.0</td>
      <td>0.277778</td>
      <td>368.0</td>
      <td>T59W</td>
      <td>2021-09-11</td>
      <td>N</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2251981</th>
      <td>23854049823</td>
      <td>E16W</td>
      <td>2022-09-29 07:20:37</td>
      <td>34.823700</td>
      <td>30.868530</td>
      <td>21.0</td>
      <td>134.0</td>
      <td>873.0</td>
      <td>35.0</td>
      <td>15.277778</td>
      <td>911.0</td>
      <td>E16W</td>
      <td>NaT</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2251982</th>
      <td>23854049824</td>
      <td>E16W</td>
      <td>2022-09-29 07:30:38</td>
      <td>34.841583</td>
      <td>30.892664</td>
      <td>37.0</td>
      <td>227.0</td>
      <td>1058.0</td>
      <td>35.0</td>
      <td>13.055556</td>
      <td>802.0</td>
      <td>E16W</td>
      <td>NaT</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2251983</th>
      <td>23854049825</td>
      <td>E16W</td>
      <td>2022-09-29 07:40:38</td>
      <td>34.831387</td>
      <td>30.891184</td>
      <td>47.0</td>
      <td>315.0</td>
      <td>1204.0</td>
      <td>35.0</td>
      <td>4.444444</td>
      <td>934.0</td>
      <td>E16W</td>
      <td>NaT</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2251984</th>
      <td>23854049826</td>
      <td>E16W</td>
      <td>2022-09-29 07:50:38</td>
      <td>34.851757</td>
      <td>30.920616</td>
      <td>36.0</td>
      <td>238.0</td>
      <td>1106.0</td>
      <td>35.0</td>
      <td>9.166667</td>
      <td>731.0</td>
      <td>E16W</td>
      <td>NaT</td>
      <td>N</td>
    </tr>
    <tr>
      <th>2251985</th>
      <td>23854049827</td>
      <td>E16W</td>
      <td>2022-09-29 08:00:38</td>
      <td>34.895168</td>
      <td>30.957092</td>
      <td>32.0</td>
      <td>273.0</td>
      <td>1169.0</td>
      <td>36.0</td>
      <td>3.611111</td>
      <td>750.0</td>
      <td>E16W</td>
      <td>NaT</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
<p>2251986 rows × 14 columns</p>

```python
df_ornitela_target[(df_ornitela_joined["is_at_risk"] == "Y")]
```

Let's also print the the distribution of at risk vultures versus not at risk:

```python
pd.crosstab(index=df_ornitela_target["is_at_risk"], columns='% observations', normalize='columns')*100
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>col_0</th>
      <th>% observations</th>
    </tr>
    <tr>
      <th>is_at_risk</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>N</th>
      <td>99.925133</td>
    </tr>
    <tr>
      <th>Y</th>
      <td>0.074867</td>
    </tr>
  </tbody>
</table>


The vast majority of time series records (over 99.9%) correspond to non-at-risk events. This is mainly due to two causes:

- Death events are quite rare, especially over the timespan of a vulture's deployment in the field
- Death creates a statistical bias in the time series data: a vultures' tag will be removed after their death, so we stopped seeing those records in the time series data

For this project, the difference in the risk distribution might have dowstream consequences on what ML model we select, and whether we need to resample the data to make the difference less significant. For now, I will keep all the records to save the dataframe and later on we can resample this dataframe.
Finally, let's save the notebook so we can utilise it in the next step for training:

```python
save_target_df = True
if save_target_df:
    print(f"Saving the Ornitela target dataframe:")
    print('='*100)
    out_dir = './../data'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print(f"A {out_dir} directory was created")
        print('-'*60)
    else:
        pass
    out_path = os.path.join(out_dir,'df_ornitela_target.parquet')
    df_ornitela_target.to_parquet(out_path)
    print(f"The Ornitela target dataframe was succesfully saved in {out_path}.")
    print('='*100)
```

```
Saving the Ornitela target dataframe:
====================================================================================================
The Ornitela target dataframe was succesfully saved in ./../data/df_ornitela_target.parquet.
====================================================================================================
```