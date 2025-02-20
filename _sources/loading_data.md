# Exploring Data Sets

- This project was simply to find datasets online then create a python helper file.
- The helper file is simply a dictionary with link to the location of the file and pandas read functions

This is what the file looked like:


```python
myDictionary  = [
    {
        "URL" : 'http://users.stat.ufl.edu/~winner/data/ufo_location_shape.csv',
        "name" : "UFO",
        "load_function" : pd.read_csv
    },
    {
        "URL" : 'https://storage.googleapis.com/kagglesdsdata/datasets/2021/5514/cereal.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241026%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241026T200323Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9c2da66351987f3fac5078571e5e1af112e507f9f47a20b165f07dd08e7466ee7b31dde7cdd26383c940a6912e59b614841a7d02904ef4b688f35d1638c57fee86921e10a94ceca264e1fb3db92fa67fde73ae9587f7df4b0a32aa4c543c0b5ba377f40191f870d0e64dda52e3e826a146c2d8da5953668d7fb0bccc9f73a78ee6570c565848af002b2b4dbea15580db88200468fa36b0ae1551c624fc910b783efc8af0e5a4b1a8163eb2e93598d6116dc757f23af153ae11576c72e7626e80c383ad55567635c9b23646c414015631ffeca954490dd8513e2bd942913dac5b3ab8552d68b9836cb7e26ef91f8a77d6e8a858f2b97a87850f790528f503213c',
        "name" : "CEREAL",
        "load_function" : pd.read_csv
    },
    {
        "URL" : 'https://eazyml.com/documents/Coffee%20As%20A%20Stimulant%20-%20Training%20Data.xlsx',
        "name" : "COFFEE",
        "load_function" :  pd.read_excel
    }
]

```


```python
import pandas as pd
```


```python
import dataSets
```

## UFO sigtings and shape descriptions 

This data set was simply collected to see what similar atributes and looks have been associated with UFO Sigtings.

[More info can be found here](https://www.uforesearchdb.com/)

Questions:
- How was this data collected?
- How was this used in any actual research?


```python
UFO = dataSets.myDictionary[0]["load_function"](dataSets.myDictionary[0]["URL"])
UFO
```




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
      <th>Event.Date</th>
      <th>Shape</th>
      <th>Location</th>
      <th>State</th>
      <th>Country</th>
      <th>Source</th>
      <th>USA</th>
      <th>weekday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6/18/2016</td>
      <td>Boomerang/V-Shaped</td>
      <td>South Barrington</td>
      <td>IL</td>
      <td>USA</td>
      <td>NUFORC</td>
      <td>1</td>
      <td>7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6/17/2016</td>
      <td>Boomerang/V-Shaped</td>
      <td>Kuna</td>
      <td>ID</td>
      <td>USA</td>
      <td>NUFORC</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5/30/2016</td>
      <td>Boomerang/V-Shaped</td>
      <td>Lake Stevens</td>
      <td>WA</td>
      <td>USA</td>
      <td>NUFORC</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5/27/2016</td>
      <td>Boomerang/V-Shaped</td>
      <td>Gerber</td>
      <td>CA</td>
      <td>USA</td>
      <td>NUFORC</td>
      <td>1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5/24/2016</td>
      <td>Boomerang/V-Shaped</td>
      <td>Camdenton</td>
      <td>MO</td>
      <td>USA</td>
      <td>NUFORC</td>
      <td>1</td>
      <td>3</td>
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
    </tr>
    <tr>
      <th>3641</th>
      <td>11/2/2015</td>
      <td>Unknown</td>
      <td>Phnom Penh</td>
      <td>NaN</td>
      <td>Cambodia</td>
      <td>NUFORC</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3642</th>
      <td>4/15/2015</td>
      <td>Unknown</td>
      <td>Hemel Hempstead</td>
      <td>NaN</td>
      <td>England/UK</td>
      <td>NUFORC</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3643</th>
      <td>1/2/2005</td>
      <td>Unknown</td>
      <td>Manat</td>
      <td>NaN</td>
      <td>Puerto Rico</td>
      <td>NUFORC</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3644</th>
      <td>5/4/1988</td>
      <td>Unknown</td>
      <td>Bounty (the ship)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NUFORC</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3645</th>
      <td>11/15/1978</td>
      <td>Unknown</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Tonga</td>
      <td>NUFORC</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>3646 rows × 8 columns</p>
</div>



## 80 Cereals

This data set contains info on sugar, calorie, health rating, and brand association.  I was meant to show how unhealthy cerals can be.

[More can be found here](https://www.kaggle.com/datasets/crawford/80-cereals)

Questions:
-  The original data was collected by Petra Isenberg, Pierre Dragicevic and Yvonne Jansen, was there any bias?
-  What do the most healthy cereals have in common?
-  Are the healthy cerals that far from the less healthjy ones?


```python
cereal = dataSets.myDictionary[1]["load_function"](dataSets.myDictionary[1]["URL"])
cereal
```




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
      <th>name</th>
      <th>mfr</th>
      <th>type</th>
      <th>calories</th>
      <th>protein</th>
      <th>fat</th>
      <th>sodium</th>
      <th>fiber</th>
      <th>carbo</th>
      <th>sugars</th>
      <th>potass</th>
      <th>vitamins</th>
      <th>shelf</th>
      <th>weight</th>
      <th>cups</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>100% Bran</td>
      <td>N</td>
      <td>C</td>
      <td>70</td>
      <td>4</td>
      <td>1</td>
      <td>130</td>
      <td>10.0</td>
      <td>5.0</td>
      <td>6</td>
      <td>280</td>
      <td>25</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.33</td>
      <td>68.402973</td>
    </tr>
    <tr>
      <th>1</th>
      <td>100% Natural Bran</td>
      <td>Q</td>
      <td>C</td>
      <td>120</td>
      <td>3</td>
      <td>5</td>
      <td>15</td>
      <td>2.0</td>
      <td>8.0</td>
      <td>8</td>
      <td>135</td>
      <td>0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>33.983679</td>
    </tr>
    <tr>
      <th>2</th>
      <td>All-Bran</td>
      <td>K</td>
      <td>C</td>
      <td>70</td>
      <td>4</td>
      <td>1</td>
      <td>260</td>
      <td>9.0</td>
      <td>7.0</td>
      <td>5</td>
      <td>320</td>
      <td>25</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.33</td>
      <td>59.425505</td>
    </tr>
    <tr>
      <th>3</th>
      <td>All-Bran with Extra Fiber</td>
      <td>K</td>
      <td>C</td>
      <td>50</td>
      <td>4</td>
      <td>0</td>
      <td>140</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>0</td>
      <td>330</td>
      <td>25</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.50</td>
      <td>93.704912</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Almond Delight</td>
      <td>R</td>
      <td>C</td>
      <td>110</td>
      <td>2</td>
      <td>2</td>
      <td>200</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>8</td>
      <td>-1</td>
      <td>25</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.75</td>
      <td>34.384843</td>
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
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>72</th>
      <td>Triples</td>
      <td>G</td>
      <td>C</td>
      <td>110</td>
      <td>2</td>
      <td>1</td>
      <td>250</td>
      <td>0.0</td>
      <td>21.0</td>
      <td>3</td>
      <td>60</td>
      <td>25</td>
      <td>3</td>
      <td>1.0</td>
      <td>0.75</td>
      <td>39.106174</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Trix</td>
      <td>G</td>
      <td>C</td>
      <td>110</td>
      <td>1</td>
      <td>1</td>
      <td>140</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>12</td>
      <td>25</td>
      <td>25</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>27.753301</td>
    </tr>
    <tr>
      <th>74</th>
      <td>Wheat Chex</td>
      <td>R</td>
      <td>C</td>
      <td>100</td>
      <td>3</td>
      <td>1</td>
      <td>230</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>3</td>
      <td>115</td>
      <td>25</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.67</td>
      <td>49.787445</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Wheaties</td>
      <td>G</td>
      <td>C</td>
      <td>100</td>
      <td>3</td>
      <td>1</td>
      <td>200</td>
      <td>3.0</td>
      <td>17.0</td>
      <td>3</td>
      <td>110</td>
      <td>25</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>51.592193</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Wheaties Honey Gold</td>
      <td>G</td>
      <td>C</td>
      <td>110</td>
      <td>2</td>
      <td>1</td>
      <td>200</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>8</td>
      <td>60</td>
      <td>25</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.75</td>
      <td>36.187559</td>
    </tr>
  </tbody>
</table>
<p>77 rows × 16 columns</p>
</div>



## Coffee as a Stimulant

A simple data set to see if coffee consumtion shows any relation to how fast a participant can type.

[More can be found here](https://eazyml.com/datasets)

Questions:
-  Could the size of the person change these results?
-  why was only coffee used and not also energy drinks or other "energy" substitutes?


```python
coffee = dataSets.myDictionary[2]["load_function"](dataSets.myDictionary[2]["URL"])
coffee
```




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
      <th>Cups of coffee consumed</th>
      <th>Caffeinated or Decaffeinated</th>
      <th>Coffee Brand</th>
      <th>Time of the day</th>
      <th>Typing Speed in characters per minute</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0</td>
      <td>Caffeinated</td>
      <td>Folgers</td>
      <td>Morning</td>
      <td>260</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>Caffeinated</td>
      <td>Folgers</td>
      <td>Morning</td>
      <td>205</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>Decaffeinated</td>
      <td>Folgers</td>
      <td>Morning</td>
      <td>183</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.0</td>
      <td>Caffeinated</td>
      <td>Nescafe</td>
      <td>Morning</td>
      <td>247</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>Caffeinated</td>
      <td>Nescafe</td>
      <td>Morning</td>
      <td>211</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>78</th>
      <td>1.0</td>
      <td>Decaffeinated</td>
      <td>Himalayan</td>
      <td>Evening</td>
      <td>198</td>
    </tr>
    <tr>
      <th>79</th>
      <td>1.5</td>
      <td>Decaffeinated</td>
      <td>Folgers</td>
      <td>Morning</td>
      <td>185</td>
    </tr>
    <tr>
      <th>80</th>
      <td>1.5</td>
      <td>Decaffeinated</td>
      <td>Himalayan</td>
      <td>Morning</td>
      <td>191</td>
    </tr>
    <tr>
      <th>81</th>
      <td>1.5</td>
      <td>Decaffeinated</td>
      <td>Nescafe</td>
      <td>Afternoon</td>
      <td>187</td>
    </tr>
    <tr>
      <th>82</th>
      <td>1.5</td>
      <td>Decaffeinated</td>
      <td>Folgers</td>
      <td>Evening</td>
      <td>186</td>
    </tr>
  </tbody>
</table>
<p>83 rows × 5 columns</p>
</div>




```python
coffee.describe()
```




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
      <th>Cups of coffee consumed</th>
      <th>Typing Speed in characters per minute</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>83.000000</td>
      <td>83.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.246988</td>
      <td>220.421687</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.213173</td>
      <td>37.973393</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>176.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>190.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.000000</td>
      <td>205.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.000000</td>
      <td>257.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.500000</td>
      <td>291.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
