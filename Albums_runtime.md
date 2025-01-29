---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Top Albums and Runtime

+++

## Question: Does album run time have anything to do with being a top selling album in a year?

```{code-cell} ipython3
import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
```

This is a link to the starting wiki:
[albums](https://en.wikipedia.org/wiki/Top_Album_Sales#2024)

+++

## The first thing I do is create the bs4 object 

```{code-cell} ipython3
albums_url = 'https://en.wikipedia.org/wiki/Top_Album_Sales#2020'
```

```{code-cell} ipython3
albums_html = requests.get(albums_url).content
```

```{code-cell} ipython3
cs_albums = BeautifulSoup(albums_html, 'html.parser')
```

```{code-cell} ipython3
type(cs_albums)
```

## From here I explored the websit using inspect and found 'tbody' was holding all of the lists of names and links

```{code-cell} ipython3
wiki_content = cs_albums.find_all('tbody')
```

## I found all the links to the individual albums were wrapped in i tags, using ".a" I can isolate the individual links and names

```{code-cell} ipython3
wiki_content[1].find_all('i')
```

```{code-cell} ipython3
wiki_content[1].find('i').a
```

```{code-cell} ipython3
wiki_content[1].find('i').a.string
```

## I test looping through wiki_content[1] to make sure I can get all the names

```{code-cell} ipython3
album_names = [i.a.string for i in wiki_content[1].find_all('i')]
```

```{code-cell} ipython3
album_names
```

## I then apply this to the entire wiki_content list

```{code-cell} ipython3
album_names = [i.a.string for name in wiki_content 
               for i in name.find_all('i') 
               if i.a and i.a.string]
```

## I test getting the links and concatenate the front of the wiki website

```{code-cell} ipython3
wiki_content[1].find('i').a
```

```{code-cell} ipython3
wiki_content[1].find('i').a['href']
```

```{code-cell} ipython3
url_start = "https://en.wikipedia.org"
```

```{code-cell} ipython3
test_url = url_start + wiki_content[1].find('i').a['href']
test_url
```

## I test getting the runtime and year released from the constructed link

```{code-cell} ipython3
test_html = requests.get(test_url).content
test_info = BeautifulSoup(test_html,'html.parser')
```

```{code-cell} ipython3
test_info.find('span', class_="min")
```

```{code-cell} ipython3
test_min = test_info.find('span', class_="min").string
test_min = int(test_min.strip())
test_min
```

```{code-cell} ipython3
test_info.find('span', class_='bday dtstart published updated itvstart').string
```

```{code-cell} ipython3
test_date = test_info.find('span', class_='bday dtstart published updated itvstart').string
test_date
```

```{code-cell} ipython3
test_year = test_date.split('-')[0]
test_year
```

## I first construct all the links

```{code-cell} ipython3
album_links = [url_start + i.a['href'] 
               for link in wiki_content 
               for i in link.find_all('i') 
               if i.a and 'href' in i.a.attrs]
```

```{code-cell} ipython3
album_links[1]
```

## After finding out how to extract the run time I use a loop to do requests through all associated links

```{code-cell} ipython3
album_years = []
```

```{code-cell} ipython3
time_size = []
```

```{code-cell} ipython3
run_time = []
```

I also added a time_size catagory to help group by times

```{code-cell} ipython3
for test_url in album_links:
    
    test_html = requests.get(test_url).content
    test_info = BeautifulSoup(test_html,'html.parser')
    try: 
        test_min = test_info.find('span', class_="min").string
        test_date = test_info.find('span', class_='bday dtstart published updated itvstart').string
        test_year = test_date.split('-')[0]
        
        test_year = int(test_year.strip())
        album_years.append(test_year)
        
        test_min = int(test_min.strip())
        run_time.append(test_min)
        
        
        if test_min < 20:
            time_size.append('Very Short')
        elif test_min >= 20 and test_min < 35:
            time_size.append('Short')
        elif test_min >= 35 and test_min < 60:
            time_size.append('Average')
        elif test_min >= 60:
            time_size.append('Long')
        else:
            time_size.append(pd.NA)
    except:
        run_time.append(pd.NA)
        time_size.append(pd.NA)
        album_years.append(pd.NA)
```

```{code-cell} ipython3
albums_df = pd.DataFrame({'Album Name':album_names, 'Release Year':album_years, 'Length Rating':time_size, 'Run-Time(Mins)':run_time,'links':album_links})
```

## I drop all N/A's and duplicates

```{code-cell} ipython3
albums_df
```

## I start cleaning my data br removing all na's and duplicates

```{code-cell} ipython3
albums_clean_df = albums_df.dropna()
```

```{code-cell} ipython3
albums_clean_df.describe()
```

```{code-cell} ipython3
albums_clean_df =albums_clean_df.drop_duplicates()
```

```{code-cell} ipython3
albums_clean_df
```

```{code-cell} ipython3
albums_clean_df.describe()
```

## I noticed we really only get relevant data from 2015, so I drop anything earlier than 2015

```{code-cell} ipython3
albums_clean_df['Release Year'].value_counts()
```

```{code-cell} ipython3
albums_clean_df = albums_clean_df[albums_clean_df['Release Year'] >= 2015]
```

```{code-cell} ipython3
albums_clean_df['Release Year'].value_counts()
```

```{code-cell} ipython3
albums_clean_df['Length Rating'].value_counts()
```

```{code-cell} ipython3
albums_clean_df['Run-Time(Mins)'].mean()
```

```{code-cell} ipython3
albums_clean_df.groupby('Length Rating')['Run-Time(Mins)'].mean()
```

```{code-cell} ipython3
sns.lineplot(data=albums_clean_df,x='Release Year', y='Run-Time(Mins)')
```

```{code-cell} ipython3
sns.catplot(data=albums_clean_df,x='Release Year', y='Run-Time(Mins)',kind='bar', hue='Length Rating')
```

# Summery

+++

Overall, top chart albums stay around 45 minutes.  If we combine counts of small ,very small, and long albums we come to 122 which is still lower than average at 175.  I would say this leads to inconclusive data, if anything this may say more about the amount of work the music indusry puts into projects. Removing duplicates was useful but It would be more useful to observe months in the year to see if theres any dominating artists that write longer or shorter albums.

```{code-cell} ipython3
albums_clean_df.to_csv('Album_Runtime.csv', index=False)
```

```{code-cell} ipython3

```
