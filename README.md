#     README - Project Team 5
#     Wine Classification  
#     Project 2 


## Overview

The wine industry, is the selected industry for our project.  Specifically, we're focused on the wines that are produced in Spain.  Our dataset is one from Kaggle, the "Spanish Wine Quality Dataset". 

fedesoriano. (April 2022). Spanish Wine Quality Dataset. Retrieved Thursday, May 23, 2024 from https://www.kaggle.com/datasets/fedesoriano/spanish-wine-quality-dataset

It contains 7500 rows, but after some cleaning (dropna), the rows remaining comes out to 6329.

### Featured Winery - Contino

![Contino_winery](Project-2_EDA/images/Contino_Winery.png)

https://contino.es/en/winery/

A tidbit.....
Contino, on a percentage basis, has the most number of rows in our dataset (414).  More than any other winery represented.

Video 2:23 w/subtitles https://youtu.be/UDelPCc1dC4


## Questions

The goal or questions we aim to answer, gleened from our intial review of the columns within our selected dataset are...

     o  Determine the best model configurations to predict price
     o  Determine the best model configurations to predict a rating
     o  Through iterative means, can model scores be improved, i.e. using different scalers?
     o  Ultimately, using our model(s) here, restaurants wishing to serve Spanish wines can gleen an understanding on price vs. rating 
          when selecting wines for their inventory.   
 
     
Our first attempt at answering the above has revealed the following and provided the confidence we needed to ultimately select this dataset.  This forms our basis to improve on these numbers.

### Initial Scores on y=price

![Price Scores.....](Project-2_EDA/images/price_scores.png)


### Initial Scores on y=rating

![Rating Scores.....](Project-2_EDA/images/rating_scores.png)


### Initial Scores on y=type   

![Type Scores.....](Project-2_EDA/images/type_scores.png)



## Further Exploration

As mentioned previously, work to improve the model scores will continue.  There are other interesting columns in this dataset which may contribute or hinder this effort. Of course it is all in the details.  

### Regions

There are columns describing the regions (of Spain) where the wines are produced.  This column was excluded during our intial review, but may certainly play an important aspect going forward.  The top five regions according to our data analysis (value counts), reveals:

![region_stats.....](Project-2_EDA/images/region_stats.png)


These top five regions represent 73% of the total cleaned data in our analysis.  This analysis is corraborated by the following article describing the top six wine producing regions in Spain. https://www.lovetoknow.com/food-drink/wine/spain-wine-regions-map

Below is a map of the these regions pulled from the article.....

![Below is a map of the these regions pulled from the article.....](Project-2_EDA/images/SW_Regions.png)

Correlation to the map as decribed in the article are:

Rioja region in the dataset is in the Upper Ebro (map). 

Both Ribera del Duero and Toro regions are in the Duero Valley (map).

Priorato (dataset) is in the Catalunya (map).

Vino de Espana (dataset) is generally in multiple regions.

Of course other columns exist that may influence our model scores as we proceed with the exploration.  As touched on in the beginning, there is a column on wineries, from which Contino was found to have the most representation (rows) in our data.




## Price Predictons through Prophet

This section focuses on our effort to use Prophet to perform predictions on pricing.  Our approach entailed identifying a couple wines from our featured winery - Contino.  According to Contino's website, they produce eight different varieties. 
Our dataset reflected four of these wines as listed below.

![Contino_Wine](Project-2_EDA/images/ListOfWines.png) 

The wines selected for these price predictions are 

   1) Vina del Olivo (202 rows)
   2) Rioja Graciano  (6 rows)

A wine with several rows of data, and another with limited rows.  

To perform Prophet predictons, a datetime series dataframe is needed.  However, the only column that comes close to a date in our dataset, is the year column.  This hardly comes close to a date.  Therefore, massaging of the year column is in order. First, a dataframe containing only records representing the Contino winery was created. Second, after the year had all NAN/null rows removed, it was appended with a year-end date, by appending December 31.  For example: if the year column had 2004 as the year, it was appended with -12-31.  This turned the year into a full date as a string of 2004-12-31, thereby providing the prerequisite date format to proceed.  The next obstacle to overcome was Prophet's lack of support for a forcasting frequency by year.  The limited yearly data we had did not provide a decent basis for our forcasting. A forecast was predicted with the limited data available on both a monthly and yearly frequency going out three years.  The resulting graphs didn't portray anything of value.  Hence, more data massaging was necessary.  

Our problem was essentially a lack of data, so next was an effort to remedy such, by providing more data to the model.  Through some Pandas resampling code, a full 365/6 day year (one row for each day) was generated. This was done for every year of data that our dataset started with, and continued through the last year.  Now we have the sample data needed for better forcasting.  The resample code provides the ability to forward fill or backward fill our data column (price in this case).  This means that for all the new rows the resmaple adds, the price column will be filled with the price contained in the next row that exists looking forward (ffill).  Backward fill (bfill) does the opposite, by filling the price from the last previous row to each row added up to the next pre-existing row (which will be used subsequently in the next missing group of dates).  

For our first wine, Vina del Olivo, the Prophet graph below displays the history (both existing and added/padded forward fill rows from the resampling), as well as the forecast staring for year 2018 through 2020.  The sampling data is daily, with period and frequency at 36 and monthly respectively.  

![Vina de Olivo - ffill](Project-2_EDA/images/Prophet_Project2_3.png)

Please note:  The black lines are the observed data points stacked next to each other as they represent the daily (sampled) data.  

Our next graph is for the same wine, but the data samples are backward filled rather than forward filled as in the previous graph. 

ADD TEXT FOR THE REMAINING SLIDES.....

![Vina de Olivo - bfill](Project-2_EDA/images/Prophet_Project2_4.png)



![Vina de Olivo - bfill_qtr](Project-2_EDA/images/Prophet_Project2_5.png)



![Rioja Graciano - bfill_qtr](Project-2_EDA/images/Prophet_Project2_6.png)



![Rioja Graciano - bfill](Project-2_EDA/images/Prophet_Project2_7.png)


Duplicate above (fix)

![Rioja Graciano - bfill_month](Project-2_EDA/images/Prophet_Project2_8.png)



![Rioja Graciano - ffill_month](Project-2_EDA/images/Prophet_Project2_9.png)



![Rioja Graciano - ffill_qtr](Project-2_EDA/images/Prophet_Project2_10.png)


# Summary

This concludes the EDA at this point and will be incorporated into our final presentation.

  ![Contino_Wine](Project-2_EDA/images/Contino_Wine.png)



