# House Price Prediction Analytic 
## <p align="right">[Portfolio Main Page](https://github.com/WengWeng0410/Weng_Portfolio)</p>

In the housing market, comparative market analysis (CMA) is used to estimate value of a house with the aim of helping the sellers to set the house prices as well as buyers to make competitive offers. CMA is done by considering a number of features, such as location, years of the house, sq ft, view, grades and other features.

In this project, a machine learning model is developed to predict housing prices in King County, USA based on the properties' features such as sq ft, number of bedrooms and bathrooms, grades of the properties, view, etc.

Prior to the model development, EDA is carried out to understand and extract insights from the features. This is also to identify features that are valuable to include in the prediction model.

**Business Question**: What is the selling price of a house based on given a set of house' features?

## Code and Resources Used

**Python Version:** 3.7 <br>
**Packages:** numpy, pandas, seaborn, matplotlib, sklearn <br>
**IDE:** Google Colab <br> 
**Dataset:** https://www.kaggle.com/harlfoxem/housesalesprediction <br>
**Python Script:** [Notebook](https://drive.google.com/file/d/1txzSLXuTrfLPW_BB_SrXs6uqLo4Gdgl9/view?usp=sharing)

## Data Gathering

Data Collection

The dataset used in this project can be downloaded from Kaggle (link as shown above). The dataset contains 21613 records, each represents a house sold in betweem May 2014 to May 2015. There are 21 features in this dataset. Out of 21, there is only 1 feature with non numeric value, i.e., date, representing the selling date of a house. All the features are as shown follows.

* id - Unique ID for each house sold
* date - Date of the house sold
* price - Price of each house sold
* bedrooms - Number of bedrooms
* bathrooms - Number of bathrooms, where .5 accounts for a room with a toilet but no shower
* sqft_living - Square footage of the apartments interior living space
* sqft_lot - Square footage of the land space
* floors - Number of floors
* waterfront - A dummy variable for whether the apartment was overlooking the waterfront or not
* view - An index from 0 to 4 of how good the view of the property was
* condition - An index from 1 to 5 on the condition of the house,
* grade - An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.
* sqft_above - The square footage of the interior housing space that is above ground level
* sqft_basement - The square footage of the interior housing space that is below ground level
* yr_built - The year the house was initially built
* yr_renovated - The year of the house’s last renovation
* zipcode - What zipcode area the house
* lat - Lattitude
* long - Longitude
* sqft_living15 - The square footage of interior housing living space for the nearest 15 neighbors
* sqft_lot15 - The square footage of the land lots of the nearest 15 neighbors

## EDA

In this section, exploratory analysis on the dataset will be carried out to get some observations/insights.

Other than that, certain features are also preprocessed to better understand the dataset and prepare for the machine learning model training. The preprocessed features as follows

* Two houses with more than 10 bedrooms. As for the house with 33 bedrooms, it has only 1 floor and basement. Besides, the sqft living is also small (only 1620 as compared to 3000 for the house with 11 bedroom) which is not likely to have 33 bedrooms. Hence, the record with 33 bedroom is considered as outliner and removed.
* The yr_renovated feature (storing year of renovation) is preproccessed in order to identify if renovation has been done (Yes/No).
* The sqft_basement feature (storing are of basement) is preprocessed in order to to identify if a house has a basement (Yes/No)

#### Sqft of Living (with Different Number of Bedrooms) vs House Price
![](/images/1.HSP_bedroom.png)

Data shows that the houses sold between 2014 - 2015 are mostly with 2 - 4 bedrooms. Besides, it shows that houses with more bedrooms are wih bigger size on sqft living and able to sell at a higher price.

#### Sqft of Living (with Different Number of Floors) vs House Price
![](/images/2.HSP_floors.png)

Data shows that the houses sold between 2014 - 2015 are mostly in between 1 - 2 floors. Besides, there is no significant finding on the house price with respect to the number of floors.

#### Sqft of Living (with and without Renovation) vs House Price
![](/images/3.HSP_renovation.png)

#### Houses with Renovation vs Year Renovation
![](/images/4.HSP_renovationYear.png)

It can be seen that, most of the houses are not renovated (20699 units). There are only 914 houses renovated. It is noted that there are more houses renovated since 1979 and the year with highest number of house renovated is 2014.
Also, houses with renovation are able to sell with a higher price.

#### Sqft of Living (with and without Waterfront) vs House Price
![](/images/5.HSP_waterfront.png)

There are only 163 houses (out of 21613) with waterfront. In fact, the houses with waterfront are able to sell in a higher price.

#### Sqft of Living (with and without View) vs House Price
![](/images/6.HSP_view.png)

Data shows that most of the houses are with poor view. Also, it is noted that houses with better view are able to sell at a higher price.

#### Sqft of Living (with Different Grade) vs House Price
![](/images/7.HSP_grade.png)

Data shows that houses built are mostly rate 7 and above. Besides, it also shows that houses with high sqft living are with better grade of construction and design. Thus, the higher the selling price.

#### Sqft of Living (with Different Number of Bathrooms) vs House Price
![](/images/8.HSP_bathroom.png)

Most of the houses are with 1 - 3 full bathrooms. Besides, the houses with more bathrooms are with higher selling price.

#### Sqft of Living (with Different Condition) vs House Price
![](/images/9.HSP_condition.png)

Data shows that houses with better rating on the condition are able to sell at a higher prices, given the same sqft of living. Also, most of the houses are rated at least 3 and above.

#### Sqft of Living of Nearest 15 Neighbor (with and without View) vs House Price
![](/images/10.HSP_neighbor_view.png)

Data shows that the higher the sqft living of 15 neighbors, the higher the selling price of a house.

#### Sqft of Living and Sqft Basement (with and without Basement) vs House Price
![](/images/11.HSP_sqlivingVSbasement.png)

Data shows that houses with higher sqft living are more likely to have a basement, thus higher the houses prices. Besides, houses with basement are with higher prices, given the same sqft above the ground.

#### Latitude and Longitude of Houses (with and without Basement) vs House Price
![](/images/12.HSP_latVSlong(basement).png)

#### Latitude and Longitude (with Different Grade) vs House Price
![](/images/13.HSP_latVSlong(grade).png)

#### Latitude and Longitude of Houses (with Different View) vs House Price
![](/images/14.HSP_latVSlong(view).png)

#### Latitude and Longitude of Houses (with and without Waterfront) vs House Price
![](/images/15.HSP_latVSlong(waterfront).png)

#### Latitude and Longitude of Houses (with Different Condition) vs House Price
![](/images/16.HSP_latVSlong(condition).png)

Data shows that houses that are located at latitude of 47.5 to 47.8 and longitude of -122.4 to -122 are with higher prices, better grade (level of construction and design), view and condition of the properties, and with waterfront and basement.

#### Zipcode of Houses vs House Price
![](/images/17.HSP_zipcode.png)

Given the same zipcode, the better the condition, grade and view of a house, the higher the selling price.

#### Year Build of Houses vs House Price
![](/images/18.HSP_yrbuilt.png)

Suprisingly, houses built before year 2000 are with better condition. Besides, the many houses that are built after 1980 are of better quality.
As for the view of the houses, the better the view, the higher the selling price regardless of the year built of the houses.

### Summary of the rest of features
As for id, sqft_lot, sqft_lot15 and date, all these features did not have significant relationship with the selling price. Hence, they will not be considered for the machine learning model in the next section. 

## Model Building

In this section, three models with different methods that can make a prediction are developed. This is with the aim of identifying which model can performance the best. It should be noted that R-squared will be used as the performance metric to measure the performance of the developed models. 

Other than that, certain features are also preprocessed before feeding as input to the models, as follows

* MixMaxScaler from sklearn.preprocessing is used to scale values of 'price','sqft_living','sqft_basement','sqft_above', 'sqft_living15', 'lat', 'long' into rage of 0 to 1. Reason being this feature is measured at different scale and do not contribute equally in model training and may ended with creating a bias.
* train_test_split from sklearn.model_selection is also used to separat the dataset into training and testing set of data. In this project, 80% of the dataset is used as training set and the remaining 20% is used as testing set.

Three models have been selected:
* LinearRegression
* RandomForestRegressor
* XGBRegressor

## Performane Evaluation

Based on the result, the performance of <br> 
Linear Regression is with the R_ Square score of **69.13%**. <br>
Random Forest Regressor is with the R_ Square score of **88.80%**. <br>
XGBoost Regressor is with the R_ Square score of **89.27%**. <br>

The R_Squared value of XGBoost Regressor gives the best performance, followed by Random Forest Regressor and Linear Regression.


