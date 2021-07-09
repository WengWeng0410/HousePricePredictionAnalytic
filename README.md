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
