```python
#pip install xlsxwriter
```


```python
#pip install pandoc
```


```python
#pip install dash
```


```python
#pip install folium
```

# Sample Superstore

In today's dynamic business landscape, data-driven decision-making is paramount for staying competitive and maximizing operational efficiency. This project aims to conduct a comprehensive analysis of Using the sample superstore dataset from the tableau website (https://community.tableau.com/s/question/0D54T00000CWeX8SAL/sample-superstore-sales-excelxls), to derive valuable insights that can inform strategic business decisions. The dataset encompasses a range of features, including order details, customer information, and product-specific metrics, providing a holistic view of our sales operations.


#### Business Goal 1: Understand Sales Performance Over Time

What is the overall trend in sales over time?

How do sales vary by month or season?

Are there any notable trends or patterns in order and ship dates?


#### Business Goal 2: Optimize Geographic Sales Strategies

What are the top-selling regions or cities?

Is there a correlation between sales and the geographic location (region, city, state)?


#### Business Goal 3: Segment Customers for Targeted Marketing

How are sales distributed among different customer segments?

What types of products are popular among different customer segments?

How frequently do customers make purchases?


#### Business Goal 4: Maximize Product and Category Profitability

What are the most and least profitable products/categories?

Which products contribute the most to profit?

Are there specific products or categories where discounts are more effective?


#### Business Goal 5: Optimize Operational Efficiency

Does the choice of shipping mode affect sales or profit?

Are certain shipping modes more popular in specific regions?

How does discount impact sales and profit?


#### Business Goal 6: Understand Customer Behavior

What is the average order size and value?

How does the quantity of products ordered relate to sales and profit?

Do discounts correlate with higher sales but lower profit?


```python
#import libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn
import statsmodels.api as sm
import plotly.express as px
import folium
from scipy import stats
import xlsxwriter

# Enable inline plotting in Jupyter Notebooks
%matplotlib inline

```


```python
#import dataset
df = pd.read_csv("superstore.csv")
```


```python
#dataframe dimension
df.shape
```




    (9994, 21)




```python
df.head(5)
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
      <th>Row ID</th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Ship Date</th>
      <th>Ship Mode</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>Segment</th>
      <th>Country</th>
      <th>City</th>
      <th>...</th>
      <th>Postal Code</th>
      <th>Region</th>
      <th>Product ID</th>
      <th>Category</th>
      <th>Sub-Category</th>
      <th>Product Name</th>
      <th>Sales</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>CA-2016-152156</td>
      <td>11/8/2016</td>
      <td>11/11/2016</td>
      <td>Second Class</td>
      <td>CG-12520</td>
      <td>Claire Gute</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Henderson</td>
      <td>...</td>
      <td>42420</td>
      <td>South</td>
      <td>FUR-BO-10001798</td>
      <td>Furniture</td>
      <td>Bookcases</td>
      <td>Bush Somerset Collection Bookcase</td>
      <td>261.9600</td>
      <td>2</td>
      <td>0.00</td>
      <td>41.9136</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>CA-2016-152156</td>
      <td>11/8/2016</td>
      <td>11/11/2016</td>
      <td>Second Class</td>
      <td>CG-12520</td>
      <td>Claire Gute</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Henderson</td>
      <td>...</td>
      <td>42420</td>
      <td>South</td>
      <td>FUR-CH-10000454</td>
      <td>Furniture</td>
      <td>Chairs</td>
      <td>Hon Deluxe Fabric Upholstered Stacking Chairs,...</td>
      <td>731.9400</td>
      <td>3</td>
      <td>0.00</td>
      <td>219.5820</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>CA-2016-138688</td>
      <td>6/12/2016</td>
      <td>6/16/2016</td>
      <td>Second Class</td>
      <td>DV-13045</td>
      <td>Darrin Van Huff</td>
      <td>Corporate</td>
      <td>United States</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>90036</td>
      <td>West</td>
      <td>OFF-LA-10000240</td>
      <td>Office Supplies</td>
      <td>Labels</td>
      <td>Self-Adhesive Address Labels for Typewriters b...</td>
      <td>14.6200</td>
      <td>2</td>
      <td>0.00</td>
      <td>6.8714</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>US-2015-108966</td>
      <td>10/11/2015</td>
      <td>10/18/2015</td>
      <td>Standard Class</td>
      <td>SO-20335</td>
      <td>Sean O'Donnell</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Fort Lauderdale</td>
      <td>...</td>
      <td>33311</td>
      <td>South</td>
      <td>FUR-TA-10000577</td>
      <td>Furniture</td>
      <td>Tables</td>
      <td>Bretford CR4500 Series Slim Rectangular Table</td>
      <td>957.5775</td>
      <td>5</td>
      <td>0.45</td>
      <td>-383.0310</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>US-2015-108966</td>
      <td>10/11/2015</td>
      <td>10/18/2015</td>
      <td>Standard Class</td>
      <td>SO-20335</td>
      <td>Sean O'Donnell</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Fort Lauderdale</td>
      <td>...</td>
      <td>33311</td>
      <td>South</td>
      <td>OFF-ST-10000760</td>
      <td>Office Supplies</td>
      <td>Storage</td>
      <td>Eldon Fold 'N Roll Cart System</td>
      <td>22.3680</td>
      <td>2</td>
      <td>0.20</td>
      <td>2.5164</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df.tail()
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
      <th>Row ID</th>
      <th>Order ID</th>
      <th>Order Date</th>
      <th>Ship Date</th>
      <th>Ship Mode</th>
      <th>Customer ID</th>
      <th>Customer Name</th>
      <th>Segment</th>
      <th>Country</th>
      <th>City</th>
      <th>...</th>
      <th>Postal Code</th>
      <th>Region</th>
      <th>Product ID</th>
      <th>Category</th>
      <th>Sub-Category</th>
      <th>Product Name</th>
      <th>Sales</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>9989</th>
      <td>9990</td>
      <td>CA-2014-110422</td>
      <td>1/21/2014</td>
      <td>1/23/2014</td>
      <td>Second Class</td>
      <td>TB-21400</td>
      <td>Tom Boeckenhauer</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Miami</td>
      <td>...</td>
      <td>33180</td>
      <td>South</td>
      <td>FUR-FU-10001889</td>
      <td>Furniture</td>
      <td>Furnishings</td>
      <td>Ultra Door Pull Handle</td>
      <td>25.248</td>
      <td>3</td>
      <td>0.2</td>
      <td>4.1028</td>
    </tr>
    <tr>
      <th>9990</th>
      <td>9991</td>
      <td>CA-2017-121258</td>
      <td>2/26/2017</td>
      <td>3/3/2017</td>
      <td>Standard Class</td>
      <td>DB-13060</td>
      <td>Dave Brooks</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Costa Mesa</td>
      <td>...</td>
      <td>92627</td>
      <td>West</td>
      <td>FUR-FU-10000747</td>
      <td>Furniture</td>
      <td>Furnishings</td>
      <td>Tenex B1-RE Series Chair Mats for Low Pile Car...</td>
      <td>91.960</td>
      <td>2</td>
      <td>0.0</td>
      <td>15.6332</td>
    </tr>
    <tr>
      <th>9991</th>
      <td>9992</td>
      <td>CA-2017-121258</td>
      <td>2/26/2017</td>
      <td>3/3/2017</td>
      <td>Standard Class</td>
      <td>DB-13060</td>
      <td>Dave Brooks</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Costa Mesa</td>
      <td>...</td>
      <td>92627</td>
      <td>West</td>
      <td>TEC-PH-10003645</td>
      <td>Technology</td>
      <td>Phones</td>
      <td>Aastra 57i VoIP phone</td>
      <td>258.576</td>
      <td>2</td>
      <td>0.2</td>
      <td>19.3932</td>
    </tr>
    <tr>
      <th>9992</th>
      <td>9993</td>
      <td>CA-2017-121258</td>
      <td>2/26/2017</td>
      <td>3/3/2017</td>
      <td>Standard Class</td>
      <td>DB-13060</td>
      <td>Dave Brooks</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Costa Mesa</td>
      <td>...</td>
      <td>92627</td>
      <td>West</td>
      <td>OFF-PA-10004041</td>
      <td>Office Supplies</td>
      <td>Paper</td>
      <td>It's Hot Message Books with Stickers, 2 3/4" x 5"</td>
      <td>29.600</td>
      <td>4</td>
      <td>0.0</td>
      <td>13.3200</td>
    </tr>
    <tr>
      <th>9993</th>
      <td>9994</td>
      <td>CA-2017-119914</td>
      <td>5/4/2017</td>
      <td>5/9/2017</td>
      <td>Second Class</td>
      <td>CC-12220</td>
      <td>Chris Cortes</td>
      <td>Consumer</td>
      <td>United States</td>
      <td>Westminster</td>
      <td>...</td>
      <td>92683</td>
      <td>West</td>
      <td>OFF-AP-10002684</td>
      <td>Office Supplies</td>
      <td>Appliances</td>
      <td>Acco 7-Outlet Masterpiece Power Center, Wihtou...</td>
      <td>243.160</td>
      <td>2</td>
      <td>0.0</td>
      <td>72.9480</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Print the columns of the DataFrame
print(df.columns)
```

    Index(['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode',
           'Customer ID', 'Customer Name', 'Segment', 'Country', 'City', 'State',
           'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category',
           'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit'],
          dtype='object')
    


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9994 entries, 0 to 9993
    Data columns (total 21 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Row ID         9994 non-null   int64  
     1   Order ID       9994 non-null   object 
     2   Order Date     9994 non-null   object 
     3   Ship Date      9994 non-null   object 
     4   Ship Mode      9994 non-null   object 
     5   Customer ID    9994 non-null   object 
     6   Customer Name  9994 non-null   object 
     7   Segment        9994 non-null   object 
     8   Country        9994 non-null   object 
     9   City           9994 non-null   object 
     10  State          9994 non-null   object 
     11  Postal Code    9994 non-null   int64  
     12  Region         9994 non-null   object 
     13  Product ID     9994 non-null   object 
     14  Category       9994 non-null   object 
     15  Sub-Category   9994 non-null   object 
     16  Product Name   9994 non-null   object 
     17  Sales          9994 non-null   float64
     18  Quantity       9994 non-null   int64  
     19  Discount       9994 non-null   float64
     20  Profit         9994 non-null   float64
    dtypes: float64(3), int64(3), object(15)
    memory usage: 1.6+ MB
    


```python
# Check for missing values
missing_values = df.isnull()

# Count the number of missing values in each column
missing_counts = missing_values.sum()

# Print the columns with missing values and their respective counts
print(missing_counts[missing_counts > 0])
```

    Series([], dtype: int64)
    


```python
#summary statistics
df.describe()
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
      <th>Row ID</th>
      <th>Postal Code</th>
      <th>Sales</th>
      <th>Quantity</th>
      <th>Discount</th>
      <th>Profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9994.000000</td>
      <td>9994.000000</td>
      <td>9994.000000</td>
      <td>9994.000000</td>
      <td>9994.000000</td>
      <td>9994.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4997.500000</td>
      <td>55190.379428</td>
      <td>229.858001</td>
      <td>3.789574</td>
      <td>0.156203</td>
      <td>28.656896</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2885.163629</td>
      <td>32063.693350</td>
      <td>623.245101</td>
      <td>2.225110</td>
      <td>0.206452</td>
      <td>234.260108</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1040.000000</td>
      <td>0.444000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>-6599.978000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2499.250000</td>
      <td>23223.000000</td>
      <td>17.280000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.728750</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4997.500000</td>
      <td>56430.500000</td>
      <td>54.490000</td>
      <td>3.000000</td>
      <td>0.200000</td>
      <td>8.666500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7495.750000</td>
      <td>90008.000000</td>
      <td>209.940000</td>
      <td>5.000000</td>
      <td>0.200000</td>
      <td>29.364000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9994.000000</td>
      <td>99301.000000</td>
      <td>22638.480000</td>
      <td>14.000000</td>
      <td>0.800000</td>
      <td>8399.976000</td>
    </tr>
  </tbody>
</table>
</div>



### Business Goal 1: Understand Sales Performance Over Time

What is the overall trend in sales over time?



```python
# Converting 'Order Date' to datetime and set it as the index
df['Order Date'] = pd.to_datetime(df['Order Date'])
df.set_index('Order Date', inplace=True)
```


```python
# Resampling the data by month and calculate the total sales for each month
monthly_sales = df['Sales'].resample('M').sum()
```


```python
# Plotting the overall trend in sales over time
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, marker='o', linestyle='-', color='b')
plt.title('Overall Trend in Sales Over Time')
plt.xlabel('Order Date')
plt.ylabel('Total Sales')
plt.grid(True)
plt.show()
```


    
![png](output_18_0.png)
    


The 'Order Date' is converted to a datetime format, and the dataset is indexed by the order date.
Monthly sales are calculated using resampling, and the trend is visualized using a line plot.
Sales Variation by Month or Season (1.2):

How do sales vary by month or season?


```python
# Calculate mean and standard deviation of sales for each month
monthly_sales_mean = df['Sales'].resample('M').mean()
monthly_sales_std = df['Sales'].resample('M').std()
```


```python
# Plotting sales variation by month and season
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales_mean, marker='o', linestyle='-', label='Mean Sales')
plt.fill_between(monthly_sales_std.index, monthly_sales_mean - monthly_sales_std, monthly_sales_mean + monthly_sales_std, alpha=0.2, label='Sales Std Dev')
plt.title('Sales Variation by Month/Season')
plt.xlabel('Order Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

```


    
![png](output_22_0.png)
    


The mean and standard deviation of sales for each month are calculated and plotted to show variation.

Are there any notable trends or patterns in order and ship dates?


```python
# Extracting relevant columns for order and ship dates
#order_ship_dates = df[['Order Date', 'Ship Date']]
# Resampling the data by month to get the count of orders and shipments for each month
#monthly_orders = order_ship_dates['Order Date'].resample('M').count()
#monthly_shipments = order_ship_dates['Ship Date'].resample('M').count()
```


```python
# Plotting the number of orders and shipments over time
#plt.figure(figsize=(12, 6))
#plt.plot(monthly_orders, marker='o', linestyle='-', label='Orders', color='b')
#plt.plot(monthly_shipments, marker='o', linestyle='-', label='Shipments', color='r')
#plt.title('Number of Orders and Shipments Over Time')
#plt.xlabel('Order Date')
#plt.ylabel('Count')
#plt.legend()
#plt.grid(True)
#plt.show()
```

### Business Goal 2: Optimize Geographic Sales Strategies

What are the top-selling regions or cities?


```python
# Grouping the data by Region and calculate the total sales for each region
region_sales = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)

# Plotting the top-selling regions
plt.figure(figsize=(10, 6))
region_sales.plot(kind='bar', color='skyblue')
plt.title('Top-Selling Regions')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.show()

```


    
![png](output_28_0.png)
    


The data is grouped by 'Region' and 'City,' and the total sales for each region and city are calculated.
Bar plots are created to visualize the top-selling regions and cities.


```python
# Group the data by City and calculate the total sales for each city
city_sales = df.groupby('City')['Sales'].sum().sort_values(ascending=False).head(10)

# Plotting the top-selling cities
plt.figure(figsize=(14, 6))
city_sales.plot(kind='bar', color='salmon')
plt.title('Top-Selling Cities')
plt.xlabel('City')
plt.ylabel('Total Sales')
plt.show()

# 2.2 Is there a correlation between sales and the geographic location (region, city, state)?


```


    
![png](output_30_0.png)
    


Is there a correlation between sales and the geographic location (region, city, state)?


```python
# Grouping the data by Region and calculate the average sales for each region
region_avg_sales = df.groupby('Region')['Sales'].mean()

# Grouping the data by City and calculate the average sales for each city
city_avg_sales = df.groupby('City')['Sales'].mean()

# Plotting the correlation between average sales and regions/cities
plt.figure(figsize=(14, 6))
sns.scatterplot(x='Region', y='Sales', data=region_avg_sales.reset_index(), color='blue', label='Average Sales by Region')
sns.scatterplot(x='City', y='Sales', data=city_avg_sales.reset_index(), color='red', label='Average Sales by City')
plt.title('Correlation between Sales and Geographic Location')
plt.xlabel('Geographic Location')
plt.ylabel('Average Sales')
plt.xticks(rotation=90)
plt.legend()
plt.show()
```


    
![png](output_32_0.png)
    


The data is grouped by 'Region' and 'City,' and the average sales for each region and city are calculated.
Scatter plots are created to visualize the correlation between average sales and regions/cities.

### Business Goal 3: Segment Customers for Targeted Marketing

How are sales distributed among different customer segments?



```python
# Grouping the data by Segment and calculate the total sales for each segment
segment_sales = df.groupby('Segment')['Sales'].sum().sort_values(ascending=False)

# Plotting the sales distribution among different customer segments
plt.figure(figsize=(10, 6))
segment_sales.plot(kind='bar', color='purple')
plt.title('Sales Distribution Among Customer Segments')
plt.xlabel('Customer Segment')
plt.ylabel('Total Sales')
plt.show()

```


    
![png](output_35_0.png)
    


The data is grouped by 'Segment,' and the total sales for each segment are calculated.
A bar plot is created to visualize the distribution of sales among different customer segments.

What types of products are popular among different customer segments?


```python
# Group the data by Segment and Category, and calculate the total sales for each category in each segment
segment_category_sales = df.groupby(['Segment', 'Category'])['Sales'].sum().unstack()

# Plotting the types of products that are popular among different customer segments
plt.figure(figsize=(14, 6))
segment_category_sales.plot(kind='bar', stacked=False)
plt.title('Popular Products Among Customer Segments')
plt.xlabel('Customer Segment')
plt.ylabel('Total Sales')
plt.legend(title='Product Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

```


    <Figure size 1400x600 with 0 Axes>



    
![png](output_38_1.png)
    


The data is grouped by both 'Segment' and 'Category,' and the total sales for each category in each segment are calculated.
A stacked bar plot is created to visualize the types of products that are popular among different customer segments.

How frequently do customers make purchases?


```python
# Calculate the frequency of purchases for each customer (Customer ID)
purchase_frequency = df.groupby('Customer ID')['Order ID'].count()

# Plotting the frequency distribution of customer purchases
plt.figure(figsize=(12, 6))
sns.histplot(purchase_frequency, bins=30, color='green', kde=True)
plt.title('Frequency Distribution of Customer Purchases')
plt.xlabel('Number of Purchases')
plt.ylabel('Frequency')
plt.show()

```


    
![png](output_41_0.png)
    


.

### Business Goal 4: Maximize Product and Category Profitability

What are the most and least profitable products/categories?


```python
# Grouping the data by Product Name and calculate the total profit for each product
product_profit = df.groupby('Product Name')['Profit'].sum().sort_values(ascending=False)

# Plotting the most and least profitable products
plt.figure(figsize=(14, 6))
product_profit.head(10).plot(kind='bar', color='aqua', label='Most Profitable Products')
product_profit.tail(10).plot(kind='bar', color='tomato', label='Least Profitable Products')
plt.title('Most and Least Profitable Products')
plt.xlabel('Product Name')
plt.ylabel('Total Profit')
plt.legend()
plt.show()
```


    
![png](output_44_0.png)
    


The data is grouped by 'Product Name' and 'Category,' and the total profit for each product and category is calculated.
Bar plots are created to visualize the most and least profitable products and product categories.

Which products contribute the most to profit?


```python
# Grouping the data by Category and calculate the total profit for each category
category_profit = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)

# Plotting the most and least profitable product categories
plt.figure(figsize=(10, 6))
category_profit.plot(kind='bar', color='blue')
plt.title('Profit Distribution Among Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Total Profit')
plt.show()

# 4.2 Which products contribute the most to profit?

# Group the data by Product Name and calculate the average profit for each product
avg_product_profit = df.groupby('Product Name')['Profit'].mean()

# Plotting the products that contribute the most to profit
plt.figure(figsize=(14, 6))
avg_product_profit.sort_values(ascending=False).head(10).plot(kind='bar', color='orange')
plt.title('Products Contributing the Most to Profit')
plt.xlabel('Product Name')
plt.ylabel('Average Profit')
plt.show()


```


    
![png](output_47_0.png)
    



    
![png](output_47_1.png)
    


The data is grouped by 'Product Name,' and the average profit for each product is calculated.
A bar plot is created to visualize the products that contribute the most to profit.

Are there specific products or categories where discounts are more effective?


```python
# Grouping the data by Category and Discount, and calculate the average profit for each category and discount level
category_discount_profit = df.groupby(['Category', 'Discount'])['Profit'].mean().unstack()

# Plotting the effectiveness of discounts in different product categories
plt.figure(figsize=(14, 6))
sns.heatmap(category_discount_profit, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('Effectiveness of Discounts in Different Product Categories')
plt.xlabel('Discount')
plt.ylabel('Product Category')
plt.show()
```


    
![png](output_50_0.png)
    


The data is grouped by 'Category' and 'Discount,' and the average profit for each category and discount level is calculated.
A heatmap is created to visualize the effectiveness of discounts in different product categories.

### Business Goal 5: Optimize Operational Efficiency

Does the choice of shipping mode affect sales or profit?


```python
# Grouping the data by Shipping Mode and calculate the total sales and profit for each mode
shipping_mode_performance = df.groupby('Ship Mode')[['Sales', 'Profit']].sum()

# Plotting the impact of shipping mode on sales and profit
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Sales by Shipping Mode
axes[0].pie(shipping_mode_performance['Sales'], labels=shipping_mode_performance.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
axes[0].set_title('Sales Distribution by Shipping Mode')

# Profit by Shipping Mode
axes[1].pie(shipping_mode_performance['Profit'], labels=shipping_mode_performance.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
axes[1].set_title('Profit Distribution by Shipping Mode')

plt.show()
```


    
![png](output_53_0.png)
    


The data is grouped by 'Ship Mode,' and the total sales and profit for each mode are calculated.
Pie charts are created to visualize the distribution of sales and profit among different shipping modes.

Are certain shipping modes more popular in specific regions?


```python
# Creating a cross-tabulation between Region and Ship Mode
region_ship_mode_count = pd.crosstab(df['Region'], df['Ship Mode'])

# Plotting the popularity of shipping modes in different regions
plt.figure(figsize=(12, 6))
sns.heatmap(region_ship_mode_count, cmap='Blues', annot=True, fmt='d', linewidths=.5)
plt.title('Popularity of Shipping Modes in Different Regions')
plt.xlabel('Ship Mode')
plt.ylabel('Region')
plt.show()
```


    
![png](output_56_0.png)
    


A cross-tabulation is created between 'Region' and 'Ship Mode' to visualize the popularity of shipping modes in different regions using a heatmap.

How does discount impact sales and profit?


```python
# Grouping the data by Discount and calculate the average sales and profit for each discount level
discount_performance = df.groupby('Discount')[['Sales', 'Profit']].mean()

# Plotting the impact of discount on sales and profit
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Average Sales by Discount
axes[0].bar(discount_performance.index, discount_performance['Sales'], color='orange')
axes[0].set_title('Average Sales by Discount')
axes[0].set_xlabel('Discount')
axes[0].set_ylabel('Average Sales')

# Average Profit by Discount
axes[1].bar(discount_performance.index, discount_performance['Profit'], color='purple')
axes[1].set_title('Average Profit by Discount')
axes[1].set_xlabel('Discount')
axes[1].set_ylabel('Average Profit')

plt.show()

```


    
![png](output_59_0.png)
    


The data is grouped by 'Discount,' and the average sales and profit for each discount level are calculated.
Bar plots are created to visualize the impact of discount on average sales and profit.

### Business Goal 6: Understand Customer Behavior

What is the average order size and value?


```python
# Calculating the average order size (number of items per order)
average_order_size = df.groupby('Order ID')['Quantity'].sum().mean()

# Calculating the average order value
average_order_value = df.groupby('Order ID')['Sales'].sum().mean()

print(f"Average Order Size: {average_order_size:.2f} items")
print(f"Average Order Value: ${average_order_value:.2f}")
```

    Average Order Size: 7.56 items
    Average Order Value: $458.61
    

This means that, on average, each order consists of approximately 7.56 items, and the average order value is $458.61. These metrics provide insights into the typical size and value of orders, which can be valuable for understanding customer behavior and making informed business decisions.

How does the quantity of products ordered relate to sales and profit?


```python
# Scatter plot to visualize the relationship between quantity ordered and sales/profit
plt.figure(figsize=(12, 6))

# Quantity vs. Sales
plt.subplot(1, 2, 1)
sns.scatterplot(x='Quantity', y='Sales', data=df, color='blue')
plt.title('Quantity vs. Sales')
plt.xlabel('Quantity')
plt.ylabel('Sales')

# Quantity vs. Profit
plt.subplot(1, 2, 2)
sns.scatterplot(x='Quantity', y='Profit', data=df, color='green')
plt.title('Quantity vs. Profit')
plt.xlabel('Quantity')
plt.ylabel('Profit')

plt.tight_layout()
plt.show()
```


    
![png](output_65_0.png)
    


Scatter plots are created to visualize the relationship between the quantity of products ordered and sales/profit.

Do discounts correlate with higher sales but lower profit?


```python
# Calculate the correlation between discount and sales/profit
correlation_discount_sales = df['Discount'].corr(df['Sales'])
correlation_discount_profit = df['Discount'].corr(df['Profit'])

print(f"Correlation between Discount and Sales: {correlation_discount_sales:.2f}")
print(f"Correlation between Discount and Profit: {correlation_discount_profit:.2f}")

```

    Correlation between Discount and Sales: -0.03
    Correlation between Discount and Profit: -0.22
    

The correlation between discounts and sales/profit is calculated and printed.

### Correlation between Discount and Sales (-0.03):

The correlation between the discount given and the sales amount is close to zero (-0.03). This suggests a very weak negative correlation, indicating that there is little to no linear relationship between the discount and the total sales.

### Correlation between Discount and Profit (-0.22):

The correlation between the discount given and the profit is also negative but slightly stronger (-0.22). This suggests a weak negative correlation, implying that, on average, as discounts increase, profit tends to decrease, but the relationship is not very strong.


```python
import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Assuming df is your DataFrame

# Initialize the Dash app
app = dash.Dash(__name__)

# Business Goal 1: Sales Performance Over Time
monthly_sales = df.resample('M').agg({'Sales': 'sum'}).reset_index()
fig1 = px.line(monthly_sales, x='Order Date', y='Sales', title='Monthly Sales Over Time')

# Business Goal 2: Geographic Sales Strategies
region_sales = df.groupby('Region')['Sales'].sum().reset_index()
fig2 = px.bar(region_sales, x='Region', y='Sales', title='Sales Distribution by Region')

# Business Goal 3: Customer Segmentation
segment_sales = df.groupby('Segment')['Sales'].sum().reset_index()
fig3 = px.pie(segment_sales, values='Sales', names='Segment', title='Sales Distribution by Customer Segment')

# Business Goal 4: Product and Category Profitability
product_profit = df.groupby('Product Name')['Profit'].sum().reset_index()
fig4 = px.bar(product_profit.head(10), x='Product Name', y='Profit', title='Top 10 Profitable Products')

# Business Goal 5: Operational Efficiency
shipping_mode_performance = df.groupby('Ship Mode')[['Sales', 'Profit']].sum().reset_index()
fig5 = px.pie(shipping_mode_performance, values='Sales', names='Ship Mode', title='Sales Distribution by Shipping Mode')

# Business Goal 6: Customer Behavior
quantity_sales_profit = df.groupby('Quantity')[['Sales', 'Profit']].mean().reset_index()
fig6 = px.scatter(quantity_sales_profit, x='Quantity', y='Sales', size='Profit', title='Quantity vs. Sales and Profit')

# Layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='Sales Analysis Dashboard'),

    dcc.Graph(
        id='sales-over-time',
        figure=fig1
    ),

    dcc.Graph(
        id='sales-by-region',
        figure=fig2
    ),

    dcc.Graph(
        id='sales-by-segment',
        figure=fig3
    ),

    dcc.Graph(
        id='top-10-profitable-products',
        figure=fig4
    ),

    dcc.Graph(
        id='sales-by-shipping-mode',
        figure=fig5
    ),

    dcc.Graph(
        id='quantity-vs-sales-profit',
        figure=fig6
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

```



<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8050/"
    frameborder="0"
    allowfullscreen

></iframe>




```python

```
