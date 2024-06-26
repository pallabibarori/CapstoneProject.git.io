# CAPSTONE PROJECT ON

## 🛒 “MARKET BASKET ANALYSIS TO UNDERSTAND THE INVENTORY BETTER” 📊

### BY

**PALLABI BARORI** 

### Symbiosis Centre for Distance Learning (SCDL)  
**2021-2023**

---

## TABLE OF CONTENTS

- **CHAPTER 1: INTRODUCTION**
- **CHAPTER 2: ANALYSIS OF WORK DONE AND DESIGN**
- **CHAPTER 3: LEARNING EXPERIENCES ON BUSINESS/TECHNOLOGY**
- **CHAPTER 4: CONCLUSION**
- **CHAPTER 5: BIBLIOGRAPHY/REFERENCES**

---

## CHAPTER 1: INTRODUCTION

### 1.1 Objective
The retail industry heavily relies on Market Basket Analysis (MBA) to understand customer preferences and drive sales. This project aims to utilize MBA techniques such as Association Rule Mining and Apriori Algorithm to uncover purchasing patterns, optimize inventory management, and enhance decision-making for retailers.

### 1.2 Scope and Background
By analyzing transactional data and applying MBA techniques, this project seeks to identify frequently bought items, understand item associations, and provide actionable insights for retailers to improve sales strategies, inventory planning, and product placement.

---

## CHAPTER 2: ANALYSIS OF WORK DONE AND DESIGN

### 2.1 Analysis Overview: The main aim of the project is to analyze data using Market Basket Analysis to understand the pattern in which customers prefer buying items so that it can be used to cross-sell products to the customers. This can be used to optimize shelf space or to create combos and discounts. Further with the data the retailers will be able to identify the items which are frequently bought and can stock them more as compared to the ones which are not sold much, hence can upgrade the inventory.

### 2.2 Data Understanding: I have used fictional data from Kaggle (https://www.kaggle.com/) and have added and modified the columns as per my usage.
This data contains 999 observations and has 19 columns. They are as follows:
1.	Column A - Transaction ID: This column indicates the unique Order ID. It is alphanumeric in nature.
2.	Column B to Column P: These columns contain the data of items ordered by per customers.
3.	Column Q - City: This column gives us information on what item is bought from what city.
4.	Column R - Order-Data : This column gives us the date on which the item is bought. The dates cover year between 2015-2018. There are 440 unique date entries.
Majority of the entries in this column in written in DD-MM-YYYY format, but some are also present in DD/MM/YYYY format. For this project I will clean up the data and make the date column in DD-MM-YYY format and in ascending order.
5.	Column S - Region: This column gives us the idea of the region in which items are bought.
All the values under the Order ID in the data is unique, and there is no null data present.
-	All the Transaction IDs are unique and there are no Null values in any row.

### 2.3 Market Basket Analysis Technique: 
For this project I will be using Association rule mining and apriori algorithm.
Apriori Algorithm – This gives the frequent itemset
Association rule mining – This creates rule that if one item is bought what are the chances of other items being bought with it as well.

Table 1 
| Item         | Total count | Mean |
|--------------|-------------|------|
| Apple        | 384         | 38.4 |
| Bread        | 385         | 38.5 |
| Butter       | 421         | 42.1 |
| Cheese       | 404         | 40.4 |
| Corn         | 407         | 40.7 |
| Dill         | 398         | 39.8 |
| Eggs         | 385         | 38.5 |
| Ice-cream    | 410         | 41.0 |
| Kidney beans | 409         | 40.9 |
| Milk         | 406         | 40.6 |
| Nutmeg       | 40.1        | 40.1 |
| Onion        | 403         | 40.3 |
| Sugar        | 409         | 40.9 |
| Yogurt       | 420         | 42.0 |
| Chocolate    | 421         | 42.1 |

We will be applying Apriori alogorithm in the above mentioned data to find out the Support, Confidence and Lift of all the 15 items throughout 1000 transactions. According to Apriori algorithm, the subset of non-empty itemsets are also frequent. This would mean that if [Bread, Apple]  are bought frequently, then individually Apple and Bread are also bought frequently. 

•	Support – This is used to measure the frequency of an itemset from the data.
Support(A) = (Transactions with (A))/ Total Transactions
•	Confidence – It is used to measure the reliability of an association rule, that if X is purchased, Y will also be purchased.
Confidence (X,Y) = ( Transactions having both X and Y)/(Transactions of X)
•	Lift – It evaluates the extent to which the presence of item X influences that of Y.
Lift (X>Y) increase in the sale of  Y when X is sold.
Lift (Y>X) increase in the sale of X when Y is sold.
Lift (XY) = Confidence(X,Y)/Support(Y)

                  Let’s consider minimum support = 11%
                                                                       = minimum_support * itemset_count
                                                                       = (11/100) * 1000 = 110 

Commands used in Jupyter notebook:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from pandas.plotting import parallel_coordinates
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/pallabibarori/Desktop/MBA.csv')

print(df)

Order ID  Apple  Bread  Butter  Cheese   Corn   Dill   Eggs  Ice cream  \
0     OD1001  False   True    True   False  False   True  False       True   
1     OD1002   True  False    True   False  False   True   True       True   
2     OD1003  False   True   False   False  False   True   True       True   
3     OD1004   True  False   False    True   True  False  False      False   
4     OD1005  False  False    True   False   True  False  False      False   
..       ...    ...    ...     ...     ...    ...    ...    ...        ...   
995   OD1996   True  False   False   False  False  False   True       True   
996   OD1997   True  False    True    True  False   True   True       True   
997   OD1998  False   True   False    True  False   True   True       True   
998   OD1999  False  False   False   False  False  False  False      False   
999   OD2000   True   True    True   False  False  False   True      False   

     Kidney Beans   Milk  Nutmeg  Onion  Sugar  Yogurt  chocolate Oredr Date  \
0            True   True   False  False  False   False      False   02/01/15   
1            True   True   False   True   True    True      False   03/01/15   
2           False  False   False  False   True   False      False   06/01/15   
3           False   True    True   True  False    True       True   06/01/15   
4            True  False   False   True  False   False      False   06/01/15   
..            ...    ...     ...    ...    ...     ...        ...        ...   
995         False   True    True  False  False    True       True  9-29-2017   
996          True  False   False  False   True   False       True  9-29-2018   
997          True   True    True   True   True    True      False  9-29-2018   
998         False  False    True  False   True   False      False  9-29-2018   
999          True   True   False  False  False   False      False  9-29-2018   

  City   Region  
0         Trichy  Central  
1          Salem     West  
2     Coimbatore  Central  
3    Kanyakumari  Central  
4     Perambalur  Central  
..           ...      ...  
995      Tenkasi     East  
996  Krishnagiri  Central  
997   Perambalur  Central  
998        Theni  Central  
999   Perambalur  Central  

[1000 rows x 19 columns]

df.drop(df.columns[0],axis=1,inplace=True)
df.shape
(1000, 18)

df.mean()
Apple           0.384
Bread           0.385
Butter          0.421
Cheese          0.404
Corn            0.407
Dill            0.398
Eggs            0.385
Ice cream       0.410
Kidney Beans    0.409
Milk            0.406
Nutmeg          0.401
Onion           0.403
Sugar           0.409
Yogurt          0.420
chocolate       0.421
dtype: float64

After filtering the rules with Confidence >= 50, we get 17 rules.
Filtered Table

| Antecedents   | Consequents        | Antecedents Support | Consequents Support | Support | Confidence | Lift |
|---------------|--------------------|---------------------|---------------------|---------|------------|------|
| Dill          | Milk, Chocolate   | 39.8                | 21.1                | 11.4    | 60         | 2.84 |
| Milk          | Dill, Chocolate   | 40.6                | 19.9                | 11.4    | 60         | 3.02 |
| Dill          | Chocolate, Milk   | 39.8                | 21.1                | 11.4    | 57.28      | 2.71 |
| Chocolate     | Dill, Milk        | 42.1                | 19                  | 11.4    | 57.28      | 3.01 |
| Ice-cream     | Kidney beans, Butter | 41                | 20.3                | 11      | 56.12      | 2.76 |
| Kidney beans  | Ice-cream, Butter | 40.9                | 20.7                | 11      | 56.12      | 2.71 |
| Butter        | Kidney beans, Ice-cream | 42.1          | 19.6                | 11      | 54.18      | 2.76 |
| Kidney beans  | Butter, Ice-cream | 40.9                | 20.7                | 11      | 54.18      | 2.62 |
| Chocolate     | Milk, Dill        | 42.1                | 19                  | 11.4    | 54.02      | 2.84 |
| Butter        | Ice-cream, Kidney beans | 42.1          | 19.6                | 11      | 53.14      | 2.71 |
| Ice-cream     | Butter, Kidney beans | 41                | 20.3                | 11      | 53.14      | 2.62 |
| Milk          | Chocolate         | 40.6                | 42.1                | 21.1    | 51.97      | 1.23 |
| Chocolate     | Milk              | 42.1                | 40.6                | 21.1    | 51.58      | 1.27 |
| Milk          | Chocolate, Dill   | 40.6                | 19.9                | 11.4    | 51.58      | 2.59 |
| Ice-cream     | Butter            | 41                  | 42.1                | 20.7    | 50.48      | 1.20 |
| Bread         | Yogurt            | 38.5                | 42                  | 19.3    | 50.12      | 1.19 |
| Dill          | Chocolate         | 39.8                | 42.1                | 19.9    | 50         | 1.19 |

<img width="523" alt="image" src="https://github.com/pallabibarori/CapstoneProject.git.io/assets/95372576/b03da207-fe21-4f67-bb04-c5e82bb23521">

Link to Tableau: https://public.tableau.com/views/MarketBasket_ConfidenceVsSupport/Dashboard1?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link 

## We can infer the following data:
 Products sold in Central: 1542
  - Maximum product sold in Central: Butter
  - Minimum product sold in Central: Milk

- Products sold in East: 1904
  - Maximum product sold in East: Yogurt
  - Minimum product sold in East: Apple, Sugar

- Products sold in North: 7
  - Maximum product sold in North: Bread, Chocolate, Corn, Dill, Ice cream, Sugar, Yogurt
  - Minimum product sold in North: Apple, Butter, Cheese, Eggs, Kidney beans, Milk, Nutmeg, Onion

- Products sold in West: 1762
  - Maximum product sold in West: Kidney Beans
  - Minimum product sold in West: Bread

- Products sold in South: 848
  - Maximum product sold in South: Milk, Sugar
  - Minimum product sold in South: Eggs

This data helps us to identify that most and least commonly bought items vary from region to region and we cannot rely on any specific region to plan the inventory.

As North has the least sales, we can use this data to further find the root cause of it. We can analyze customer centric data and geo-demographic data of that region to understand it better. Incase of an online store, we can study the patterns of views on the pages using SEO and other Visualization tools. Depending on the data, the retailer can decide on how to improve their sales in North region. Currently we don’t have that data with us and hence cannot proceed with further deep dives.

- Maximum number of products are sold in Eastern Store and the minimum number of products are sold in Northern Store.
- Butter, Chocolate, Yogurt are the top 3 Best-selling products.
- Top Selling product: Butter, Cheese
- Least Selling product: Apple
- Unique dates: 440
- Highest Selling year: 2017 with 132 unique Sale days.
- Lowest Selling year: 2016 with 85 unique sale days.

The above-mentioned data also helps us to identify the best performing region/store as the eastern store. 

<img width="425" alt="image" src="https://github.com/pallabibarori/CapstoneProject.git.io/assets/95372576/f5232c41-161d-4772-8813-9d5058545e40">

Link to tableau :  https://public.tableau.com/views/Year-wiseSalesData/Dashboard3?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link

-	2015 has 97 Unique Sale days, with Nutmeg being sold the least at 84 and Yogurt and Chocolate being sold the most at 97.
-	2016 has 85 Unique Sale days, with Dill being sold the least at 65 and Sugar and Corn being sold the most at 85.
-	2017 has 132 Unique Sale days, with Eggs being sold the least at 101 and Dill being sold the most at 132.
-	2018 has 126 Unique Sale says, with Eggs being sold the least at 106 and Nutmeg being sold the most at 126.
-	Top 3 products sold over the years:
 - 2015: Yogurt(87), Chocolate(97), Ice-cream(96), Eggs(96)
 - 2016: Sugar(85), Corn(85), Butter(84)
 - 2017: Dill(132), Chocolate(131), Milk (127)
 - 2018: Nutmeg (126), Kidney beans(124), Chocolate (124)
   
This data help us to identify ‘Chocolates’ as the most commonly bought items throughout the years as out of four years, it is present in top 3 for three years. With this data we can place similar products in and around ‘Chocolates’ and within majority customer’s reach.

<img width="427" alt="image" src="https://github.com/pallabibarori/CapstoneProject.git.io/assets/95372576/b8cdccc5-6942-4121-b36d-e08f3cf6ea3a">
<img width="427" alt="image" src="https://github.com/pallabibarori/CapstoneProject.git.io/assets/95372576/6829e56b-5411-482d-8f3d-d7ce00f662ed">



-	Animal Based products consists Milk, Butter etc
-	Plant Based products consists Apple, Corn etc
-	Sweets consists Chocolate, Ice-cream etc
-	Other consists Nutmeg

<img width="523" alt="image" src="https://github.com/pallabibarori/CapstoneProject.git.io/assets/95372576/5053ce00-276b-4874-83a8-97f1ab1bcca0">


  Clusters of Confidence VS Support
Link to Tableau Public: https://public.tableau.com/views/MarketBasket_ConfidenceVsSupport/Dashboard2?:language=en-US&publish=yes&:display_count=n&:origin=viz_share_link




---

## CHAPTER 3: LEARNING EXPERIENCES ON BUSINESS/TECHNOLOGY

Working on this project has helped me to understand Data better and the power of visualization.
This project although has data partially created by me and partially curated from a Kaggle source, has also helped me to understand the importance of clean and symmetric data as I got a chance to work on raw data and transform it as per my use for this project.

This project has also helped me apply Exploratory Data Analysis on my data set to uncover hidden patterns. 
With the help of  Market Basket Analysis and Association Rules Mining, we have been able to uncover 17 such rules that can help us predict what items are usually bought together. We were also able to analyse items that was mostly bought in the stores, Year-wise, Region-wise and Category wise.  This data can help us improve sales.

Through this project I was also able to gain hands on experience on MS Excel, SQL, Python and Tableau. This project also helped me in understanding Business Documentation and statistical application.

This project helped me to do quality research on the topics present in this paper which further helped me solidify my theoretical knowledge through this practical application. This project helped me to understand data trends and create metrics.

By applying all this knowledge in this project I have been able to project 17 rules, which will help us understand what products when bought together will raise sales and hence profit. We can use this to design offers, discounts or  combo deals to attract more customers.



---

## CHAPTER 4: CONCLUSION

In conclusion, this retail analytics project helped us gain some major insights on the buying pattern. By using these transactional data, we can predict future sales and hence the retailers can untap hidden data to drive growth in their business and success.![image]


### 4.1 Findings
1. Eastern store is the most popular store with maximum sales.
2.  Butter, Chocolate, Yogurt are the top 3 Best-selling products throughout the year across the different regions.
3. We also notice the sale of Animal based products and Plant based products are the maximum. This can help us create better deals. There is not much difference in the sale of the different variety of products.
4. Northern store has barely any sales throughout the years. Decisions can be taken on how to improve the sales by further deep diving into customer centric and geo-demographic data.
5. We see rise of sales at the end and beginning of every month and mostly remains constant throughout rest of the years.
6. We see dip in sales in year 2016 and again see it rise in the successive years. 



### 4.2 Recommendations
After analysing the rules generated by applying association rules:
1.	We can either place usually bought Item-sets together or nearby and see if the sale of those products keeps on getting higher. 
Example, we can either Chocolate and Milk or Ice-cream and Butter together or in the same refrigerators for the customers to access.
2.	We can keep the usually bought Item-sets far away, so that way when a customer goes to buy either Chocolate or Milk they need to pass through other items as well, this can provoke them to buy other products as well. This is more of an experimental idea. If this method is applied, we can track the customer buying pattern and can enhance the product placement in better ways so that customers can better notice other products in the shop as well rather than just browsing through them.
3.	We can create lucrative deals like discounts on both Ice-cream and Butter, or discounted price for combinations like Ice-cream and Butter when bought together.
4.	We can also introduce varieties of Ice-creams and Chocolates more since these are usually high in demand. These items should be placed near the exit so that all customers passing through that area can take notice of it and proceed purchasing them along with rest of the products.
5.	As we see rise in sale mostly during the beginning and end of the week, if we can offer sales during that time we can attract even more customers. 


---

## CHAPTER 5: BIBLIOGRAPHY/REFERENCES

1. Kaggle Datasets: [Comprehensive Data Analysis with Pandas](https://www.kaggle.com/code/prashant111/comprehensive-data-analysis-with-pandas)
2. Kaggle Datasets: [Datasets for Apriori](https://www.kaggle.com/datasets/ahmtcnbs/datasets-for-appiori)
3. Thomas Nield, “Getting started with SQL”
4. Joseph Adler, “R in a Nutshell, 2nd Edition”

---
