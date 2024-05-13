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
![image](https://github.com/pallabibarori/CapstoneProject.git.io/assets/95372576/d74ce1d9-e145-4030-872c-b5e128635db9)
![image](https://github.com/pallabibarori/CapstoneProject.git.io/assets/95372576/d74ce1d9-e145-4030-872c-b5e128635db9)



---

## CHAPTER 3: LEARNING EXPERIENCES ON BUSINESS/TECHNOLOGY

This project provided hands-on experience with data analysis, visualization tools like Tableau, and statistical techniques. It enhanced understanding of business documentation, SQL, Python, and Excel, and highlighted the importance of clean data for accurate analysis.

---

## CHAPTER 4: CONCLUSION

### 4.1 Findings
- Eastern store emerges as the most popular store with maximum sales.
- Butter, Chocolate, and Yogurt are the top-selling products across different regions.
- Sales patterns vary by region, suggesting the need for targeted strategies.
- Northern store exhibits minimal sales, indicating the potential for improvement.

### 4.2 Recommendations
- Optimize product placement based on association rules to increase sales.
- Offer discounts on frequently bought item combinations to incentivize purchases.
- Introduce variety in high-demand products like Ice-cream and Chocolates.
- Implement sales promotions during peak buying periods to attract more customers.

---

## CHAPTER 5: BIBLIOGRAPHY/REFERENCES

1. Kaggle Datasets: [Comprehensive Data Analysis with Pandas](https://www.kaggle.com/code/prashant111/comprehensive-data-analysis-with-pandas)
2. Kaggle Datasets: [Datasets for Apriori](https://www.kaggle.com/datasets/ahmtcnbs/datasets-for-appiori)
3. Thomas Nield, “Getting started with SQL”
4. Joseph Adler, “R in a Nutshell, 2nd Edition”

---
