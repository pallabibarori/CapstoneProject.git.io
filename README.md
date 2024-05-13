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
