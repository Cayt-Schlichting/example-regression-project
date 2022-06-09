# Estimating Tax Value Using Zillow Data 
*Audience: Zillow data science team *

## Project Summary

**Project Goal:** The goal of this project is to build an improved model for predicting property tax assessed values for Single Family Properties.  Accurate property value predictions are crucial to our business operations and maintaining our customer base.
- Optimize regression models to predict tax values of of Single Family Properties with the goal of outperforming our current models.
- Identify key drivers of property value.
- Make recommendations on what features to use in our models and next steps.

**Deliverables:**
- wrangle.py module with preparation and acquisition functions
  - wrangle_notebook.ipynb contains steps and notes on data acquisition and preparation decisions
- explore.ipynb
  - Contains exploratory analysis of the data including visualizations and hypothesis testing
- modeling.ipynb
  - Contains full modeling work, with notes on parameter choices and model evaluation
- Final_Report.ipynb
  - Contains curtailed version of project in presentable format.  

- Ask exploratory questions of the data that will give an understanding about the attributes and drivers of tax value of homes    
    - Answer questions through charts and statistical tests
- Construct the best possible model for predicting tax value of homes
    - Make predictions for a subset of out-of-sample data
- Adequately document and annotate all code
- Give a 5 minute presentation to the Zillow Data Science Team
- Field questions about my specific code, approach to the project, findings and model

***
### Data Dictionary

|Feature|Definition|
|:-------|:----------|
| value | The total tax assessed value of the parcel |
| bath        |     Number of bathrooms (includes partial bathrooms)|
| bed |     Number of bedrooms  |
| sf | Calculated total finished living area of the home  |
| zipcode  |   The zillow zip code where the property is located |
| yearbuilt  |     The Year the principal residence was built  | 
| county    |     The county in which the property is located |
| sf_per_bed | Square footage per number of bedrooms.  0 bedrooms were counted as having 1 bedroom|

---
### Questions/thoughts I have of the Data
- What features are most strongly correlated to tax value of homes?
    - Are any of these correlated to one another? Are there confounding variables? Not truly independent?
- I think lotsizesquarefeet and calculatedfinishedsquarefeet will have the strongest relationship with the target.
- I'm unsure how strongly bathroomcnt will correlate, but I suspect bedroomcnt will be relatively strong.

### Hypotheses:

I expect tax value to be primarily driven by:
- square feet: more sf will increase price values
- bed/bath number: more beds and baths will increase tax values
- sf_per_bed: Higher sf_per_bd will increase tax values
- zip: Tax values will vary high or lower than average by zip code. (there will be clustering)
- county: Orange and LA county will have higher tax values than Ventura. LA is expected to have wider range of values than Orange

### Questions:
- Does sf_per_bed correlate with tax value?
- Is there price clustering by zip code? by county?
- What correlates stronges with tax value?

---
## Project Plan 

#### Plan
- Acquire and Prepare data from the Codeup SQL Database. Create an wrangle.py module for process automation.
- Explore the data and begin to address questions and hypotheses
- Create regression models.
    - Evaluate models on train and validate datasets and adjust hyperparameters to optimize models.
- Choose the best model and run against test subset.
- Generate a Report notebook that shows a subset of this work to present to data science management.
- Create README.md with data dictionary, project goals, initial hypothesis and project plan.

## Reproduce this project
- In order to run through this project yourself you will need your own env.py file with zillow database access on the codeup server.