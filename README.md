# HMDA
This project will analyze mortgage application data in Massachusetts to understand ideal applicants and potential racial biases in loan denial.

Team Members: Matt, Yesh, Diego, Anita
Important Files: 
hmda_and_census.zip
hmda_fns.py
hmda_dashboard_general.py
hmda_dashboard_income.py
      
Goal: Use data from mortgage applications in Massachusetts to determine the qualities of an ideal applicant and explore potential biases of lenders regarding race, sex, and location. 
Results: Implemented logistic regression and random forest models to create accurate models for predicting loan acceptance in Massachusetts. Found that, when controlling for income, black applicants may be less likely to be accepted than white applicants. Overall, debt and credit factors were the most important features. 

hmda_fns.py document contains functions for cleaning data and modeling/visualizing. The dashboard scripts create interactive dashboards using streamlit and using functions defined in hmda_fns.py.

All previous scripts that went into the creation of those listed above can be found @ https://github.com/mattcolantonio/HMDA
Original dataset from Consumer Financial Protection Bureau: https://ffiec.cfpb.gov/data-browser/data/2022?category=states&items=MA
