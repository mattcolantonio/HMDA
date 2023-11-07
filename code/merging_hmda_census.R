library(readr)
library(tidycensus)
library(dplyr)
library(tidyr)
library(reshape2)

hmda_data <- read.csv("hmda_MA.csv")

# creating a list of unique census tracts
unique_census_tracts <- unique(hmda_data$census_tract)
# Convert the unique census blocks to a list
census_tract_list <- as.list(unique_census_tracts)
# you will need your own Census API key to replicate this code
census_api_key("YOUR API KEY") # enter your personal API key obtained from the census site

# Set the desired year
year <- 2021

# Specify the state and county (Allegheny, Pennsylvania)
state <- "MA"

# Set the geography to "tract"
geography <- "tract"

# Create a list of variables for the API call
variables <- c(
  "DP03_0005PE",  # Unemployment rate
  "DP03_0062E",  # Median HH income
  "DP04_0003PE",  # Vacant housing units (%)
  "DP04_0026PE",  # Total housing units built before 1939 (%)
  "DP04_0046PE",  # Tenure: owner-occupied (%)
  "DP04_0089E",  # Median value owner-occupied
  "DP04_0134E",  # Median gross rent
  "DP05_0001E",  # Total population
  "DP05_0037PE",  # Total population white (%)
  "DP05_0038PE"   # Total population black (%)
)

# Use your list of unique census tracts for the API call
# unique_census_tracts <- unique(hmda_data$census_tract)
# filtered_census_tracts <- unique_census_tracts

# Create an empty data frame to store the results
census_data <- data.frame()

# Loop through the filtered census tracts and fetch data
for (tract in unique_census_tracts) {
  data <- get_acs(geography = geography, variables = variables, 
                  year = year, state = state, tract = tract)
  
  census_data <- bind_rows(census_data, data)
}

# Further formatting...
census_data <- census_data[, !names(census_data) %in% c("NAME", "moe")]

wide_data <-  reshape(census_data, idvar = "GEOID", timevar = "variable", direction = "wide")
census_data_MA <- wide_data

colnames(census_data_MA)<- c(
  "census_tract",
  "Unemployment_Rate",
  "Median_HH_Income",
  "Vacant_Housing_Units_Percentage",
  "Total_Housing_Units_Built_Before_1939_Percentage",
  "Tenure_Owner_Occupied_Percentage",
  "Median_Value_Owner_Occupied",
  "Median_Gross_Rent",
  "Total_Population",
  "Total_Population_White_Percentage",
  "Total_Population_Black_Percentage"
)

# save the census data separately
file_path <- "HMDA/rawdata/census_data_MA.csv"
write.csv(census_data_MA, file = file_path, row.names = FALSE)


# Merge the two data frames by "census_tract"
hmda_and_census <- merge(hmda_data, census_data_MA,  by = "census_tract", all = TRUE)

# evaluate null values, can be done later as well
hmda_and_census %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  glimpse()

# parcel_and_census <- na.omit(parcel_and_census) # if you want to remove missing values 

# now save this dataframe (saveddata folder)
file_path <- "HMDA/saveddata/hmda_and_census.csv"
write.csv(hmda_and_census, file = file_path, row.names = FALSE)





