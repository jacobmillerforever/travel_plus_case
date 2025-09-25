library(readr)
library(here)

df <- read_csv(here("data", "dataTravelPlus.csv"))
# Column names and data types
str(df)

# Evaluate missingness
colSums(is.na(df))

# Peek at a few rows
head(df)

install.packages("skimr")
library(skimr)
skim(df)

