# groundwater
To predict the AnnualAverageWaterLevel using other columns as features of machine learning
The dataset contains 15,593 entries with 11 columns related to water levels, dates, and times, measured annually. Here’s a brief overview of the columns:

# AnnualAverageWaterLevel: The average water level recorded for the year.
# AnnualMaximumDailyWaterLevel: The highest daily water level recorded.
# AnnualMaximumMomentWaterLevel: The highest momentary water level.
AnnualMinimumDailyWaterLevel: The lowest daily water level.
AnnualMinimumMomentWaterLevel: The lowest momentary water level.
AnnualOccurredDateOfMaximumDailyWaterLevel: Date of the maximum daily water level.
AnnualOccurredDateOfMinimumDailyWaterLevel: Date of the minimum daily water level.
AnnualOccurredTimeOfMaximumMomentWaterLevel: Time of the maximum momentary water level.
AnnualOccurredTimeOfMinimumMomentWaterLevel: Time of the minimum momentary water level.
WellIdentifier: Identifies the well.

# the steps:

# Step 1: Data Preprocessing
Handle missing values.
Encode any categorical features (though we only have one categorical column, "WellIdentifier").
Split data into training and test sets.

# Step 2: Model Building
Use Linear Regression as the baseline model.
Train the model on the training data.

# Step 3: Evaluation
Evaluate the model using metrics like Mean Squared Error (MSE) and R².
