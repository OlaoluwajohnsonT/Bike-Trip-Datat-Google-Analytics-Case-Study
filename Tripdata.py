#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[5]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import sklearn
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
sklearn.__version__


# In[6]:


df1= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\Cylist Data Project\\202301 tripdata.csv")
df2= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\Cylist Data Project\\202302-divvy-tripdata.csv")
df3= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\Cylist Data Project\\202303-divvy-tripdata.csv")
df4= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\Cylist Data Project\\202304-divvy-tripdata.csv")
df5= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\Cylist Data Project\\202305-divvy-tripdata.csv")
df6= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\Cylist Data Project\\202306-divvy-tripdata.csv")
df7= pd.read_csv("C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\Cylist Data Project\\202307-divvy-tripdata.csv")


# In[7]:


df1.head()


# In[8]:


dataframes_list = [df1, df2, df3, df4, df5, df6, df7]

# Concatenate these DataFrames vertically into one combined DataFrame
df = pd.concat(dataframes_list, axis=0, ignore_index=True)
df.head()


# In[9]:


df.info()


# In[10]:


df.shape


# In[11]:


df.describe()


# # Let check for MISSING VALUE

# In[12]:


df.isnull().sum()


# In[13]:


#dropping missing values
# List of columns to replace missing values with 'null'
columns_to_replace = ['start_station_name', 'start_station_id', 'end_station_name', 'end_station_id']

# Replace missing values that are not numerical with 'null'
df[columns_to_replace] = df[columns_to_replace].fillna('null')
df.isnull().sum()


# In[14]:


#let drop na for the remaining numerical value
df.dropna()
df.shape


# # Adding New Columns For Ride Length

# In[15]:


# Convert 'started_at' and 'ended_at' columns to datetime objects
df['started_at'] = pd.to_datetime(df['started_at'])
df['ended_at'] = pd.to_datetime(df['ended_at'])

# Calculate ride length in minutes and add it as a new column 'ride_length_minutes'
df['ride_length_minutes'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60


# In[16]:


df.info()


# # Adding a New Column for DAY OF THE WEEK

# In[17]:


# Calculate day of the week (1 = Sunday, 7 = Saturday) and add it as a new column 'day_of_week'
df['day_of_week'] = df['started_at'].dt.weekday + 1 

# Format 'day_of_week' column as a number with no decimals
df['day_of_week'] = df['day_of_week'].astype(int)
df.head()


# calculating the day of the week using .dt.weekday, where Monday is 0 and Sunday is 6. To match your desired format (1 = Sunday, 7 = Saturday), I add 1 to the result.

# # DESCRIPTIVE ANALYSIS

# In[18]:


# 1. Summary Statistics
df.describe()


# In[19]:


# Counting of unique values in categorical columns
df.select_dtypes(include=['object']).nunique()


# In[24]:


# Frequency distribution of 'rideable_type' column
df['rideable_type'].value_counts()


# In[23]:


# let do a basic statistics for ride length in minutes
df['ride_length_minutes'].describe()


# In[25]:


# let get the count of rides per day of the week
day_names = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday"
}

# Count the number of rides per day of the week and sort by day number
rides_per_day_of_week = df['day_of_week'].value_counts().sort_index()

# Rename the index (day numbers) with day names
rides_per_day_of_week.index = rides_per_day_of_week.index.map(day_names)
print("Rides per Day of the Week:")
rides_per_day_of_week


# It shows that there are more riding on Friday follows sunday compare to other days of the week

# # the longest ride length

# In[26]:


#let get the maxium ride length
# Find the longest ride length in minutes
longest_ride_minutes = df['ride_length_minutes'].max()

# Convert the longest ride length to hours
longest_ride_hours = longest_ride_minutes / 60


# Print the results
print("Longest Ride Length in Minutes:", longest_ride_minutes)
print("Longest Ride Length in Hours:", longest_ride_hours)
# Find the row(s) with the longest ride length
df[df['ride_length_minutes'] == longest_ride_minutes]


# It is shows that the member that has the longest ride is a casual with docked_bike at Wabash Ave & Wacker Pl. This occur on forst day of the week

# # Exploration Analysis

# # Calculate the average ride_length for members and casual riders

# In[27]:


# First have to create a pivot table to calculate the average ride_length for members and casual riders
pivot_table_option1 = pd.pivot_table(df, values='ride_length_minutes', index='member_casual', aggfunc='mean')

# Rename the columns for clarity
pivot_table_option1.columns = ['Average Ride Length (Minutes)']

print("Option 1: Average Ride Length for Members and Casual Riders")
pivot_table_option1


# In[28]:


pivot_table_option1.plot(kind='bar', title='Average Ride Length for Members and Casual Riders')
plt.xlabel('User Type')
plt.ylabel('Average Ride Length (Minutes)')
plt.xticks(rotation=0)
plt.show()


# Casual spend more time riding than members

# # Calculate the average ride_length for users by day_of_week.

# In[29]:


# Create a pivot table to calculate the average ride_length for users by day_of_week
pivot_table_option2 = pd.pivot_table(df, values='ride_length_minutes', index='member_casual', columns='day_of_week', aggfunc='mean')

# Rename the columns to day names for clarity
day_names = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday"
}
pivot_table_option2.columns = [day_names[col] for col in pivot_table_option2.columns]
pivot_table_option2


# In[30]:


pivot_table_option2.T.plot(kind='line', title='Average Ride Length for Users by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Ride Length (Minutes)')
plt.legend(title='User Type')
plt.xticks(rotation=0)
plt.show()


# Analysing the average ride perday of the week shows that casual ride more than the members

# # Calculate the number of rides for users by day_of_week.

# In[31]:


# Create a pivot table to calculate the number of rides for users by day_of_week
pivot_table_option3 = pd.pivot_table(df, values='ride_id', index='member_casual', columns='day_of_week', aggfunc='count')

# Rename the columns to day names for clarity
pivot_table_option3.columns = [day_names[col] for col in pivot_table_option3.columns]
pivot_table_option3


# In[32]:


pivot_table_option3.T.plot(kind='bar', title='Number of Rides for Users by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Rides')
plt.legend(title='User Type')
plt.xticks(rotation=0)
plt.show()


# # Visualising the number of ride overtime

# In[33]:


# Convert 'started_at' column to datetime
df['started_at'] = pd.to_datetime(df['started_at'])

# Set 'started_at' as the DataFrame's index
df.set_index('started_at', inplace=True)

# Resample the data by day and count the number of rides per day
rides_per_day = df['ride_id'].resample('D').count()

# Create a time series plot
plt.figure(figsize=(12, 6))
plt.plot(rides_per_day.index, rides_per_day.values, marker='o', linestyle='-')
plt.title('Number of Rides per Day')
plt.xlabel('Date')
plt.ylabel('Number of Rides')
plt.grid(True)
plt.show()


# # Analysing Rideable type

# In[35]:


#let check the values counts
rideable_type_counts = df['rideable_type'].value_counts()
print(rideable_type_counts)


# In[36]:


import matplotlib.pyplot as plt

# Plot a bar chart for rideable type counts
rideable_type_counts.plot(kind='bar', title='Rideable Type Counts')
plt.xlabel('Rideable Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# There are fews of docked type bike

# In[37]:


import seaborn as sns

# Plot a box plot to visualize ride length by rideable type
plt.figure(figsize=(10, 6))
sns.boxplot(x='rideable_type', y='ride_length_minutes', data=df)
plt.title('Ride Length by Rideable Type')
plt.xlabel('Rideable Type')
plt.ylabel('Ride Length (Minutes)')
plt.xticks(rotation=45)
plt.show()


# In[39]:


# Example: Box plot to compare ride length distribution between casual and member riders
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.boxplot(x='member_casual', y='ride_length_minutes', data=df)
plt.title('Ride Length Distribution by Rider Type')
plt.xlabel('Rider Type')
plt.ylabel('Ride Length (Minutes)')
plt.show()


# # Analysing Member and Casual Trend

# In[49]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming your DataFrame is named 'df'

# Calculate the total number of rides for members and casual riders
total_rides = len(df)
total_member_rides = len(df[df['member_casual'] == 'member'])
total_casual_rides = len(df[df['member_casual'] == 'casual'])

# Create a pie chart
labels = ['Member', 'Casual']
sizes = [total_member_rides, total_casual_rides]
colors = ['#66b3ff', '#ff9999']  # Blue for Member, Red for Casual

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
plt.title('Total Number of Rides by Rider Type for Jan-July 2023')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
print(total_rides)
plt.show()


# In[44]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your DataFrame is named 'df'

# Create a pivot table to calculate the total number of rides by day_of_week and member_casual
pivot_table = df.pivot_table(index='day_of_week', columns='member_casual', values='ride_id', aggfunc='count')

# Plot a stacked bar chart
plt.figure(figsize=(12, 6))
sns.set_palette('Set3')  # Set color palette
pivot_table.plot(kind='bar', stacked=True)
plt.title('Total Number of Rides by Day of the Week (Casual vs. Member)')
plt.xlabel('Day of the Week')
plt.ylabel('Total Number of Rides')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.legend(title='Rider Type', labels=['Casual', 'Member'])
plt.show()


# In[41]:


# Specify the file path where you want to save the CSV file
file_path = "C:\\Users\\TOJMARK LTD\\DATA SCIENCE PROJECT\\Cylist Data Project\\Clean2023data.csv"
# Export the DataFrame to a CSV file
df.to_csv(file_path, index=False)  # Use index=False to exclude the index column from the CSV file

print(f"DataFrame has been exported to {file_path}")


# In[48]:


df.head()


# In[ ]:




