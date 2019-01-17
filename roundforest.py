import pandas as pd
import numpy as np

df = pd.read_csv('C:\\Users\\Betzalel Fialkoff\\Downloads\\Roundforest DS task.csv')
sales_person_stats = df.groupby('Salesman', as_index=False)['Purchased'].mean()
campaign_channel_stats = df.groupby('Camp_Channel', as_index=False)['Purchased'].mean()

histogram_of_purchases_by_age = df.groupby('Age')['Purchased'].sum()
histogram_of_purchases_by_age.hist(bins=np.sort(df['Age'].unique()))
