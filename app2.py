import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle
import os #paths to file
import warnings# warning filter
import scipy as sp #pivot egineering


#ML model
from sklearn.metrics.pairwise import cosine_similarity


#default theme and settings
pd.options.display.max_columns

#warning hadle
warnings.filterwarnings("always")
warnings.filterwarnings("ignore")
anime_df = pd.read_csv('C:/Users/tatie/OneDrive/Máy tính/python/recommendanime/model/anime.csv')
rating_df = pd.read_csv('C:/Users/tatie/OneDrive/Máy tính/python/recommendanime/model/rating.csv')
anime_df = anime_df[~np.isnan(anime_df["rating"])]

# filling mode value for genre and type
anime_df['genre'] = anime_df['genre'].fillna(
anime_df['genre'].dropna().mode().values[0])

anime_df['type'] = anime_df['type'].fillna(
anime_df['type'].dropna().mode().values[0])

#checking if all null values are filled
anime_df.isnull().sum()
anime_df = anime_df[anime_df['type']=='TV']

#step 2
rated_anime = rating_df.merge(anime_df, left_on = 'anime_id', right_on = 'anime_id', suffixes= ['_user', ''])

#step 3
rated_anime =rated_anime[['user_id', 'name', 'rating']]

#step 4
rated_anime_7500= rated_anime[rated_anime.user_id <= 7500]
pivot = rated_anime_7500.pivot_table(index=['user_id'], columns=['name'], values='rating')
# step 1
pivot_n = pivot.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)

# step 2
pivot_n.fillna(0, inplace=True)

# step 3
pivot_n = pivot_n.T

# step 4
pivot_n = pivot_n.loc[:, (pivot_n != 0).any(axis=0)]

# step 5
piv_sparse = sp.sparse.csr_matrix(pivot_n.values)
anime_similarity = cosine_similarity(piv_sparse)

#Df of anime similarities
ani_sim_df = pd.DataFrame(anime_similarity, index = pivot_n.index, columns = pivot_n.index)
rating_df['rating'] = rating_df['rating'].apply(lambda x: np.nan if x==-1 else x)
@st.cache()
def recommend(ani_name):
    number = 1
    res = []
    for anime in ani_sim_df.sort_values(by = ani_name, ascending = False).index[1:6]:
        print(f'#{number}: {anime}, {round(ani_sim_df[anime][ani_name]*100,2)}% match')
        number +=1  
        res.append(anime)
    return res

st.title('Anime Recommender')
selected_anime = st.selectbox(
'Which anime did you like?',
(anime_df['name'].values))

if st.button('Recommend'):
    with st.spinner(text='In progress'):
        anime = recommend(selected_anime)
        for i in range(5):
            st.write(f"{i+1})"+"Title  :  "+str(anime[i]))
        st.success('Done')