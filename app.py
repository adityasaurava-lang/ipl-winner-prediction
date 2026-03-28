import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('ipl_winning.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))
st.header("Author:- ADITYA RAJ")

st.title("IPL Match Winner Predictor")

batting_team = st.selectbox('Batting Team', ['Mumbai Indians','Chennai Super Kings','Royal Challengers Bangalore','Kolkata Knight Riders','Delhi Capitals','Sunrisers Hyderabad'])

bowling_team = st.selectbox('Bowling Team', ['Mumbai Indians','Chennai Super Kings','Royal Challengers Bangalore','Kolkata Knight Riders','Delhi Capitals','Sunrisers Hyderabad'])

venue = st.text_input("Venue")

runs_left = st.number_input("Runs Left")
balls_left = st.number_input("Balls Left")
wickets_left = st.number_input("Wickets Left")

if st.button("Predict"):
    rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team':[batting_team],
        'bowling_team':[bowling_team],
        'venue':[venue],
        'runs_left':[runs_left],
        'balls_left':[balls_left],
        'wickets_left':[wickets_left],
        'rrr':[rrr]
    })

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=columns, fill_value=0)

    result = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    if result == 1:
        st.success(f"{batting_team} will win 🏏")
    else:
        st.success(f"{bowling_team} will win 🏏")

    st.write("Win Probability:")
    st.write(f"{batting_team}: {proba[1]*100:.2f}%")
    st.write(f"{bowling_team}: {proba[0]*100:.2f}%")
    
