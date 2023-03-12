import streamlit as st

st.title("Pakistani News...")


query = st.text_input("Enter Your Query", "Type Here ...")
 
# display the name when the submit button is clicked
# .title() is used to get the input text string
if(st.button('Submit')):
    result = name.title()
    st.success(result)
