# app.py
import streamlit as st
from pages.login import login_page
from pages.signup import signup_page
from pages.main import main_page
import time

def main():
    # Initialize session state
    if 'recent_stocks' not in st.session_state:
        st.session_state.recent_stocks = []
    if 'search_triggered' not in st.session_state:
        st.session_state.search_triggered = False
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'last_activity_time' not in st.session_state:
        st.session_state.last_activity_time = time.time()
    if 'page' not in st.session_state:
        st.session_state.page = "Login"
    if 'flag' not in st.session_state:
        st.session_state.flag = False

    # Check for session timeout (1 minute)
    current_time = time.time()
    if st.session_state.logged_in and (current_time - st.session_state.last_activity_time > 60):
        st.session_state.logged_in = False
        st.session_state.page = "Login"
        st.session_state.flag = True  # Trigger page reload

    page = st.session_state.page

    if page == "Login":
        login_page()
    elif page == "Sign Up":
        signup_page()
    elif page == "Main Content":
        if not st.session_state.logged_in:
            st.session_state.page = "Login"
            st.session_state.flag = True  # Trigger page reload
        main_page()

if __name__ == "__main__":
    main()