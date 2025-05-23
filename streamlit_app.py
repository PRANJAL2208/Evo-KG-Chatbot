import datetime  # Added for query limit reset logic
import json
import logging
import os
import pathlib

import dotenv
import kani_utils.kani_streamlit_server as ks
import requests
import streamlit as st
from kani.engines.openai import OpenAIEngine

from agents import EvoKgAgent

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path for local assets to avoid path issues
current_dir = pathlib.Path(__file__).parent.absolute()
logo_path = str(current_dir / "logo.png")
bg_image_path = str(current_dir / "floating-graph-nodes.png")

# Check if files exist
if not pathlib.Path(logo_path).exists():
    logger.warning(f"Logo file not found at {logo_path}")
    logo_path = None

if not pathlib.Path(bg_image_path).exists():
    logger.warning(f"Background image not found at {bg_image_path}, using default URL")
    bg_image_path = "https://www.nayuki.io/res/animated-floating-graph-nodes/floating-graph-nodes.png"
else:
    logger.info(f"Using local background image: {bg_image_path}")


# --- API Interaction Functions ---
# Must be set in .env file or as environment variables
# This is just a placeholder, replace with your actual API URL in production
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def get_user_details(token: str):
    """Fetches user details from the FastAPI backend."""
    url = f"{API_BASE_URL}/users/me"  # Assuming this endpoint exists
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get user details: {e}")
        error_message = "Failed to fetch user data."
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
                error_message = f"Failed to fetch user data: {error_detail}"
            except json.JSONDecodeError:
                error_message = f"Failed to fetch user data: {e.response.status_code} - {e.response.reason}"
        return False, {"error": error_message}
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching user details: {e}")
        return False, {"error": "An unexpected error occurred."}


def update_user_query_limits(token: str, query_limits: int, last_query_reset: str):
    """Updates user query limits on the FastAPI backend."""
    url = f"{API_BASE_URL}/users/me/query_limits"  # Assuming this endpoint exists
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"query_limits": query_limits, "last_query_reset": last_query_reset}
    try:
        response = requests.put(url, headers=headers, json=payload)
        response.raise_for_status()
        return True, response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to update query limits: {e}")
        error_message = "Failed to update query limits."
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
                error_message = f"Failed to update query limits: {error_detail}"
            except json.JSONDecodeError:
                error_message = f"Failed to update query limits: {e.response.status_code} - {e.response.reason}"
        return False, {"error": error_message}
    except Exception as e:
        logger.error(f"An unexpected error occurred while updating query limits: {e}")
        return False, {"error": "An unexpected error occurred."}


def register_user(
    username, email, password, first_name, last_name, organization, openai_api_key
):  # Added openai_api_key
    """Sends signup request to the FastAPI backend."""
    url = f"{API_BASE_URL}/auth/signup"
    payload = {
        "username": username,
        "email": email,
        "password": password,
        "first_name": first_name,
        "last_name": last_name,
        "organization": organization,
        "OPENAI_API_KEY": openai_api_key,
    }
    response = None  # Initialize response to None
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return True, response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Registration request failed: {e}")
        error_message = f"Registration failed: {e}"
        # Check if a response object exists in the exception
        if e.response is not None:
            try:
                # Try to get more specific error from response body if available
                error_detail = e.response.json().get("detail", str(e))
                error_message = f"Registration failed: {error_detail}"
            except (AttributeError, json.JSONDecodeError):
                # Fallback if response exists but cannot be parsed
                error_message = f"Registration failed: {e.response.status_code} - {e.response.reason}"
        # If e.response is None, the initial error_message based on 'e' is used
        return False, {"error": error_message}
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during registration: {e}")
        return False, {"error": f"An unexpected error occurred: {e}"}


def login_user(username, password):
    """Sends login request to the FastAPI backend."""
    url = f"{API_BASE_URL}/auth/login"
    # FastAPI expects form data for OAuth2PasswordRequestForm
    payload = {"username": username, "password": password}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = None  # Initialize response to None
    try:
        # Send data as x-www-form-urlencoded
        response = requests.post(url, data=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        # Expecting {'access_token': '...', 'token_type': 'bearer'}
        if "access_token" in data:
            # Fetch user details after successful login
            fetch_success, user_details = get_user_details(data["access_token"])
            if fetch_success:
                # Store user profile fields
                st.session_state["query_limits"] = user_details.get("query_limits", 10)
                st.session_state["last_query_reset"] = user_details.get(
                    "last_query_reset", datetime.datetime.utcnow().isoformat()
                )
                st.session_state["first_name"] = user_details.get("first_name", "")
                st.session_state["last_name"] = user_details.get("last_name", "")
                st.session_state["organization"] = user_details.get("organization", "")
                st.session_state["openai_api_key"] = user_details.get("OPENAI_API_KEY")
            else:
                # Handle error or set defaults if fetch fails
                st.session_state["query_limits"] = 10
                st.session_state["last_query_reset"] = (
                    datetime.datetime.utcnow().isoformat()
                )
                # Defaults for profile fields
                st.session_state["first_name"] = ""
                st.session_state["last_name"] = ""
                st.session_state["organization"] = ""
                st.session_state["openai_api_key"] = (
                    None  # Ensure API key is None on fetch failure
                )
                st.warning(
                    user_details.get(
                        "error", "Could not fetch user data, using defaults."
                    )
                )
            return True, data
        else:
            return False, {"error": "Login successful, but no token received."}
    except requests.exceptions.RequestException as e:
        logger.error(f"Login request failed: {e}")
        error_message = f"Login failed: {e}"
        # Check if a response object exists in the exception
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", str(e))
                # Handle specific FastAPI validation error format
                if (
                    isinstance(error_detail, list)
                    and len(error_detail) > 0
                    and "msg" in error_detail[0]
                ):
                    error_message = f"Login failed: {error_detail[0]['msg']}"
                elif isinstance(error_detail, str):
                    error_message = f"Login failed: {error_detail}"
                else:
                    error_message = (
                        f"Login failed: {e.response.status_code} - {e.response.reason}"
                    )

            except (json.JSONDecodeError, IndexError, KeyError):
                # Fallback if response exists but cannot be parsed or details are unexpected
                error_message = (
                    f"Login failed: {e.response.status_code} - {e.response.reason}"
                )
        # If e.response is None, the initial error_message based on 'e' is used
        return False, {"error": error_message}
    except Exception as e:  # Catch any other unexpected errors
        logger.error(f"An unexpected error occurred during login: {e}")
        return False, {"error": f"An unexpected error occurred: {e}"}


# Define custom page rendering functions
def render_about_page():
    logger.info("Rendering About Us page")

    st.title("Contact Us")
    st.markdown(
        """
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h2>Get in Touch</h2>
            <p>For questions, feedback, or support regarding the EvoKG Chatbot:</p>
            <ul>
                <li><strong>Email:</strong> <a href="mailto:gaurav.ahuja@iiitd.ac.in" style="color: #ADD8E6;">gaurav.ahuja@iiitd.ac.in</a></li>
                <li><strong>GitHub Repository:</strong> <a href="https://github.com/zakmii/Evo-KG-Chatbot/tree/kani_backend" target="_blank" style="color: #ADD8E6;">Evo-KG-Chatbot</a></li>
            </ul>
            <hr style="border-top: 1px solid rgba(255, 255, 255, 0.2);">
            <h3>Developers</h3>
            <ul>
                <li>Ankit Singh: <a href="https://github.com/zakmii" target="_blank" style="color: #ADD8E6;">GitHub Profile</a></li>
                <li>Arushi Sharma: <a href="https://github.com/AruShar" target="_blank" style="color: #ADD8E6;">GitHub Profile</a></li>
            </ul>
            <hr style="border-top: 1px solid rgba(255, 255, 255, 0.2);">
            <h3>Report Issues</h3>
            <p>If you encounter any problems or have suggestions for improvement, please <a href="https://github.com/zakmii/Evo-KG-Chatbot/issues" target="_blank" style="color: #ADD8E6;">report them on our GitHub repository</a>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evokg_intro():
    logger.info("Rendering EvoKG Introduction page")

    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 2.5em;">Welcome to the EvoKG Chatbot</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    intro_container = st.container(border=True)
    with intro_container:
        st.markdown(
            """
            <div class="hover-section" style="padding: 20px; border-radius: 10px;">
              <h2 style="color: #90EE90;">What is EvoKG?</h2>
              <p style="font-size: 1.1em;">
                EvoKG is a groundbreaking Evolutionary Knowledge Graph that integrates
                biological insights across six key species, organized by evolutionary progression:
              </p>
              <ul style="font-size: 1.1em; list-style-type: '🧬 '; padding-left: 20px;">
                <li><strong>Y:</strong> <em>Saccharomyces cerevisiae</em> (Yeast)</li>
                <li><strong>C:</strong> <em>Caenorhabditis elegans</em> (Nematode)</li>
                <li><strong>D:</strong> <em>Drosophila melanogaster</em> (Fruit Fly)</li>
                <li><strong>Z:</strong> <em>Danio rerio</em> (Zebrafish)</li>
                <li><strong>M:</strong> <em>Mus musculus</em> (Mouse)</li>
                <li><strong>H:</strong> <em>Homo sapiens</em> (Human)</li>
              </ul>
              <p style="font-size: 1.1em;">This chatbot allows you to query EvoKG, explore relationships between biological entities, and even predict potential connections.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        st.markdown(
            """
            <style>
            .stButton>button {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 12px;
                transition: background-color 0.3s ease;
                width: 100%;
            }
            .stButton>button:hover {
                background-color: #45a049;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start Chatting →", key="start_chat_button"):
            st.session_state.current_page = "chat"
            st.rerun()


custom_pages = {
    "intro": ("Introduction", render_evokg_intro, "🏠"),
    "about": ("About Us", render_about_page, "📧"),
    "chat": ("Chatbot", None, "💬"),
}

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    st.set_page_config(
        page_title="Login - EvoLLM",
        page_icon="🧬",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user_token" not in st.session_state:
    st.session_state["user_token"] = None
if "username" not in st.session_state:
    st.session_state["username"] = None
if "auth_view" not in st.session_state:
    st.session_state["auth_view"] = "Login"
if "query_limits" not in st.session_state:  # Ensure query_limits is initialized
    st.session_state["query_limits"] = 10
if "last_query_reset" not in st.session_state:
    st.session_state["last_query_reset"] = datetime.datetime.utcnow().isoformat()
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = None

if not st.session_state["logged_in"]:
    col1, col2, col3 = st.columns([0.5, 2, 0.5])
    with col2:
        if logo_path:
            st.image(logo_path, width=150)
        st.title("Welcome to EvoLLM")

        auth_choice = st.radio(
            "Choose Action",
            ["Login", "Sign Up"],
            key="auth_choice",
            horizontal=True,
            index=0 if st.session_state["auth_view"] == "Login" else 1,
        )
        st.session_state["auth_view"] = auth_choice

        if st.session_state["auth_view"] == "Login":
            st.subheader("Login")
            with st.form("login_form"):
                login_username = st.text_input("Username", key="login_user")
                login_password = st.text_input(
                    "Password", type="password", key="login_pass"
                )
                login_button = st.form_submit_button("Login")

                if login_button:
                    if not login_username or not login_password:
                        st.error("Please enter both username and password.")
                    else:
                        success, data = login_user(login_username, login_password)
                        if success:
                            st.session_state["logged_in"] = True
                            st.session_state["user_token"] = data.get("access_token")
                            st.session_state["username"] = login_username
                            st.session_state["auth_view"] = "Login"
                            logger.info(
                                f"User '{login_username}' logged in successfully."
                            )
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error(
                                data.get(
                                    "error",
                                    "Login failed. Please check your credentials.",
                                )
                            )

            # Add this section for quick test credentials
            st.markdown("-----")  # Optional: adds a horizontal line for separation
            test_username = os.getenv("TEST_ACCOUNT_USERNAME", "testuser")
            test_password = os.getenv("TEST_ACCOUNT_PASSWORD", "testpassword123")
            st.info(
                f"""
                **Quick Test Account:**

                To quickly test the chatbot, you can use the following credentials:
                - **Username:** `{test_username}`
                - **Password:** `{test_password}`
                """
            )

        elif st.session_state["auth_view"] == "Sign Up":
            st.subheader("Sign Up")
            with st.form("signup_form"):
                username = st.text_input("Username")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                first_name = st.text_input("First Name")
                last_name = st.text_input("Last Name")
                organization = st.text_input("Organization")
                openai_api_key = st.text_input("OpenAI API Key", type="password")
                st.markdown(
                    "<a href='https://platform.openai.com/account/api-keys' target='_blank'>How to get an API key?</a>",
                    unsafe_allow_html=True,
                )
                submitted = st.form_submit_button("Sign Up")
                if submitted:
                    if not all(
                        [
                            username,
                            email,
                            password,
                            first_name,
                            last_name,
                            organization,
                            openai_api_key,
                        ]
                    ):
                        st.error("Please fill in all fields.")
                    else:
                        success, data = register_user(
                            username,
                            email,
                            password,
                            first_name,
                            last_name,
                            organization,
                            openai_api_key,
                        )
                        if success:
                            st.success(
                                "Signup successful! Please login with your new credentials."
                            )
                            st.session_state["auth_view"] = "Login"
                            st.rerun()  # Rerun to switch to login view
                        else:
                            st.error(data.get("error", "Signup failed."))

    st.stop()

elif st.session_state["logged_in"]:
    # Store the function in session_state so kani_streamlit_server.py can access it
    st.session_state.update_user_query_limits_func = update_user_query_limits

    ks.initialize_app_config(
        show_function_calls=False,
        page_title="EvoLLM",
        app_title="EvoLLM",
        logo_path=logo_path,
        background_image=bg_image_path,
        page_icon="🧬",
        initial_sidebar_state="expanded",
        custom_pages=custom_pages,
        menu_items={
            "Get Help": "https://github.com/zakmii/Evo-KG-Chatbot",
            "Report a Bug": "https://github.com/zakmii/Evo-KG-Chatbot/issues",
            "About": "EvoLLM is built on GPT-4o-mini, Streamlit, zhudotexe/kani, hourfu/redlines, and oneilsh/kani-utils.",
        },
    )

    with st.sidebar:
        st.write(f"Welcome, {st.session_state['username']}!")
        if st.button("Logout"):
            logger.info(f"User '{st.session_state['username']}' logging out.")
            st.session_state["logged_in"] = False
            st.session_state["user_token"] = None
            st.session_state["username"] = None
            st.session_state["openai_api_key"] = None
            st.session_state["current_page"] = "intro"
            st.rerun()

    user_api_key = st.session_state.get("openai_api_key")

    if user_api_key:
        engine = OpenAIEngine(user_api_key, model="gpt-4o-mini")

        def get_agents():
            return {
                "EvoLLM (4o-mini)": EvoKgAgent(engine),
            }

        ks.set_app_agents(get_agents)
        ks.serve_app()
    else:
        st.error(
            "OpenAI API key not found for your account. "
            "Chatbot functionality cannot be initialized. "
            "Please ensure your API key was correctly submitted during registration and is available in your profile."
        )
        # If custom pages should still be accessible, ks.serve_app() might need to be called
        # outside this if/else, and get_agents() would need to handle the no-key case gracefully.
        # For now, the main chat app part won't load if the key is missing.
