import kani_utils.kani_streamlit_server as ks
import os
import dotenv
from kani.engines.openai import OpenAIEngine
from agents import EvoKgAgent
import pathlib
import logging
import streamlit as st
import streamlit_authenticator as stauth  # Import the library
import yaml  # To load config if using YAML
from yaml.loader import SafeLoader  # To load config if using YAML

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
    # Use remote URL as fallback
    bg_image_path = "https://www.nayuki.io/res/animated-floating-graph-nodes/floating-graph-nodes.png"
else:
    logger.info(f"Using local background image: {bg_image_path}")


# Function to apply premium login styling
def _apply_login_styling(bg_image=bg_image_path):
    login_style = f"""
    <style>
        /* --- General Page Styling --- */
        body {{
            background-color: #1a1a2e; /* Dark background */
        }}

        /* --- Main Container Styling --- */
        [data-testid="stAppViewContainer"] > .main {{
            background-image: linear-gradient(rgba(26, 26, 46, 0.9), rgba(26, 26, 46, 0.95)), url("{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh; /* Full viewport height */
        }}

        /* --- Login Form Container --- */
        div[data-testid="stVerticalBlock"] {{
            /* Find the specific block containing the login form - this might need adjustment based on Streamlit updates */
            /* A common pattern is multiple nested vertical blocks */
        }}

        /* Target the inner container likely holding the form elements */
        div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] > div[data-testid="stVerticalBlock"] {{
            background-color: rgba(40, 40, 60, 0.85); /* Slightly lighter card background */
            padding: 2.5rem 3rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
            max-width: 450px; /* Limit form width */
            margin: auto; /* Center horizontally */
        }}

        /* --- Form Elements Styling --- */
        h1 {{
            color: #e0e0e0;
            text-align: center;
            margin-bottom: 1.5rem;
            font-weight: 300;
            letter-spacing: 1px;
        }}

        /* Input field labels */
        label[data-testid="stWidgetLabel"] p {{
            color: #b0c4de !important; /* Light blue/grey */
            font-size: 0.95em !important;
            margin-bottom: 0.3rem !important;
        }}

        /* Input fields */
        div[data-testid="stTextInput"] input,
        div[data-testid="stPasswordInput"] input {{
            background-color: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            border-radius: 8px !important;
            color: #e0e0e0 !important;
            padding: 0.75rem 1rem !important;
            transition: border-color 0.3s ease, box-shadow 0.3s ease;
        }}
        div[data-testid="stTextInput"] input:focus,
        div[data-testid="stPasswordInput"] input:focus {{
            border-color: #87CEEB !important; /* Sky blue on focus */
            box-shadow: 0 0 8px rgba(135, 206, 235, 0.3) !important;
        }}

        /* Login Button */
        div[data-testid="stButton"] button {{
            background-color: #5a67d8; /* Indigo */
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-size: 1em;
            font-weight: 600;
            width: 100%; /* Full width */
            margin-top: 1.5rem;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.1s ease;
        }}
        div[data-testid="stButton"] button:hover {{
            background-color: #4c51bf; /* Darker Indigo */
            transform: translateY(-2px);
        }}
        div[data-testid="stButton"] button:active {{
            transform: translateY(0);
        }}

        /* Error/Warning Messages */
        div[data-testid="stAlert"] {{
            background-color: rgba(220, 53, 69, 0.1); /* Reddish background */
            border: 1px solid rgba(220, 53, 69, 0.5);
            border-radius: 8px;
            padding: 0.8rem 1rem;
            margin-top: 1rem;
        }}
        div[data-testid="stAlert"] div[role="alert"] {{
             color: #f8d7da; /* Light red text */
        }}

        /* Hide Streamlit Header/Footer */
        [data-testid="stHeader"], [data-testid="stFooter"] {{
            display: none;
        }}

        /* Hide Sidebar on Login */
        [data-testid="stSidebar"] {{
            display: none;
        }}

    </style>
    """
    st.markdown(login_style, unsafe_allow_html=True)


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

    # Centered button with enhanced styling
    st.markdown("---")  # Visual separator
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col2:
        st.markdown(
            """
            <style>
            .stButton>button {
                background-color: #4CAF50; /* Green */
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
                width: 100%; /* Make button fill column */
            }
            .stButton>button:hover {
                background-color: #45a049; /* Darker Green */
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Start Chatting →", key="start_chat_button"):
            st.session_state.current_page = "chat"
            st.rerun()


# Define custom pages dict with tuples of (page_name, render_function, icon)
custom_pages = {
    "intro": ("Introduction", render_evokg_intro, "🏠"),
    "about": ("About Us", render_about_page, "📧"),
    # Chat page is handled separately by the framework
    "chat": ("Chatbot", None, "💬"),
}

# --- Initialize App Config FIRST ---
# This must be the first Streamlit command
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
# --- End Initialize App Config ---


# --- Authentication Setup (streamlit-authenticator) ---
try:
    # Use a path relative to the script file for robustness
    config_path = pathlib.Path(__file__).parent / "config.yaml"
    with open(config_path) as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    logger.error("Authentication configuration file (config.yaml) not found.")
    st.error(
        "Authentication configuration file (config.yaml) not found. Please create it."
    )
    st.stop()
except yaml.YAMLError as e:
    logger.error(f"Error parsing authentication configuration file: {e}")
    st.error(f"Error parsing authentication configuration file: {e}")
    st.stop()
except Exception as e:
    logger.error(f"Error loading authentication configuration: {e}")
    st.error(f"Error loading authentication configuration: {e}")
    st.stop()
# ---

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
    config["preauthorized"],
)

# --- End Authentication Setup ---


# --- Main App Logic ---

# Render the login widget
# This should be one of the first things called in your script
name, authentication_status, username = (
    authenticator.login()
)  # Renders login form and returns status

if not authentication_status:
    _apply_login_styling()  # Apply custom styling for login page
    st.error("Username/password is incorrect")
    st.stop()  # Stop execution if login fails
elif authentication_status is None:
    _apply_login_styling()  # Apply custom styling for login page
    st.warning("Please enter your username and password")
    st.stop()  # Stop execution if waiting for input
elif authentication_status:  # Login successful
    logger.info(f"User '{username}' logged in successfully.")
    # Proceed with initializing and running the main application

    # define an engine to use (see Kani documentation for more info)
    engine = OpenAIEngine(os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
    # mistralEngine = HuggingEngine(
    #     model_id="mistralai/Mistral-7B-Instruct-v0.3", token=os.environ["MISTRAL_TOKEN"]
    # )

    # We also have to define a function that returns a dictionary of agents to serve
    def get_agents():
        return {
            "EvoLLM (4o-mini)": EvoKgAgent(engine),
            # "EvoLLM (Mistral)": EvoKgAgent(mistralEngine),
        }

    # tell the app to use that function to create agents when needed
    ks.set_app_agents(get_agents)

    ########################
    ##### 3 - Serve App
    ########################

    # The serve_app function itself needs authentication checks internally
    ks.serve_app(authenticator)  # Pass authenticator to serve_app

# --- End Main App Logic ---
