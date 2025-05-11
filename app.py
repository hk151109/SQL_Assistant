from dotenv import load_dotenv
import streamlit as st
import os
import sqlite3
import google.generativeai as genai
import pandas as pd
import io
import uuid
import glob
import json
import shutil
import time
import atexit
import hashlib
import datetime
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# Configure the API Key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    st.error("No API key found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Initialize session state for user management
if 'session_id' not in st.session_state:
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    st.session_state.session_id = session_id
    st.session_state.session_start_time = time.time()

# Create user-specific directories
user_dir = f"user_data/{st.session_state.session_id}"
user_db_dir = f"{user_dir}/databases"
user_metadata_dir = f"{user_dir}/metadata"

# Create directories if they don't exist
os.makedirs(user_db_dir, exist_ok=True)
os.makedirs(user_metadata_dir, exist_ok=True)

# Initialize session state for database management
if 'current_db' not in st.session_state:
    st.session_state.current_db = None
if 'current_db_name' not in st.session_state:
    st.session_state.current_db_name = None
if 'prompt' not in st.session_state:
    st.session_state.prompt = None
if 'databases_created' not in st.session_state:
    st.session_state.databases_created = []

# Function to clean up user data when session ends
def cleanup_user_data(session_id):
    user_dir = f"user_data/{session_id}"
    if os.path.exists(user_dir):
        try:
            # First, close all database connections for this user
            for db_path in glob.glob(f"{user_dir}/databases/*.db"):
                close_db_connections(db_path)
            
            # Then remove the directory
            shutil.rmtree(user_dir)
            print(f"Cleaned up user data for session {session_id}")
        except Exception as e:
            print(f"Error cleaning up user data: {str(e)}")

# Helper function to close all connections to a database
def close_db_connections(db_path):
    try:
        # Create a temporary connection and execute PRAGMA to close any open connections
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA optimize;")
        conn.execute("PRAGMA wal_checkpoint(FULL);")
        conn.close()
    except Exception as e:
        print(f"Error closing database connections: {str(e)}")

# Register cleanup function to run when the script exits
atexit.register(lambda: cleanup_user_data(st.session_state.session_id))

# Function to generate a prompt based on database schema
def generate_prompt(db_path, db_name):
    table_info = get_table_info(db_path)
    
    prompt_template = f"""
    You are an expert in converting natural language questions to precise SQL queries!
    
    Your task is to generate ONLY the SQL query that answers the user's question - no explanations or additional text.
    
    Database information:
    - Database name: {db_name}
    - Tables and their columns:
    """
    
    for table_name, info in table_info.items():
        columns_str = ", ".join(info['columns'])
        prompt_template += f"\n    * {table_name}: {columns_str}"
    
    prompt_template += """
    
    Guidelines:
    1. Return ONLY the SQL query without any markdown formatting, comments, or explanations
    2. For filtering text values, always use proper SQL syntax with quotes (e.g., WHERE column = "value")
    3. Make sure your queries handle potential case sensitivity properly
    4. If the question is ambiguous, make a reasonable assumption and provide the most likely query
    5. Handle aggregation functions (COUNT, AVG, SUM, etc.) appropriately
    6. Do not include SQL keywords like "SQL" or markdown delimiters like ``` in your response
    7. If multiple tables are involved, use proper JOIN operations
    
    Example conversions:
    - "How many records are there in table X?" → SELECT COUNT(*) FROM X;
    - "Show all records where column Y equals Z" → SELECT * FROM X WHERE Y = "Z";
    - "Count records grouped by column A" → SELECT A, COUNT(*) FROM X GROUP BY A;
    """
    
    return [prompt_template]

# Function to Load the Google Gemini Model and Provide SQL query as response
def get_gemini_response(question, prompt):
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            contents=[{
                "parts": [
                    {"text": prompt[0]},
                    {"text": question}
                ]
            }]
        )
        
        # Clean the response text to remove any markdown code formatting or backticks
        sql_query = response.text.strip()
        # Remove markdown code block formatting if present
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        # Remove any leading/trailing quotes
        sql_query = sql_query.strip('"\'')
        
        return sql_query
    except Exception as e:
        return f"Error generating SQL query: {str(e)}"

# Function to retrieve data from the SQLite database
def read_sql_query(sql, db):
    conn = None
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        # Get column names
        column_names = [description[0] for description in cur.description]
        return rows, column_names
    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
        return [], []
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return [], []
    finally:
        if conn:
            conn.close()

# Function to get all tables in the database
def get_table_info(db_path):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        # Get all tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        table_info = {}
        
        # For each table, get its structure and sample data
        for table in tables:
            table_name = table[0]
            # Get columns
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns_info = cursor.fetchall()
            columns = [col[1] for col in columns_info]
            
            # Get sample data (first 5 rows)
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            sample_data = cursor.fetchall()
            
            table_info[table_name] = {
                'columns': columns,
                'sample_data': sample_data
            }
        
        return table_info
    except Exception as e:
        st.error(f"Error getting table information: {str(e)}")
        return {}
    finally:
        if conn:
            conn.close()

# Function to import CSV or XLSX to SQLite database
def import_file_to_sqlite(file, table_name, db_path, file_type):
    conn = None
    try:
        # Read file based on type
        if file_type == 'csv':
            df = pd.read_csv(file)
        elif file_type == 'xlsx':
            df = pd.read_excel(file)
        else:
            return False, 0, "Unsupported file type"
        
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        
        # Write dataframe to SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Save metadata about the database
        metadata = {
            "tables": [table_name],
            "original_filename": file.name,
            "file_type": file_type,
            "created_at": datetime.datetime.now().isoformat(),
            "session_id": st.session_state.session_id
        }
        
        metadata_path = f"{user_metadata_dir}/{os.path.basename(db_path)}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Add to list of created databases
        if db_path not in st.session_state.databases_created:
            st.session_state.databases_created.append(db_path)
        
        return True, df.shape[0], ""
    except Exception as e:
        error_msg = str(e)
        return False, 0, error_msg
    finally:
        if conn:
            conn.close()

# Function to delete a database
def delete_database(db_path):
    try:
        # First, close any open connections to the database
        close_db_connections(db_path)
        
        # Create a "vacuum" connection to defragment the database
        # This ensures we're not leaving any orphaned pages
        conn = sqlite3.connect(db_path)
        conn.execute("VACUUM;")
        conn.close()
        
        # Delete the database file with proper file locking
        if os.path.exists(db_path):
            os.remove(db_path)
            
            # Verify the database is actually deleted
            if os.path.exists(db_path):
                raise Exception(f"Failed to delete database file at {db_path}")
        
        # Delete the metadata file
        metadata_path = f"{user_metadata_dir}/{os.path.basename(db_path)}.json"
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            
            # Verify the metadata is actually deleted
            if os.path.exists(metadata_path):
                raise Exception(f"Failed to delete metadata file at {metadata_path}")
        
        # Remove from list of created databases
        if db_path in st.session_state.databases_created:
            st.session_state.databases_created.remove(db_path)
        
        return True, ""
    except Exception as e:
        return False, str(e)

# Function to list available databases for the current session
def list_available_databases():
    try:
        db_files = glob.glob(f"{user_db_dir}/*.db")
        databases = []
        
        for db_file in db_files:
            # Check if the file actually exists and has content
            if not os.path.exists(db_file) or os.path.getsize(db_file) == 0:
                continue
                
            db_name = os.path.basename(db_file)
            
            # Try to load metadata
            metadata_file = f"{user_metadata_dir}/{db_name}.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    original_filename = metadata.get('original_filename', 'Unknown')
                    created_at = metadata.get('created_at', 'Unknown')
                    file_type = metadata.get('file_type', 'Unknown')
            else:
                original_filename = 'Unknown'
                created_at = 'Unknown'
                file_type = 'Unknown'
            
            databases.append({
                'path': db_file,
                'name': db_name,
                'original_filename': original_filename,
                'created_at': created_at,
                'file_type': file_type
            })
        
        return databases
    except Exception as e:
        st.error(f"Error listing databases: {str(e)}")
        return []

# Function to run cleanup for expired sessions
def check_and_cleanup_old_sessions():
    try:
        # Current timestamp
        current_time = time.time()
        
        # Get all session directories
        session_dirs = glob.glob("user_data/*")
        
        for session_dir in session_dirs:
            session_id = os.path.basename(session_dir)
            
            # Skip current session
            if session_id == st.session_state.session_id:
                continue
                
            # Check if session data is older than 1 hour
            dir_stat = os.stat(session_dir)
            dir_time = dir_stat.st_mtime  # Last modification time
            
            # If older than 1 hour (3600 seconds), clean it up
            if current_time - dir_time > 3600:
                cleanup_user_data(session_id)
    except Exception as e:
        print(f"Error checking old sessions: {str(e)}")

# Calculate session time remaining
def get_session_time_info():
    current_time = time.time()
    elapsed_time = current_time - st.session_state.session_start_time
    
    # Session timeout after 1 hour (3600 seconds)
    session_timeout = 3600
    remaining_time = session_timeout - elapsed_time
    
    if remaining_time <= 0:
        # Session expired, force cleanup and restart
        cleanup_user_data(st.session_state.session_id)
        # Generate new session
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_start_time = current_time
        st.session_state.current_db = None
        st.session_state.current_db_name = None
        st.session_state.prompt = None
        st.session_state.databases_created = []
        remaining_time = session_timeout
        
    # Convert to minutes for display
    remaining_minutes = int(remaining_time / 60)
    remaining_seconds = int(remaining_time % 60)
    
    return remaining_minutes, remaining_seconds

# Run the cleanup check for old sessions
check_and_cleanup_old_sessions()

# Streamlit App
st.set_page_config(page_title="SQL Natural Language Query Assistant", page_icon="🔍", layout="wide")
st.title("🔍 SQL Natural Language Query Assistant")

# Session information in sidebar
with st.sidebar:
    st.subheader("Session Information")
    remaining_minutes, remaining_seconds = get_session_time_info()
    st.info(f"Session time remaining: {remaining_minutes}m {remaining_seconds}s")
    st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
    
    if st.button("New Session", type="primary", help="Start a new session and clear all databases"):
        # Clean up current session data
        cleanup_user_data(st.session_state.session_id)
        # Generate new session
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_start_time = time.time()
        st.session_state.current_db = None
        st.session_state.current_db_name = None
        st.session_state.prompt = None
        st.session_state.databases_created = []
        st.rerun()

st.markdown("""
This app translates your natural language questions into SQL queries and retrieves data from databases.
Upload a CSV or XLSX file to create a new database, or select an existing database to query.
**Note:** All databases are tied to your current session and will be deleted when your session expires.
""")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Query Interface", "Database Upload", "Database Reference"])

# Database Upload Tab
with tab2:
    st.subheader("Upload New Database")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV or XLSX file", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        # Determine file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        file_type = 'csv' if file_extension == '.csv' else 'xlsx' if file_extension == '.xlsx' else 'unknown'
        
        if file_type == 'unknown':
            st.error("Unsupported file type. Please upload a CSV or XLSX file.")
        else:
            # Preview the file
            try:
                if file_type == 'csv':
                    df_preview = pd.read_csv(uploaded_file)
                else:  # xlsx
                    df_preview = pd.read_excel(uploaded_file)
                
                st.write(f"{file_type.upper()} Preview:")
                st.dataframe(df_preview.head(), use_container_width=True)
                
                # Reset the file buffer position for later processing
                uploaded_file.seek(0)
                
                # Form for database details
                with st.form("db_details_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        table_name = st.text_input("Table Name", value="table1")
                    
                    with col2:
                        db_name = st.text_input("Database Name", value=f"{os.path.splitext(uploaded_file.name)[0]}_{uuid.uuid4().hex[:8]}")
                    
                    # Add .db extension if not present
                    if not db_name.endswith('.db'):
                        db_name = f"{db_name}.db"
                    
                    db_path = f"{user_db_dir}/{db_name}"
                    
                    submit_button = st.form_submit_button("Create Database")
                    
                    if submit_button:
                        # Reset the file buffer position
                        uploaded_file.seek(0)
                        
                        # Import file to SQLite
                        success, rows_imported, error_msg = import_file_to_sqlite(uploaded_file, table_name, db_path, file_type)
                        
                        if success:
                            st.success(f"Successfully created database '{db_name}' with {rows_imported} rows imported into table '{table_name}'")
                            
                            # Update session state
                            st.session_state.current_db = db_path
                            st.session_state.current_db_name = db_name
                            st.session_state.prompt = generate_prompt(db_path, db_name)
                            
                            st.success("Database created! Please go to the 'Query Interface' tab to start asking questions.")
                        else:
                            st.error(f"Failed to create database: {error_msg}")
            except Exception as e:
                st.error(f"Error previewing file: {str(e)}")
    
    # Display existing databases
    st.subheader("Available Databases")
    databases = list_available_databases()
    
    if databases:
        for i, db in enumerate(databases):
            with st.expander(f"{db['name']} (from {db['original_filename']})"):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"Created: {db['created_at']}")
                    st.write(f"File Type: {db.get('file_type', 'Unknown').upper()}")
                    st.write(f"Path: {db['path']}")
                
                with col2:
                    if st.button(f"Use This Database", key=f"use_db_{i}"):
                        st.session_state.current_db = db['path']
                        st.session_state.current_db_name = db['name']
                        st.session_state.prompt = generate_prompt(db['path'], db['name'])
                        st.success(f"Now using database: {db['name']}")
                        st.rerun()
                
                with col3:
                    if st.button(f"Delete", key=f"delete_db_{i}", type="primary", help="Delete this database permanently"):
                        success, error = delete_database(db['path'])
                        if success:
                            st.success(f"Database '{db['name']}' deleted successfully!")
                            # Clear session state if the deleted database was the current one
                            if st.session_state.current_db == db['path']:
                                st.session_state.current_db = None
                                st.session_state.current_db_name = None
                                st.session_state.prompt = None
                            st.rerun()
                        else:
                            st.error(f"Error deleting database: {error}")
    else:
        st.info("No databases available. Upload a CSV or XLSX file to create one.")

# Query Interface Tab
with tab1:
    if st.session_state.current_db is None:
        st.info("Please select or upload a database first (go to the 'Database Upload' tab)")
    else:
        st.info(f"Currently using database: {st.session_state.current_db_name}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            question = st.text_input("Ask a question about the data:", 
                                   placeholder="Example: Show all records in the table",
                                   key="input")
        
        with col2:
            submit = st.button("Run Query", type="primary", use_container_width=True)
        
        # Display the generated SQL and results
        if submit and question:
            with st.spinner("Generating SQL query..."):
                sql_query = get_gemini_response(question, st.session_state.prompt)
                
            # Display the SQL query in a code block
            st.subheader("Generated SQL Query:")
            st.code(sql_query, language="sql")
            
            # Execute the query and show results
            with st.spinner("Executing query..."):
                try:
                    rows, column_names = read_sql_query(sql_query, st.session_state.current_db)
                    
                    if rows and column_names:
                        st.subheader("Query Results:")
                        df = pd.DataFrame(rows, columns=column_names)
                        st.dataframe(df, use_container_width=True)
                        
                        # Add download button for results
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name=f"query_results_{int(time.time())}.csv",
                            mime="text/csv",
                        )
                        
                        # Show number of results
                        st.caption(f"Found {len(rows)} result{'s' if len(rows) != 1 else ''}")
                    elif column_names:  # Query executed but no results
                        st.info("Query executed successfully, but no results were returned.")
                except Exception as e:
                    st.error(f"Error executing query: {str(e)}")
                    st.warning("This might be due to referencing columns that don't exist in the database. Try rephrasing your question.")

# Database Reference Tab
with tab3:
    if st.session_state.current_db is None:
        st.info("Please select or upload a database first (go to the 'Database Upload' tab)")
    else:
        st.subheader(f"Database Structure: {st.session_state.current_db_name}")
        
        # Get and display table information
        table_info = get_table_info(st.session_state.current_db)
        
        if table_info:
            for table_name, info in table_info.items():
                st.markdown(f"### Table: {table_name}")
                
                # Display columns
                st.markdown("**Columns:**")
                st.code(", ".join(info['columns']))
                
                # Display sample data
                st.markdown("**Sample Data:**")
                df = pd.DataFrame(info['sample_data'], columns=info['columns'])
                st.dataframe(df, use_container_width=True)
        else:
            st.warning("Could not retrieve database structure. Make sure the database file exists.")

# Add footer with instructions
st.markdown("---")
st.markdown("""
**Tips for asking questions:**
- Use specific terms like "show", "find", "list", "count", etc.
- You can ask for filtering: "Show records where [column] equals [value]"
- You can ask for aggregations: "How many records are in each [column]?"
- You can ask for sorting: "List all records ordered by [column]"

**Session Information:**
- Your databases are temporary and will be deleted when your session expires
- Session timeout: 60 minutes of inactivity
- You can start a new session at any time using the button in the sidebar
""")