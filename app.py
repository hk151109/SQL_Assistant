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

# Load environment variables
load_dotenv()

# Configure the API Key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    st.error("No API key found. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=api_key)

# Ensure directories exist
os.makedirs("databases", exist_ok=True)
os.makedirs("metadata", exist_ok=True)

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
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        # Get column names
        column_names = [description[0] for description in cur.description]
        conn.close()
        return rows, column_names
    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
        return [], []
    except Exception as e:
        st.error(f"Error executing query: {str(e)}")
        return [], []

# Function to get all tables in the database
def get_table_info(db_path):
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
        
        conn.close()
        return table_info
    except Exception as e:
        st.error(f"Error getting table information: {str(e)}")
        return {}

# Function to import CSV to SQLite database
def import_csv_to_sqlite(csv_file, table_name, db_path):
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        
        # Write dataframe to SQLite
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Close connection
        conn.close()
        
        # Save metadata about the database
        metadata = {
            "tables": [table_name],
            "original_filename": csv_file.name,
            "created_at": pd.Timestamp.now().isoformat()
        }
        
        metadata_path = f"metadata/{os.path.basename(db_path)}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return True, df.shape[0]
    except Exception as e:
        st.error(f"Error importing CSV: {str(e)}")
        return False, 0

# Function to list available databases
def list_available_databases():
    try:
        db_files = glob.glob("databases/*.db")
        databases = []
        
        for db_file in db_files:
            db_name = os.path.basename(db_file)
            
            # Try to load metadata
            metadata_file = f"metadata/{db_name}.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    original_filename = metadata.get('original_filename', 'Unknown')
                    created_at = metadata.get('created_at', 'Unknown')
            else:
                original_filename = 'Unknown'
                created_at = 'Unknown'
            
            databases.append({
                'path': db_file,
                'name': db_name,
                'original_filename': original_filename,
                'created_at': created_at
            })
        
        return databases
    except Exception as e:
        st.error(f"Error listing databases: {str(e)}")
        return []

# Streamlit App
st.set_page_config(page_title="SQL Natural Language Query Assistant", page_icon="🔍", layout="wide")
st.title("🔍 SQL Natural Language Query Assistant")
st.markdown("""
This app translates your natural language questions into SQL queries and retrieves data from databases.
Upload a CSV file to create a new database, or select an existing database to query.
""")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Query Interface", "Database Upload", "Database Reference"])

# Initialize session state
if 'current_db' not in st.session_state:
    st.session_state.current_db = None
if 'current_db_name' not in st.session_state:
    st.session_state.current_db_name = None
if 'prompt' not in st.session_state:
    st.session_state.prompt = None

# Database Upload Tab
with tab2:
    st.subheader("Upload New Database")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Preview the CSV
        df_preview = pd.read_csv(uploaded_file)
        st.write("CSV Preview:")
        st.dataframe(df_preview.head(), use_container_width=True)
        
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
            
            db_path = f"databases/{db_name}"
            
            submit_button = st.form_submit_button("Create Database")
            
            if submit_button:
                # Reset the file buffer position
                uploaded_file.seek(0)
                
                # Import CSV to SQLite
                success, rows_imported = import_csv_to_sqlite(uploaded_file, table_name, db_path)
                
                if success:
                    st.success(f"Successfully created database '{db_name}' with {rows_imported} rows imported into table '{table_name}'")
                    
                    # Update session state
                    st.session_state.current_db = db_path
                    st.session_state.current_db_name = db_name
                    st.session_state.prompt = generate_prompt(db_path, db_name)
                    
                    # Note: Instead of using a button to switch tabs, we'll use a success message
                    st.success("Database created! Please go to the 'Query Interface' tab to start asking questions.")
                else:
                    st.error("Failed to create database. Please check your CSV file and try again.")
    
    # Display existing databases
    st.subheader("Available Databases")
    databases = list_available_databases()
    
    if databases:
        for i, db in enumerate(databases):
            with st.expander(f"{db['name']} (from {db['original_filename']})"):
                st.write(f"Created: {db['created_at']}")
                st.write(f"Path: {db['path']}")
                
                if st.button(f"Use This Database", key=f"use_db_{i}"):
                    st.session_state.current_db = db['path']
                    st.session_state.current_db_name = db['name']
                    st.session_state.prompt = generate_prompt(db['path'], db['name'])
                    st.success(f"Now using database: {db['name']}")
    else:
        st.info("No databases available. Upload a CSV file to create one.")

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
""")