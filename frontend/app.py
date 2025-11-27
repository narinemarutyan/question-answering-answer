"""Streamlit UI for the Question-Answering Agent."""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from datetime import datetime

from ai.agents import kb_agent
from backend.db import (
    init_db,
    create_session,
    get_messages,
    list_sessions,
    delete_session,
    add_message,
)
from ai.vector_store import add_document, list_documents, delete_document

# Initialize database
init_db()

# Page configuration
st.set_page_config(
    page_title="Q&A Agent",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def load_chat_messages(chat_id: int):
    """Load messages for a chat and update session state."""
    messages = get_messages(chat_id)
    st.session_state.messages = messages
    st.session_state.current_chat_id = chat_id


def create_new_chat():
    """Create a new chat session."""
    chat_id = create_session()
    st.session_state.current_chat_id = chat_id
    st.session_state.messages = []
    return chat_id


def format_timestamp(timestamp_str: str) -> str:
    """Format timestamp for display."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return timestamp_str


# Sidebar
with st.sidebar:
    st.title("💬 Q&A Agent")
    st.markdown("---")
    
    # Chat Management
    st.subheader("Chat Management")
    
    if st.button("➕ New Chat", use_container_width=True):
        create_new_chat()
        st.rerun()
    
    st.markdown("---")
    
    # List of chats
    st.subheader("Chat History")
    sessions = list_sessions()
    
    if not sessions:
        st.info("No chats yet. Create a new chat to get started!")
    else:
        for session in sessions:
            col1, col2 = st.columns([4, 1])
            with col1:
                is_selected = st.button(
                    f"Chat #{session['id']}",
                    key=f"chat_{session['id']}",
                    use_container_width=True,
                )
                if is_selected:
                    load_chat_messages(session["id"])
                    st.rerun()
            
            with col2:
                if st.button("🗑️", key=f"delete_{session['id']}"):
                    delete_session(session["id"])
                    if st.session_state.current_chat_id == session["id"]:
                        st.session_state.current_chat_id = None
                        st.session_state.messages = []
                    st.rerun()
    
    st.markdown("---")
    
    # Knowledge Base Management
    st.subheader("📚 Knowledge Base")
    
    # Upload document
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["txt"],
        help="Upload a .txt file to add to the knowledge base",
    )
    
    if uploaded_file is not None:
        if st.button("Upload to Knowledge Base", use_container_width=True):
            try:
                content = uploaded_file.read().decode("utf-8")
                doc_hash = add_document(content, file_name=uploaded_file.name)
                st.success(f"✅ Document uploaded successfully!")
                st.info(f"Document hash: {doc_hash[:16]}...")
            except Exception as e:
                st.error(f"Error uploading document: {str(e)}")
    
    # List documents
    st.markdown("**Documents in Knowledge Base:**")
    documents = list_documents()
    if documents:
        for doc_hash in documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"📄 {doc_hash[:16]}...")
            with col2:
                if st.button("🗑️", key=f"del_doc_{doc_hash}"):
                    delete_document(doc_hash)
                    st.rerun()
    else:
        st.info("No documents in knowledge base")


# Main content area
st.title("💬 Question-Answering Agent")
st.markdown("Ask questions and get answers powered by GPT-4 with knowledge base search!")

# Display current chat info
if st.session_state.current_chat_id:
    st.info(f"📝 Current Chat: #{st.session_state.current_chat_id}")
else:
    st.warning("⚠️ No chat selected. Create a new chat or select an existing one from the sidebar.")

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role):
            st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Ensure we have a chat session
    if st.session_state.current_chat_id is None:
        create_new_chat()
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare messages for agent
                agent_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                ]
                
                # Invoke agent
                result = kb_agent.invoke(
                    {"messages": agent_messages},
                    context={"user_role": "expert"},
                )
                
                reply = result["messages"][-1].content
                
                # Display response
                st.markdown(reply)
                
                # Add assistant message to session state
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
                # Save messages to database
                add_message(st.session_state.current_chat_id, "user", prompt)
                add_message(st.session_state.current_chat_id, "assistant", reply)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

