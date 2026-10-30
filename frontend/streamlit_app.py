
import os

import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="Customer Support AI",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Customer Support AI")
st.caption(
    "Ask a question about orders, payments, refunds, delivery, "
    "accounts, and more."
)

with st.sidebar:
    st.header("About")
    st.write(
        "This assistant uses semantic search over customer-support "
        "FAQs and a Groq language model to generate answers."
    )

    st.markdown("**Technology stack**")
    st.write(
        "- FastEmbed (ONNX embeddings)\n"
        "- Qdrant vector database\n"
        "- LangGraph and LangChain\n"
        "- Groq LLM\n"
        "- FastAPI"
    )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display the existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("View retrieved FAQs"):
                for index, faq in enumerate(message["sources"], start=1):
                    st.markdown(f"**FAQ {index}**")
                    st.write("Question:", faq.get("question", ""))
                    st.write("Intent:", faq.get("intent", ""))
                    st.write(
                        "Similarity score:",
                        round(float(faq.get("score", 0)), 4),
                    )

        if message["role"] == "assistant" and message.get("processing_time_ms") is not None:
            st.caption(
                f"Processing time: "
                f"{message['processing_time_ms'] / 1000:.2f} seconds"
            )

question = st.chat_input("How can I cancel my order?")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching FAQs and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"question": question},
                    timeout=90,
                )
                response.raise_for_status()
                result = response.json()

                answer = result["answer"]
                sources = result.get("retrieved_faqs", [])
                processing_time_ms = result.get("processing_time_ms")

                st.markdown(answer)

                if sources:
                    with st.expander("View retrieved FAQs"):
                        for index, faq in enumerate(sources, start=1):
                            st.markdown(f"**FAQ {index}**")
                            st.write("Question:", faq.get("question", ""))
                            st.write("Intent:", faq.get("intent", ""))
                            st.write(
                                "Similarity score:",
                                round(float(faq.get("score", 0)), 4),
                            )

                if processing_time_ms is not None:
                    st.caption(
                        f"Processing time: "
                        f"{processing_time_ms / 1000:.2f} seconds"
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "processing_time_ms": processing_time_ms,
                    }
                )

            except requests.exceptions.Timeout:
                error_message = (
                    "The request took too long. Please try again."
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

            except requests.exceptions.ConnectionError:
                error_message = (
                    "Cannot connect to the backend. Make sure FastAPI "
                    "is running and API_URL is configured correctly."
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

            except requests.exceptions.HTTPError as exc:
                detail = exc.response.text[:500] if exc.response else str(exc)
                error_message = f"The backend returned an error: {detail}"
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )

            except (KeyError, ValueError) as exc:
                error_message = (
                    f"The backend returned an unexpected response: {exc}"
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
