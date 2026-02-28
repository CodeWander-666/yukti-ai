# ---------- Main Chat Interface ----------
# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response using thinking engine
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        final_output = {"answer": "", "monologue": "", "sources": [], "thinking_time": 0.0}

        try:
            if not st.session_state.knowledge_base_ready:
                final_output["answer"] = "The knowledge base is not ready. Please click 'Update Knowledge Base' in the sidebar first."
                response_placeholder.markdown(final_output["answer"])
            else:
                from think import think
                history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[-10:]
                ]

                # Show "Thinking..." with a spinner and timer
                with st.spinner(f"Thinking... (0s)"):
                    start_time = time.time()
                    # Update spinner with elapsed time
                    spinner_placeholder = st.empty()
                    while time.time() - start_time < 0.5:  # brief initial delay
                        time.sleep(0.1)
                    # After initial delay, show timer
                    while True:
                        elapsed = time.time() - start_time
                        spinner_placeholder.text(f"Thinking... ({elapsed:.1f}s)")
                        if elapsed > 0.2:  # arbitrary – we'll just loop until done
                            # In reality, we'd need async to update during LLM call.
                            # For simplicity, we'll just show one final time.
                            break
                    # Actually call think() – this blocks
                    final_output = think(prompt, history)
                    spinner_placeholder.empty()  # remove spinner

                # Display thinking time
                st.caption(f"Thought for {final_output['thinking_time']:.2f}s")

                # Display thinking process toggle
                if final_output["monologue"]:
                    with st.expander("Show thinking process"):
                        st.markdown(final_output["monologue"])

                # Display the answer
                response_placeholder.markdown(final_output["answer"])

                # Display sources if available
                if final_output["sources"]:
                    with st.expander("View source documents"):
                        for i, doc in enumerate(final_output["sources"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.write(doc.page_content)
                            if i < len(final_output["sources"]) - 1:
                                st.divider()

        except FileNotFoundError as e:
            final_output["answer"] = "The knowledge base is not ready. Please click 'Update Knowledge Base' in the sidebar first."
            response_placeholder.markdown(final_output["answer"])
        except Exception as e:
            final_output["answer"] = f"An error occurred: {e}"
            response_placeholder.markdown(final_output["answer"])

    # Save assistant message (store only the answer for history)
    st.session_state.messages.append({"role": "assistant", "content": final_output["answer"]})
