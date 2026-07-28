import streamlit as st
import speech_recognition as sr
import base64


def _render_content_item(item: dict) -> None:
    """Render Gemini/LangChain multimodal content instead of image placeholders."""
    if item.get("type") == "text":
        st.write(item.get("text", ""))
        return

    if item.get("type") != "image_url":
        return

    url = item.get("image_url", {}).get("url", "")
    if not url.startswith("data:image/") or ";base64," not in url:
        st.caption("Image unavailable")
        return
    try:
        st.image(base64.b64decode(url.split(";base64,", 1)[1]), width="stretch")
    except Exception:
        st.caption("Image could not be decoded")

def render_conversation_tab():
    if not st.session_state.agent: return st.info("LLM Agent offline.")

    st.markdown("""
        <style>
        button[data-testid="stChatInputSubmitButton"] { display: none !important; }
        div[data-testid="stChatInput"] { margin-right: 0 !important; }
        </style>
    """, unsafe_allow_html=True)

    col_v, col_c = st.columns([1, 1])
    
    with col_v:
        try:
            imgs = st.session_state.agent.fetch_camera_images_base64()
            if imgs: st.image(f"data:image/jpeg;base64,{imgs[0]}", width="stretch")
        except: st.error("Vision broken")
        
        try:
            if getattr(st.session_state.agent, "latest_lidar_b64", None):
                st.image(f"data:image/png;base64,{st.session_state.agent.latest_lidar_b64}", width="stretch")
        except: pass
        
        status_container = st.container()
        grasp = st.session_state.get("grasp_controller")
        if grasp is not None and grasp.status.get("stage") not in (None, "idle"):
            st.caption(f"Grasp: {grasp.status.get('stage')} - {grasp.status.get('message', '')}")
                
    with col_c:
        reasoning_tokens = getattr(st.session_state.agent, "latest_reasoning_tokens", 0)
        if reasoning_tokens:
            st.caption(f"Gemini reasoning used in the latest step: {reasoning_tokens} tokens")
        chat_container = st.container(height=370)
        with chat_container:
            for msg in st.session_state.agent.message_history:
                if msg.type == "system": continue
                if (
                    msg.type == "human"
                    and isinstance(msg.content, list)
                    and any(
                        isinstance(item, dict) and item.get("text") == "Main camera view:"
                        for item in msg.content
                    )
                ):
                    continue
                if msg.type == "tool":
                    with st.chat_message("assistant"):
                        label = "Agent observation and plan" if msg.name == "report_observation_and_plan" else (msg.name or "System Action")
                        with st.expander(f"🛠️ {label}"): st.write(msg.content)
                    continue
                observation = msg.type == "human" and str(getattr(msg, "name", "")).endswith("_observation")
                with st.chat_message("assistant" if observation else ("user" if msg.type == "human" else "assistant")):
                    if observation:
                        with st.expander(str(msg.name).replace("_observation", "").replace("_", " ").title()):
                            for item in msg.content:
                                _render_content_item(item)
                        continue
                    if isinstance(msg.content, str): st.write(msg.content)
                    elif isinstance(msg.content, list):
                        for item in msg.content:
                            _render_content_item(item)
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls: st.info(f"⚙️ {tc['name']}")
        
        c_in, c_mic = st.columns([4.2, 2.8], vertical_alignment="bottom")
        with c_in: p_text = st.chat_input("Command...", disabled=st.session_state.agent_active)
        with c_mic: audio = st.audio_input("Mic", disabled=st.session_state.agent_active, label_visibility="collapsed")
        
        final_p = p_text
        if audio and audio != st.session_state.get("last_audio"):
            st.session_state.last_audio = audio
            try:
                r = sr.Recognizer()
                with sr.AudioFile(audio) as src: final_p = r.recognize_google(r.record(src))
            except: status_container.error("Mic error")

        if final_p:
            st.session_state.agent.task, st.session_state.agent_active, st.session_state.agent_step = final_p, True, 0
            st.rerun()

    if st.session_state.agent_active:
        with status_container:
            if st.button("🛑 STOP", use_container_width=True):
                grasp = st.session_state.get("grasp_controller")
                if grasp is not None:
                    grasp.cancel()
                st.session_state.agent_active, st.session_state.agent.task = False, None
                st.rerun()
                
            with st.spinner(f"🧠 Step {st.session_state.agent_step+1}"):
                try:
                    res = st.session_state.agent.main_loop_content()
                except Exception as e:
                    if "Check bit not equal to 1" in str(e) or "Wrong body size" in str(e):
                        st.rerun()
                    else:
                        raise e
                st.session_state.agent_step += 1
        
        last_msg = st.session_state.agent.message_history[-1]
        
        if res is not None or st.session_state.agent.task is None or (
            last_msg.type == "ai" and not getattr(last_msg, "tool_calls", [])
        ):
            st.session_state.agent_active, st.session_state.agent.task = False, None
            
        st.rerun()
