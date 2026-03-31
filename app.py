import streamlit as st
import streamlit.components.v1 as components
from neo4j import GraphDatabase
from pyvis.network import Network
import os
from dotenv import load_dotenv
from openai import OpenAI

# 导入之前写的图谱特征提取器
from graph_rag import WheatGraphRAG

load_dotenv()

# --- 1. 初始化配置 ---
URI = "bolt://localhost:7687"  
AUTH = ("neo4j", "xuyf10224219")  

# 定义支持的大模型配置字典 (你可以随时在这里添加新的模型，比如 Kimi, 智谱等)
MODEL_CONFIGS = {
    "DeepSeek": {
        "api_key": "sk-ab8f579e0c7e4ab6b759dc1531682b9b", 
        "base_url": "https://api.deepseek.com",
        "model_name": "deepseek-chat"
    },
    "通义千问 (Qwen)": {
        "api_key": "sk-d6a929086835478c96bea0cbc6e4a3ae", 
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model_name": "qwen-plus"
    }
}

@st.cache_resource
def get_driver():
    return GraphDatabase.driver(URI, auth=AUTH)

driver = get_driver()
rag_tool = WheatGraphRAG(URI, AUTH)


# --- 2. 页面与样式配置 ---
st.set_page_config(layout="wide", page_title="小麦图谱查询系统")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; max-width: 100%; } /* 这里之前少了一个 } */
    /* 让聊天窗口有滚动条，且高度固定，适配右侧布局 */
    .stChatMessage { background-color: #f8f9fa; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🌾 知识图谱增强：小麦表型预测 (Graph-RAG )")

# --- 3. 初始化会话状态 (Session State) ---
# 用于保存聊天记录和当前激活的图谱上下文
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是你的农业图谱 AI 助手。请先在左侧查询品种生成图谱，然后就可以向我提问啦！"}]
if "current_context" not in st.session_state:
    st.session_state.current_context = ""

# --- 4. 侧边栏控制区 ---
with st.sidebar:
    # === 修改为多选框 ===
    st.header("🤖 AI 引擎设置")
    selected_models = st.multiselect(
        "选择进行对比的大模型 (可多选):", 
        list(MODEL_CONFIGS.keys()),
        default=["DeepSeek"]
    )
    st.markdown("---")

    st.header("🔍 图谱查询")
    search_variety = st.text_input("输入查询品种 (如 GID1 或 ADT_1):", "GID1")
    search_btn = st.button("生成图谱并载入 AI 上下文")
    
    # === 知识库动态扩展区 ===
    st.markdown("---")
    st.header("🛠️ 知识库扩展 (模拟新数据)")
    st.info("在此输入新表型，它将自动挂载到上方查询的品种节点上。")
    
    new_p_name = st.text_input("新表型名称 (如: 叶绿素含量):")
    new_p_val = st.text_input("测定数值 (如: 55.2):")
    
    if st.button("将新数据录入并自动挂载"):
        if new_p_name and new_p_val:
            # 只有当确保品种存在时，才允许挂载
            with driver.session() as session:
                # 检查品种是否存在
                check_query = "MATCH (v:Variety {id: $vid}) RETURN v"
                if not session.run(check_query, vid=search_variety).single():
                    st.error(f"图谱中不存在品种 {search_variety}，请先在上方输入正确的品种。")
                else:
                    # 使用 MERGE 动态创建表型节点并建立连线
                    query = """
                    MATCH (v:Variety {id: $vid})
                    MERGE (p:Phenotype {name: $pname})
                    MERGE (v)-[r:HAS_PHENOTYPE]->(p)
                    SET r.value = $val
                    """
                    session.run(query, vid=search_variety, pname=new_p_name, val=float(new_p_val))
                    st.success(f"✅ 成功！表型 '{new_p_name}' 已挂载。请重新点击最上方的【生成图谱】按钮查看变化。")
        else:
            st.error("请完整填写表型名称和数值！")
            
    # === 聊天记录清空按钮 ===
    st.markdown("---")
    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = [{"role": "assistant", "content": "聊天记录已清空。"}]
        st.rerun()

# --- 5. 核心布局：左右分栏 ---
# col_graph 占 70% 宽度，col_chat 占 30% 宽度
col_graph, col_chat = st.columns([7, 3], gap="large")

# ==================== 左侧：图谱可视化区 ====================
with col_graph:
    st.subheader("🕸️ 知识图谱视图")
    
    # 1. 只有当点击左侧的“生成”按钮时，才去数据库拉取数据并重新画图
    if search_btn:
        with st.spinner('正在渲染图谱并提取知识...'):
            st.session_state.current_context = rag_tool.extract_features_for_llm(search_variety)
            
            with driver.session() as session:
                query = """
                MATCH (v:Variety {id: $vid})-[r]-(n)
                RETURN v AS n1, type(r) AS rel_type, r AS rel_props, n AS n2 
                LIMIT 200
                """
                records = list(session.run(query, vid=search_variety))

            if not records:
                st.warning(f"未找到品种 {search_variety} 的直接关联数据！")
            else:
                net = Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="black", cdn_resources='remote')
                net.force_atlas_2based(gravity=-60, spring_length=150)
                nodes_added = set()

                def add_node(node_dict):
                    n_id = str(node_dict.get("id") or node_dict.get("name"))
                    if n_id not in nodes_added:
                        labels = list(node_dict.labels)
                        if "Variety" in labels: net.add_node(n_id, label=n_id, color="#2ECC71", size=30, font={"size": 16, "bold": True})
                        elif "Phenotype" in labels: net.add_node(n_id, label=n_id, color="#FF7675", size=20, shape="hexagon")
                        elif "SNP" in labels: net.add_node(n_id, label=n_id, color="#F39C12", size=15)
                        nodes_added.add(n_id)
                    return n_id

                for record in records:
                    id1 = add_node(record["n1"])
                    id2 = add_node(record["n2"])
                    if not id1 or not id2: continue
                    
                    rel_type = record["rel_type"]
                    color, dash, label = "#BDC3C7", False, ""
                    if rel_type == "HAS_PHENOTYPE": color, label = "#A9DFBF", str(record["rel_props"].get("value", ""))
                    elif rel_type == "HAS_ALLELE": color, label = "#FAD7A1", str(record["rel_props"].get("allele", ""))
                    
                    try: net.add_edge(id1, id2, title=rel_type, label=label, arrows="to", color=color, dashes=dash)
                    except: pass

                try:
                    net.save_graph("kg_graph.html")
                    with open("kg_graph.html", 'r', encoding='utf-8') as f:
                        # 【核心记忆功能】：只存入 session_state，绝对不要在这里直接画图！
                        st.session_state["saved_graph_html"] = f.read()
                    st.success(f"图谱渲染完毕。当前仅显示 {search_variety} 的直接测定数据。")
                except Exception as e:
                    st.error(f"渲染错误: {e}")

    # 2. 【唯一渲染出口】：整个左侧区域，只允许在这里出现一次 components.html
    if "saved_graph_html" in st.session_state:
        components.html(st.session_state["saved_graph_html"], height=1000, scrolling=True)


# ==================== 右侧：AI 对话聊天区 ====================
with col_chat:
    st.subheader("💬 图谱 AI 助手 (多模型对比)")
    
    # 一个定高的容器，用来装聊天记录
    chat_container = st.container(height=1200)
    
    # 渲染历史聊天记录
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                # 如果是多模型对比的结果（字典格式），则拆分多列渲染
                if isinstance(message["content"], dict):
                    cols = st.columns(len(message["content"]))
                    for idx, (m_name, m_reply) in enumerate(message["content"].items()):
                        with cols[idx]:
                            st.markdown(f"**[{m_name}]**")
                            st.markdown(m_reply)
                else:
                    st.markdown(message["content"])

    # 聊天输入框 (放在下方)
    if prompt := st.chat_input("向我提问，例如：预测它的抗旱指数并给出理由"):
        
        # 拦截：如果用户把左侧的模型全清空了，提示他选一个
        if not selected_models:
            st.warning("⚠️ 请至少在左侧选择一个大模型！")
            st.stop()
            
        # 1. 把用户的输入展示出来并存入历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. 调用大模型进行回复
            with st.chat_message("assistant"):
                st.markdown("🧠 **多模型正在并行思考与对比中...**")
                
                # 构建发给大模型的系统指令
                system_prompt = f"""
                你是一个资深的农业计算生物学大模型。你的任务是解答用户的提问。
                【当前激活图谱的底层硬核数据如下，这是回答用户问题的绝对依据】：
                {st.session_state.current_context}
                
                请结合上述知识图谱中提取的精确数值和基因型，进行专业、严谨的分析回答。
                """
                
                api_messages = [{"role": "system", "content": system_prompt}]
                
                # 附带历史聊天记录
                for msg in st.session_state.messages[-6:-1]:
                    if isinstance(msg["content"], dict):
                        # 把之前多模型的回答融合成一段文本，作为记忆喂给 AI
                        combined_memory = "\n\n".join([f"[{k}] 的回答: {v}" for k, v in msg["content"].items()])
                        api_messages.append({"role": "assistant", "content": combined_memory})
                    else:
                        api_messages.append(msg)
                
                # === 核心逻辑：动态分列 ===
                cols = st.columns(len(selected_models))
                model_responses = {} 
                
                for idx, model_name in enumerate(selected_models):
                    config = MODEL_CONFIGS[model_name]
                    # 在专属的列中渲染
                    with cols[idx]:
                        st.markdown(f"### 💡 {model_name}")
                        response_placeholder = st.empty()
                        response_placeholder.markdown("⏳ 思考中...")
                        
                        try:
                            # 实例化对应的 API
                            dynamic_client = OpenAI(
                                api_key=config["api_key"], 
                                base_url=config["base_url"]
                            )
                            
                            response = dynamic_client.chat.completions.create(
                                model=config["model_name"],
                                messages=api_messages,
                                temperature=0.3
                            )
                            reply_text = response.choices[0].message.content
                            
                            # 覆盖掉“思考中”的提示
                            response_placeholder.markdown(reply_text)
                            model_responses[model_name] = reply_text
                            
                        except Exception as e:
                            error_msg = f"❌ 调用失败\n报错信息: {e}"
                            response_placeholder.error(error_msg)
                            model_responses[model_name] = error_msg
                
                # 将字典格式的对比结果存入会话
                st.session_state.messages.append({"role": "assistant", "content": model_responses})