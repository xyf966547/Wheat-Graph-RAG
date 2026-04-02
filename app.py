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

# 定义支持的大模型配置字典
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
    },
    "Kimi (月之暗面)": {
        "api_key": "sk-bH4c2rhTy1qmW5Qr9HuhLanSskLUZfRjpeD4YmlBgvDqMppB", 
        "base_url": "https://api.moonshot.cn/v1",
        "model_name": "moonshot-v1-8k"
    },
    "智谱 GLM-4": {
        "api_key": "db3d6c35dd744a84a548b20c82b2766c.52EiUTIAvoH1sAZi", 
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model_name": "glm-4"
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
    .block-container { padding-top: 1.5rem; padding-bottom: 0rem; max-width: 100%; }
    .stChatMessage { background-color: #f8f9fa; border-radius: 10px; padding: 10px; margin-bottom: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title(" 知识图谱增强：小麦表型预测 (Graph-RAG)")

# --- 3. 初始化会话状态 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "你好！我是你的农业图谱 AI 助手。请先在左侧查询品种生成图谱，然后就可以向我提问啦！"}]
if "current_context" not in st.session_state:
    st.session_state.current_context = ""

# --- 4. 侧边栏控制区 ---
with st.sidebar:
    st.header(" AI 引擎设置")
    selected_models = st.multiselect(
        "选择进行对比的大模型 (可多选):", 
        list(MODEL_CONFIGS.keys()),
        default=["DeepSeek"]
    )
    st.markdown("---")

    st.header(" 图谱查询")
    search_variety = st.text_input("输入查询品种 (如 GID1 或 ADT_52):", "GID1")
    search_btn = st.button("生成图谱并载入 AI 上下文")
    
    st.markdown("---")
    st.header(" 知识库扩展 (模拟新数据)")
    st.info("在此输入新表型，它将自动挂载到上方查询的品种节点上。")
    
    new_p_name = st.text_input("新表型名称 (如: 叶绿素含量):")
    new_p_val = st.text_input("测定数值 (如: 55.2):")
    
    if st.button("将新数据录入并自动挂载"):
        if new_p_name and new_p_val:
            with driver.session() as session:
                check_query = "MATCH (v:Variety {id: $vid}) RETURN v"
                if not session.run(check_query, vid=search_variety).single():
                    st.error(f"图谱中不存在品种 {search_variety}，请先在上方输入正确的品种。")
                else:
                    query = """
                    MATCH (v:Variety {id: $vid})
                    MERGE (p:Phenotype {name: $pname})
                    MERGE (v)-[r:HAS_PHENOTYPE]->(p)
                    SET r.value = $val
                    """
                    session.run(query, vid=search_variety, pname=new_p_name, val=float(new_p_val))
                    st.success(f" 成功！表型 '{new_p_name}' 已挂载。请重新点击最上方的【生成图谱】按钮查看变化。")
        else:
            st.error("请完整填写表型名称和数值！")
            
    st.markdown("---")
    if st.button(" 清空聊天记录"):
        st.session_state.messages = [{"role": "assistant", "content": "聊天记录已清空。"}]
        st.rerun()

# --- 5. 核心布局：左右分栏 ---
col_graph, col_chat = st.columns([7, 3], gap="large")

# ==================== 左侧：图谱可视化区 ====================
with col_graph:
    st.subheader(" 知识图谱视图")
    
    if search_btn:
        with st.spinner('正在渲染图谱并提取知识...'):
            st.session_state.current_context = rag_tool.extract_features_for_llm(search_variety)
            
            with driver.session() as session:
                # 1. 查：品种 -> 表型
                query_pheno = """
                MATCH (v:Variety {id: $vid})-[r:HAS_PHENOTYPE]->(p:Phenotype)
                RETURN v AS n1, type(r) AS rel_type, r AS rel_props, p AS n2 
                """
                records_pheno = list(session.run(query_pheno, vid=search_variety))

                # 2. 查：品种 -> 基因型 (抽取100个代表)
                query_snp = """
                MATCH (v:Variety {id: $vid})-[r:HAS_ALLELE]->(s:SNP)
                RETURN v AS n1, type(r) AS rel_type, r AS rel_props, s AS n2 
                LIMIT 100
                """
                records_snp = list(session.run(query_snp, vid=search_variety))
                
                # 3. 【全新升级】：基因型 -> 表型调控关系 (随机洗牌，雨露均沾版)
                query_sp = """
                MATCH (v:Variety {id: $vid})-[r_allele:HAS_ALLELE]->(s:SNP)
                MATCH (v)-[:HAS_PHENOTYPE]->(p:Phenotype)
                MATCH (s)-[r_inf:INFLUENCES]->(p)
                WITH v, s, r_allele, r_inf, p
                ORDER BY rand()  // 核心魔法：彻底打乱顺序，让所有表型公平竞争
                LIMIT 200        // 抽取200条精美的调控通路，避免浏览器卡死
                RETURN v, s, r_allele, r_inf, p
                """
                sp_results = session.run(query_sp, vid=search_variety)
                
                records_sp = []
                for res in sp_results:
                    # 1. 画出漂亮的紫线 (调控关系: SNP -> 表型)
                    records_sp.append({
                        "n1": res["s"], 
                        "rel_type": "INFLUENCES", 
                        "rel_props": res["r_inf"], 
                        "n2": res["p"]
                    })
                    # 2. 强行补上品种到这些 SNP 的连线，防止被抽中的基因节点孤立悬空
                    records_sp.append({
                        "n1": res["v"], 
                        "rel_type": "HAS_ALLELE", 
                        "rel_props": res["r_allele"], 
                        "n2": res["s"]
                    })
                
                # 将三部分数据合并画图
                records = records_pheno + records_snp + records_sp

            if not records:
                st.warning(f"未找到品种 {search_variety} 的关联数据！如果确实没数据，请尝试查询其他品种。")
            else:
                net = Network(height="1000px", width="100%", bgcolor="#ffffff", font_color="black", cdn_resources='remote')
                net.force_atlas_2based(gravity=-60, spring_length=150)
                nodes_added = set()

                def add_node(node_dict):
                    n_id = str(node_dict.get("id") or node_dict.get("name"))
                    if n_id not in nodes_added:
                        labels = list(node_dict.labels)
                        # 已知的核心节点
                        if "Variety" in labels: net.add_node(n_id, label=n_id, color="#2ECC71", size=30, font={"size": 16, "bold": True})
                        elif "Phenotype" in labels: net.add_node(n_id, label=n_id, color="#FF7675", size=20, shape="hexagon")
                        elif "SNP" in labels: net.add_node(n_id, label=n_id, color="#F39C12", size=15)
                        
                        # 🛡️【前端自适应兜底】：如果遇到未来新增的、代码不认识的节点，统一画成灰色圆点，绝不报错！
                        else: 
                            net.add_node(n_id, label=n_id, color="#95A5A6", size=15, shape="dot")
                            
                        nodes_added.add(n_id)
                    return n_id

                for record in records:
                    id1 = add_node(record["n1"])
                    id2 = add_node(record["n2"])
                    if not id1 or not id2: continue
                    
                    rel_type = record["rel_type"]
                    # 【连线自适应兜底】：默认全部用灰色实线画出来
                    color, dash, label = "#BDC3C7", False, str(rel_type)
                    
                    # 如果是认识的，再覆盖成特定颜色
                    if rel_type == "HAS_PHENOTYPE": 
                        color, label = "#A9DFBF", str(record["rel_props"].get("value", ""))
                    elif rel_type == "HAS_ALLELE": 
                        color, label = "#FAD7A1", str(record["rel_props"].get("allele", ""))
                    elif rel_type == "INFLUENCES": 
                        color, dash, label = "#9B59B6", True, "调控"
                    
                    try: net.add_edge(id1, id2, title=rel_type, label=label, arrows="to", color=color, dashes=dash)
                    except: pass

                try:
                    net.save_graph("kg_graph.html")
                    with open("kg_graph.html", 'r', encoding='utf-8') as f:
                        st.session_state["saved_graph_html"] = f.read()
                    st.success(f"图谱渲染完毕。已显示 {search_variety} 的表型、部分关键基因型数据，及其调控关系。")
                except Exception as e:
                    st.error(f"渲染错误: {e}")

    if "saved_graph_html" in st.session_state:
        components.html(st.session_state["saved_graph_html"], height=1000, scrolling=True)


# ==================== 右侧：AI 对话聊天区 ====================
with col_chat:
    st.subheader(" 图谱 AI 助手 (多模型对比)")
    chat_container = st.container(height=1200)
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if isinstance(message["content"], dict):
                    cols = st.columns(len(message["content"]))
                    for idx, (m_name, m_reply) in enumerate(message["content"].items()):
                        with cols[idx]:
                            st.markdown(f"**[{m_name}]**")
                            st.markdown(m_reply)
                else:
                    st.markdown(message["content"])

    if prompt := st.chat_input("向我提问，例如：预测它的抗旱指数并给出理由"):
        if not selected_models:
            st.warning(" 请至少在左侧选择一个大模型！")
            st.stop()
            
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st.markdown(" **多模型正在并行思考与对比中...**")
                
                system_prompt = f"""
                你是一个资深的农业计算生物学大模型。你的任务是解答用户的提问。
                【当前激活图谱的底层硬核数据如下，这是回答用户问题的绝对依据】：
                {st.session_state.current_context}
                
                请结合上述知识图谱中提取的精确数值和基因型，进行专业、严谨的分析回答。
                """
                
                api_messages = [{"role": "system", "content": system_prompt}]
                
                for msg in st.session_state.messages[-6:]:
                    if isinstance(msg["content"], dict):
                        combined_memory = "\n\n".join([f"[{k}] 的回答: {v}" for k, v in msg["content"].items()])
                        api_messages.append({"role": "assistant", "content": combined_memory})
                    else:
                        api_messages.append(msg)
                
                cols = st.columns(len(selected_models))
                model_responses = {} 
                
                for idx, model_name in enumerate(selected_models):
                    config = MODEL_CONFIGS[model_name]
                    with cols[idx]:
                        st.markdown(f"###  {model_name}")
                        response_placeholder = st.empty()
                        response_placeholder.markdown(" 思考中...")
                        
                        try:
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
                            
                            response_placeholder.markdown(reply_text)
                            model_responses[model_name] = reply_text
                            
                        except Exception as e:
                            error_msg = f" 调用失败\n报错信息: {e}"
                            response_placeholder.error(error_msg)
                            model_responses[model_name] = error_msg
                
                st.session_state.messages.append({"role": "assistant", "content": model_responses})
