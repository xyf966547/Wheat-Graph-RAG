import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase

# 引入我们上一步写的图谱特征提取器
from graph_rag import WheatGraphRAG

load_dotenv()

# --- 1. 初始化配置 ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

API_KEY = os.getenv("DEEPSEEK_API_KEY")
BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 初始化 LLM 客户端
llm_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm_for_prediction(prompt_context):
    """调用大模型进行推理预测"""
    print("\n⏳ 正在呼叫大模型进行推理，请稍候...")
    
    # 增加系统指令，规范大模型的输出格式
    system_instruction = """
    你是一个严谨的农业计算生物学大模型。
    请根据提供的表型和基因型上下文，推测目标表型的数值。
    在给出你的推理过程后，你必须将最终预测的唯一具体数值放在 <prediction> 和 </prediction> 标签之间。
    例如：<prediction>85.5</prediction>
    """
    
    response = llm_client.chat.completions.create(
        model="deepseek-chat", # 如果使用其他模型，请更改模型名称
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_context}
        ],
        temperature=0.3 # 调低温度，让数值预测更稳定
    )
    
    full_reply = response.choices[0].message.content
    print("\n🤖 大模型回复:\n" + "-"*40)
    print(full_reply)
    print("-" * 40)
    
    return full_reply

def extract_prediction_value(reply_text):
    """从大模型的回复中提取预测数值"""
    match = re.search(r"<prediction>([\d\.]+)</prediction>", reply_text)
    if match:
        return float(match.group(1))
    else:
        raise ValueError("未能在大模型回复中找到 <prediction> 标签，提取失败。")

def write_prediction_to_kg(driver, variety_id, target_phenotype, predicted_value):
    """将大模型预测的结果写回知识图谱"""
    print(f"\n✍️ 正在将预测结果 [{target_phenotype}: {predicted_value}] 写回图谱...")
    
    # 我们加了一个 is_predicted: true 的属性，用来在图谱中区分“真实测定数据”和“AI预测数据”
    query = """
    MATCH (v:Variety {id: $vid})
    MERGE (p:Phenotype {name: $pname})
    MERGE (v)-[r:HAS_PHENOTYPE]->(p)
    SET r.value = $val, r.is_predicted = true
    """
    with driver.session() as session:
        session.run(query, vid=variety_id, pname=target_phenotype, val=predicted_value)
    
    print("✅ 成功！知识图谱已动态更新。")

if __name__ == "__main__":
    target_variety = "GID1"
    target_phenotype = "Drought_Resistance_Index"
    
    # 1. 实例化图谱工具
    rag_tool = WheatGraphRAG(NEO4J_URI, NEO4J_AUTH)
    driver = rag_tool.driver
    
    try:
        # 2. 从图谱抽取特征 Prompt
        print(f"开始提取 {target_variety} 的知识图谱上下文...")
        prompt = rag_tool.extract_features_for_llm(target_variety)
        
        # 3. 将 Prompt 加入具体的任务目标，发给 LLM
        full_prompt = prompt + f"\n\n请预测该品种的 {target_phenotype}。"
        llm_reply = call_llm_for_prediction(full_prompt)
        
        # 4. 解析预测数值
        predicted_val = extract_prediction_value(llm_reply)
        
        # 5. 写回知识图谱
        write_prediction_to_kg(driver, target_variety, target_phenotype, predicted_val)
        
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
    finally:
        rag_tool.close()