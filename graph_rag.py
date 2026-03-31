from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

class WheatGraphRAG:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def extract_features_for_llm(self, variety_id):
        """核心：通过 Cypher 查询提取特征，并组装成 LLM Prompt"""
        
        phenotypes = []
        genotypes = []
        
        with self.driver.session() as session:
            # 1. 查询该品种的所有表型及其数值
            pheno_query = """
            MATCH (v:Variety {id: $vid})-[r:HAS_PHENOTYPE]->(p:Phenotype)
            RETURN p.name AS trait, r.value AS val
            """
            pheno_results = session.run(pheno_query, vid=variety_id)
            for record in pheno_results:
                phenotypes.append(f"- {record['trait']}: {record['val']}")
                
            # 2. 查询该品种的所有 SNP 基因型及其等位基因
            geno_query = """
            MATCH (v:Variety {id: $vid})-[r:HAS_ALLELE]->(s:SNP)
            RETURN s.id AS snp, r.allele AS allele
            """
            geno_results = session.run(geno_query, vid=variety_id)
            for record in geno_results:
                genotypes.append(f"{record['snp']}({record['allele']})")

        # 如果没有找到该品种，直接返回提示
        if not phenotypes and not genotypes:
            return f"未在知识图谱中找到品种 {variety_id} 的相关数据。"

        # 3. 组装发给大模型的 Prompt 上下文
        # 为了防止基因型数据过长（几万个SNP）导致大模型上下文溢出，这里做个截断展示
        # 实际应用中，你可能需要用特定的算法预先筛选出与目标表型最相关的核心 SNP
        max_snps_to_show = 100 
        genotype_str = ", ".join(genotypes[:max_snps_to_show])
        if len(genotypes) > max_snps_to_show:
            genotype_str += f" ... 等共计 {len(genotypes)} 个 SNP 位点。"

        phenotype_str = "\n".join(phenotypes)

        prompt_context = f"""
【背景知识】
你是一个资深的农业大模型，现在请你基于以下小麦品种的知识图谱数据进行分析：

品种名称: {variety_id}

[已知的表型数据]:
{phenotype_str}

[部分的基因型数据 (SNP)]:
{genotype_str}

【任务要求】
基于上述已有的表型与基因特征，结合你的作物遗传学知识，预测该品种在“Drought_Resistance_Index”（抗旱指数）上的表现，并说明你推测的理由。
"""
        return prompt_context

if __name__ == "__main__":
    rag_system = WheatGraphRAG(URI, AUTH)
    
    # 我们以 GID1 为例进行测试
    target_variety = "GID1"
    print(f"正在从知识图谱中抽取 {target_variety} 的特征...\n")
    
    llm_prompt = rag_system.extract_features_for_llm(target_variety)
    
    print("="*50)
    print(">>> 生成的大模型 Prompt 如下：\n")
    print(llm_prompt)
    print("="*50)
    
    rag_system.close()