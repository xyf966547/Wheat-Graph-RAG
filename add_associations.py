from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

def create_snp_phenotype_links():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    # 这个 Cypher 语句会随机选取 10 个 SNP 和 3 个表型，建立直接关联，模拟 GWAS 的显著性结果
    query = """
    MATCH (s:SNP), (p:Phenotype)
    WITH s, p ORDER BY rand() LIMIT 30  // 随机组合产生 30 条显著关联的边
    MERGE (s)-[r:ASSOCIATED_WITH]->(p)
    SET r.weight = round(rand() * 100) / 100.0,  // 模拟相关性权重 (0~1)
        r.p_value = 0.01                         // 模拟显著性
    RETURN count(r) AS links_created
    """
    
    with driver.session() as session:
        result = session.run(query)
        count = result.single()["links_created"]
        print(f"✅ 成功！已在基因型(SNP)和表型(Phenotype)之间建立了 {count} 条直接关联的边。")
        print("💡 注：在实际科研中，你可以将此脚本修改为读取你的 GWAS 结果表格来进行定向连线。")
        
    driver.close()

if __name__ == "__main__":
    create_snp_phenotype_links()