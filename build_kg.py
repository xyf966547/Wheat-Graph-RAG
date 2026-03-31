import pandas as pd
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))

class WheatKGBuilder:
    def __init__(self, uri, auth):
        self.driver = GraphDatabase.driver(uri, auth=auth)

    def close(self):
        self.driver.close()

    def init_schema(self):
        """初始化约束，防止节点重复"""
        with self.driver.session() as session:
            # 1. 创建 Variety 和 Phenotype 的约束
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:Variety) REQUIRE v.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Phenotype) REQUIRE p.name IS UNIQUE")
            
            # 2. 安全地创建 SNP 的约束
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:SNP) REQUIRE s.id IS UNIQUE")
            except Exception as e:
                # 如果捕获到索引冲突的报错
                if "IndexAlreadyExists" in str(e):
                    print("⚠️ 检测到旧的 SNP 索引冲突，正在自动修复...")
                    # 查找冲突的索引名称
                    result = session.run("SHOW INDEXES YIELD name, labelsOrTypes, properties WHERE labelsOrTypes = ['SNP'] AND properties = ['id'] RETURN name")
                    for record in result:
                        index_name = record["name"]
                        # 删除旧的普通索引
                        session.run(f"DROP INDEX {index_name}")
                        print(f"已删除冲突索引: {index_name}")
                    
                    # 重新创建唯一性约束
                    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:SNP) REQUIRE s.id IS UNIQUE")
                    print("✅ SNP 唯一性约束重建成功！")
                else:
                    # 如果是其他报错，则抛出
                    raise e

    def create_phenotype_node(self, name):
        """允许独立添加表型节点（即使没有数据）"""
        query = "MERGE (p:Phenotype {name: $name})"
        with self.driver.session() as session:
            session.run(query, name=name)

    def process_phenotype_data(self, df, id_col, source_name):
        """处理表型数据并建立 品种-表型 关系"""
        with self.driver.session() as session:
            for index, row in df.iterrows():
                variety_id = str(row[id_col]).strip()
                
                # 【修改点】分离 MERGE 和 SET，只依赖唯一 ID 进行查找或创建
                session.run("""
                MERGE (v:Variety {id: $id})
                SET v.source = $source
                """, id=variety_id, source=source_name)
                
                # 遍历所有表型列
                for col in df.columns:
                    if col != id_col:
                        val = row[col]
                        if pd.notna(val):
                            # 创建表型节点并建立关系
                            self.create_phenotype_node(col)
                            query = """
                            MATCH (v:Variety {id: $vid})
                            MATCH (p:Phenotype {name: $pname})
                            MERGE (v)-[r:HAS_PHENOTYPE]->(p)
                            SET r.value = $val
                            """
                            session.run(query, vid=variety_id, pname=col, val=float(val))

    def process_genotype_data(self, df, snp_col, id_prefix, source_name):
        """处理基因型数据（宽表转长表），并使用 UNWIND 批量导入防死锁提速"""
        # 提取所有的品种列（以 GID 或 ADT 开头）
        variety_cols = [col for col in df.columns if str(col).startswith(id_prefix)]
        
        # 宽表转长表
        melted_df = pd.melt(df, id_vars=[snp_col], value_vars=variety_cols, 
                            var_name='Variety_ID', value_name='Allele')
        melted_df = melted_df.dropna(subset=['Allele'])

        # --- 1. 批量创建 SNP 节点 ---
        print(f"  [{source_name}] 正在批量创建 SNP 节点...")
        unique_snps = [{'id': str(snp).strip()} for snp in melted_df[snp_col].unique()]
        with self.driver.session() as session:
            # 使用 UNWIND 语法一次性处理所有 SNP 节点
            session.run("""
            UNWIND $snps AS snp
            MERGE (s:SNP {id: snp.id})
            """, snps=unique_snps)

        # --- 2. 分批次建立 品种-SNP 关系 ---
        print(f"  [{source_name}] 正在批量建立 品种-SNP 关系 (共 {len(melted_df)} 条记录)...")
        # 将 DataFrame 转换为字典列表，方便传给 Neo4j
        records = melted_df.rename(columns={snp_col: 'SNP_ID'}).to_dict('records')
        
        # 定义一个批量执行的事务函数
        def batch_create_rels(tx, batch_data):
            query = """
            UNWIND $batch AS row
            MATCH (v:Variety {id: trim(toString(row.Variety_ID))})
            MATCH (s:SNP {id: trim(toString(row.SNP_ID))})
            MERGE (v)-[r:HAS_ALLELE]->(s)
            SET r.allele = trim(toString(row.Allele))
            """
            tx.run(query, batch=batch_data)

        # 每批处理 5000 条数据（可根据电脑内存调整）
        batch_size = 20000
        with self.driver.session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                # execute_write 会自动处理事务提交，如果遇到瞬时死锁它会自动重试
                session.execute_write(batch_create_rels, batch)
                
                # 打印进度条
                current_count = min(i + batch_size, len(records))
                print(f"    进度: {current_count} / {len(records)} ...")

if __name__ == "__main__":
    builder = WheatKGBuilder(URI, AUTH)
    builder.init_schema()
    print("数据库 Schema 初始化完成。")

    # --- 处理 Set 1 数据 ---
    print("开始处理 Set 1 表型数据...")
    df_pheno1 = pd.read_csv("data/set1/Phenotypic_data.txt", sep='\t')
    builder.process_phenotype_data(df_pheno1, id_col="ID", source_name="Set1_GID")

    print("开始处理 Set 1 基因型数据...")
    df_geno1 = pd.read_csv("data/set1/Genotypic_data.csv")
    builder.process_genotype_data(df_geno1, snp_col="rs#", id_prefix="GID", source_name="Set1_GID")

    # --- 处理 Set 2 数据 ---
    print("开始处理 Set 2 表型数据...")
    df_pheno2 = pd.read_csv("data/set2/Phenotypic_Data_282.csv")
    builder.process_phenotype_data(df_pheno2, id_col="Genotype", source_name="Set2_ADT")

    print("开始处理 Set 2 基因型数据...")
    df_geno2 = pd.read_csv("data/set2/SNP_ARRAY_282.csv")
    builder.process_genotype_data(df_geno2, snp_col="rs", id_prefix="ADT", source_name="Set2_ADT")

    # --- 演示：添加一个完全没有数据的表型 ---
    print("添加未知的表型节点...")
    builder.create_phenotype_node("Drought_Resistance_Index") 

    builder.close()
    print("全部数据导入完毕！")