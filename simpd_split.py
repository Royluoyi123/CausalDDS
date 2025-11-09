import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.cluster import KMeans
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载和预处理数据"""
    # 读取药物协同数据
    synergy_df = pd.read_csv('DDS/ddsdata/DrugCombDB640/drug_synergy.csv')
    
    # 读取药物SMILES数据
    drug_smiles_df = pd.read_csv('DDS/ddsdata/DrugCombDB640/drug_smiles.csv')
    #drug_to_smiles = dict(zip(drug_smiles_df['pubchemid'], drug_smiles_df['isosmiles']))
    drug_to_smiles = dict(zip(drug_smiles_df['Drug_name'], drug_smiles_df['Smiles']))
    
    return synergy_df, drug_to_smiles

def get_drug_fingerprints(drug_to_smiles):
    """计算药物的分子指纹"""
    drug_fps = {}
    drug_mols = {}
    
    for drug, smiles in tqdm(drug_to_smiles.items(), desc="计算分子指纹"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                drug_fps[drug] = fp
                drug_mols[drug] = mol
        except:
            continue
    
    return drug_fps, drug_mols

def cluster_drugs(drug_fps, n_clusters=5):
    """对药物进行聚类"""
    # 将指纹转换为数组
    drug_names = list(drug_fps.keys())
    fp_array = np.array([np.array(fp) for fp in drug_fps.values()])
    
    # 使用K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(fp_array)
    
    drug_clusters = dict(zip(drug_names, clusters))
    return drug_clusters, kmeans

def calculate_drug_complexity(drug_mols):
    """计算药物复杂度分数"""
    complexity_scores = {}
    
    for drug, mol in drug_mols.items():
        # 使用简单的分子描述符作为复杂度代理
        # 可以包括分子量、重原子数、环数等
        mol_weight = Descriptors.MolWt(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        rings = Descriptors.RingCount(mol)
        
        # 综合复杂度分数
        complexity = mol_weight * 0.4 + heavy_atoms * 0.3 + rings * 0.3
        complexity_scores[drug] = complexity
    
    return complexity_scores

def simp_kfold_split(synergy_df, drug_clusters, drug_fps, complexity_scores, n_splits=5):
    """执行五折交叉验证的 SIMPD 风格分割"""
    
    drug_pairs = []
    features = []
    labels = []
    
    for idx, row in tqdm(synergy_df.iterrows(), total=len(synergy_df), desc="处理药物对"):
        drug_a = row['drug_a']
        drug_b = row['drug_b']
        
        if drug_a in drug_fps and drug_b in drug_fps:
            fp_a = np.array(drug_fps[drug_a])
            fp_b = np.array(drug_fps[drug_b])
            combined_fp = np.concatenate([fp_a, fp_b])
            
            cluster_a = drug_clusters.get(drug_a, -1)
            cluster_b = drug_clusters.get(drug_b, -1)
            complexity_a = complexity_scores.get(drug_a, 0)
            complexity_b = complexity_scores.get(drug_b, 0)
            
            additional_features = [
                cluster_a, cluster_b, 
                complexity_a, complexity_b,
                abs(complexity_a - complexity_b)
            ]
            
            full_features = np.concatenate([combined_fp, additional_features])
            
            drug_pairs.append((drug_a, drug_b, row['cell_line']))
            features.append(full_features)
            labels.append(row['synergy'])
    
    features = np.array(features)
    labels = np.array(labels)
    
    # 计算复杂度分箱，用于分层
    pair_complexities = []
    for drug_a, drug_b, _ in drug_pairs:
        comp_a = complexity_scores.get(drug_a, 0)
        comp_b = complexity_scores.get(drug_b, 0)
        pair_complexities.append((comp_a + comp_b) / 2)
    
    complexity_bins = pd.qcut(pair_complexities, q=5, labels=False, duplicates='drop')
    
    # 五折交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    folds = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(features, complexity_bins)):
        train_data = synergy_df.iloc[train_idx].copy()
        test_data = synergy_df.iloc[test_idx].copy()
        
        folds.append((train_data, test_data))
        
        # 也可以在这里保存文件
        train_data.to_csv(f'DDS/ddsdata/DrugCombDB640/raw/regression/SIMPD/train_{fold_idx+1}.csv', index=False)
        test_data.to_csv(f'DDS/ddsdata/DrugCombDB640/raw/regression/SIMPD/test_{fold_idx+1}.csv', index=False)
        
        print(f"Fold {fold_idx+1}: train={len(train_data)}, test={len(test_data)}")
    
    return folds

def analyze_split_results(train_data, test_data, drug_clusters, complexity_scores):
    """分析分割结果"""
    print("=== 分割结果分析 ===")
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    print(f"分割比例: {len(test_data)/len(train_data):.2%}")
    
    # 分析药物分布
    train_drugs = set(train_data['drug_a']).union(set(train_data['drug_b']))
    test_drugs = set(test_data['drug_a']).union(set(test_data['drug_b']))
    
    print(f"\n训练集唯一药物数: {len(train_drugs)}")
    print(f"测试集唯一药物数: {len(test_drugs)}")
    print(f"重叠药物数: {len(train_drugs.intersection(test_drugs))}")
    
    # 分析复杂度分布
    train_complexities = []
    for _, row in train_data.iterrows():
        comp_a = complexity_scores.get(row['drug_a'], 0)
        comp_b = complexity_scores.get(row['drug_b'], 0)
        train_complexities.append((comp_a + comp_b) / 2)
    
    test_complexities = []
    for _, row in test_data.iterrows():
        comp_a = complexity_scores.get(row['drug_a'], 0)
        comp_b = complexity_scores.get(row['drug_b'], 0)
        test_complexities.append((comp_a + comp_b) / 2)
    
    print(f"\n训练集平均复杂度: {np.mean(train_complexities):.2f}")
    print(f"测试集平均复杂度: {np.mean(test_complexities):.2f}")
    
    # 分析聚类分布
    train_clusters = []
    for drug in train_drugs:
        if drug in drug_clusters:
            train_clusters.append(drug_clusters[drug])
    
    test_clusters = []
    for drug in test_drugs:
        if drug in drug_clusters:
            test_clusters.append(drug_clusters[drug])
    
    print(f"\n训练集聚类分布: {pd.Series(train_clusters).value_counts().to_dict()}")
    print(f"测试集聚类分布: {pd.Series(test_clusters).value_counts().to_dict()}")

def main():
    """主函数"""

    print("开始SIMPD风格的五折交叉验证分割...")
    
    synergy_df, drug_to_smiles = load_and_preprocess_data()
    drug_fps, drug_mols = get_drug_fingerprints(drug_to_smiles)
    drug_clusters, kmeans = cluster_drugs(drug_fps, n_clusters=5)
    complexity_scores = calculate_drug_complexity(drug_mols)
    
    folds = simp_kfold_split(synergy_df, drug_clusters, drug_fps, complexity_scores, n_splits=5)
    
    return folds


if __name__ == "__main__":
    folds = main()