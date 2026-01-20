import os
import json
import pandas as pd
from tqdm import tqdm
from tree_sitter import Language, Parser
from DFG import DFG_solidity
from utils import (
    tree_to_token_index,
    index_to_code_token
)

PRECOMPILED_LIB_PATH = './my-languages.so' 


INPUT_JSONL_FILE = './dataset.jsonl'
OUTPUT_CSV_FILE = './dfg.csv'


def setup_parser_from_precompiled(lib_path):
    """
    直接从一个预编译好的库文件中加载语言并返回解析器。
    """
    if not os.path.exists(lib_path):
        print(f"致命错误: 预编译的语言库文件未找到: {lib_path}")
        print("请确保 SourceP 项目中的 'my-languages.so' 文件与此脚本位于同一目录。")
        exit() 

   
    solidity_lang = Language(lib_path, 'solidity')
    parser = Parser()
    parser.set_language(solidity_lang)
    print(f"成功从 {lib_path} 加载 Solidity 语言解析器。")
    return parser


def main():
    """主执行函数"""
    print("--- 步骤 1: 设置 Tree-sitter 解析器 (从预编译库加载) ---")
   
    parser = setup_parser_from_precompiled(PRECOMPILED_LIB_PATH)

    print(f"--- 步骤 2: 从 {INPUT_JSONL_FILE} 读取源代码并构建 DFG ---")
    all_dfg_data = []

    try:
        with open(INPUT_JSONL_FILE, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="处理合约中"):
                try:
                    data = json.loads(line)
                    address = data.get('address')
                    source_code = data.get('source_code')

                    if not address or not source_code:
                        continue

                    tree = parser.parse(bytes(source_code, "utf8"))
                    root_node = tree.rootNode

                    tokens_index = tree_to_token_index(root_node)
                    code = source_code.split('\n')
                    index_to_code = {index: (idx, code_token) for idx, (index, code_token) in
                                     enumerate(index_to_code_token(tokens_index, code))}

                    dfg_raw, _ = DFG_solidity(root_node, index_to_code, {})

                    nodes = []
                    for item in dfg_raw:
                        nodes.append(item[0])
                        nodes.extend(item[3])

                    unique_nodes = sorted(list(set(nodes)))
                    node_to_id = {node: i for i, node in enumerate(unique_nodes)}

                    edges = []
                    for item in dfg_raw:
                        target_node = item[0]
                        source_nodes = item[3]

                        if target_node not in node_to_id: continue

                        target_id = node_to_id[target_node]
                        for source_node in source_nodes:
                            if source_node in node_to_id:
                                source_id = node_to_id[source_node]
                                if source_id != target_id:
                                    edges.append([source_id, target_id])

                    all_dfg_data.append({
                        'address': address,
                        'node_feature': json.dumps(unique_nodes),
                        'edge_index': json.dumps(edges)
                    })

                except Exception as e:
                    print(f"\n处理合约 {address} 时发生错误: {e}")
                    continue

    except FileNotFoundError:
        print(f"错误: 输入文件 {INPUT_JSONL_FILE} 未找到!")
        return

    print(f"--- 步骤 3: 将提取的 DFG 数据保存到 {OUTPUT_CSV_FILE} ---")
    if not all_dfg_data:
        print("警告: 未能成功提取任何 DFG 数据。")
        return

    df_out = pd.DataFrame(all_dfg_data)
    df_out = df_out[['address', 'node_feature', 'edge_index']]
    df_out.to_csv(OUTPUT_CSV_FILE, index=False)
    print("DFG 数据提取并保存成功！")


if __name__ == '__main__':
    main()