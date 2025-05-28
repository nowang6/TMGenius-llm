# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import glob
import re

import json
import pandas as pd
import torch

from c2lb import lb_and_intra_layer_affinity_redundancy_deploy
from c2lb_dynamic import lb_redundancy_deploy_for_dynamic
from speculative_moe import speculative_moe_algo_multi_process
from speculative_moe_a3 import speculative_moe_algo_multi_process_a3
from c2lb_a3 import lb_and_intra_layer_affinity_redundancy_deploy_a3
from components.utils.log import logger
from components.utils.file_open_check import ms_open
from components.expert_load_balancing.elb.constant import PREFILL, DECODE, DECODE_FILE_NAME, \
                        PREFILL_FILE_NAME, ALGORITHM_C2LB, ALGORITHM_SPECULATIVE_MOE, ALGORITHM_DYNAMIC_C2LB, A2, A3


def load_expert_popularity_csv(csv_file_path):
    """
    从指定目录加载 decode_info.csv 和 prefill_info.csv 文件，支持以下场景：
    - 仅存在其中一个文件。
    - 同时存在两个文件。
    
    参数:
    - csv_file_path (str): 包含 CSV 文件的目录路径。
    
    返回:
    - (decode_df, prefill_df): 返回元组。
        - decode_df: decode_info.csv 的内容，若文件不存在或读取失败则为 None。
        - prefill_df: prefill_info.csv 的内容，若文件不存在或读取失败则为 None。
    """
    decode_info = None
    prefill_info = None

    for file_name in ["decode_info.csv", "prefill_info.csv"]:
        file_path = os.path.join(csv_file_path, file_name)
        
        if not os.path.isfile(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {file_name}.")
            
            if file_name == "decode_info.csv":
                decode_info = df
            elif file_name == "prefill_info.csv":
                prefill_info = df
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"decode_info or prefill_info is empty: {file_path}") from e
        except pd.errors.ParserError as e:
            raise ValueError(f"Failed to parse CSV: {file_path}") from e
        except Exception as e:
            raise RuntimeError(f"Error reading {file_path}: {str(e)}") from e

    is_decode_missing = decode_info is None or decode_info.empty
    is_prefill_missing = prefill_info is None or prefill_info.empty
    if is_decode_missing and is_prefill_missing:
        raise FileNotFoundError(f"No decode_info.csv or prefill_info.csv found in {csv_file_path}."
                                f"Check that the input file is correct to generate the files.")
    
    return decode_info, prefill_info


def get_csv_path(csv_file_path, file_name):
    """
    从指定目录得到decode_info.csv和prefill_info.csv文件路径。
    
    参数:
    - csv_file_path: 包含CSV文件的目录路径。
    
    返回:
    - decode_df: decode_info.csv 的内容，若文件不存在或读取失败则为 None。
    - prefill_df: prefill_info.csv 的内容，若文件不存在或读取失败则为 None。
    """
    file_path = os.path.join(csv_file_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist."
                                f"Check whether the output path in the input parameter -o is correct.")

    return file_path


def merge_csv_columns(csv_path, pattern_prefix):
    """
    合并指定目录下以特定前缀开头的所有csv文件的列。

    参数:
    - csv_path: 文件所在的目录路径。
    - pattern_prefix: 文件名前缀。
    
    返回:
    - 生成的DataFrame对象。
    """
    file_pattern = os.path.join(csv_path, f"{pattern_prefix}_*.csv")
    
    def sort_filenames(filenames):
        return sorted(filenames, key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
    
    files = glob.glob(file_pattern)
    filtered_files = [f for f in files if re.match(rf".*{pattern_prefix}_(\d+)\.csv", f)]
    if not filtered_files:
        raise FileNotFoundError(f"No files found matching the pattern {file_pattern}.")

    filtered_files = sort_filenames(filtered_files)

    logger.debug(f"{pattern_prefix} file number is {len(filtered_files)}")
    all_columns = []
    column_count = 0
    
    for file in filtered_files:
        logger.debug(f"Processing file: {file}.")
        df = pd.read_csv(file, header=None)
        if df.empty:
            raise ValueError(f"File {file} is empty. Check the input decode or prefill file.")
        
        for col in range(df.shape[1]):
            all_columns.append(df.iloc[:, col].rename(f'expert_{column_count}'))
            column_count += 1
    
    if not all_columns:
        raise ValueError("No columns to concatenate. Check whether the ipnut csv path is correct."
                         "Or whether the csv file in the folder is valid.")
    
    final_df = pd.concat(all_columns, axis=1)
    return final_df


def save_matrix_to_json(output_path, file_name, deployment):
    def get_ndim(obj):
        if isinstance(obj, list):
            return 1 + (get_ndim(obj[0]) if obj else 0)
        return 0
    
    cur_dim = get_ndim(deployment)
    if cur_dim != 3:
        raise ValueError(f"部署矩阵必须是三维数组，但当前维度为 {cur_dim}D。")
    num_layers = len(deployment)
    num_cards = len(deployment[0])

    data = {"moe_layer_count": num_layers}
    layer_list = []
    for i in range(num_layers):
        layer = {"layer_id": i, "device_count": num_cards}
        device_list = []
        for j in range(num_cards):
            device = {"device_id": j, "device_expert": list(deployment[i][j])}
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)
    data["layer_list"] = layer_list

    file_name = os.path.join(output_path, f"{file_name}.json")

    # 保存为 JSON 文件
    try:
        with ms_open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"保存json文件 {deployment} 时出错") from e


def dump_tables(deploy_fp, d2e_tables_list, n_devices):
    layer_list = []

    for layer_idx, d2e_tables in enumerate(d2e_tables_list):
        device_list = [
            {"device_id": d, "device_expert": d2e_tables[layer_idx][d].tolist()} 
            for d in range(n_devices)
        ]
        layer_list.append({
            "layer_id": layer_idx,
            "device_count": n_devices,
            "device_list": device_list
        })
    json_data = {
        "moe_layer_count": len(layer_list),
        "layer_list": layer_list
    }
    logger.debug(json_data)
    with ms_open(deploy_fp, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def save_matrix_to_json_a3(output_path, file_name, deployment):
    if deployment.ndim < 3:
        raise ValueError(f"输出文件的实际维度有误: {deployment.ndim}, 需要3维数据")
    num_layers = len(deployment)
    num_cards = len(deployment[0])

    data = {"moe_layer_count": num_layers}
    layer_list = []
    for i in range(num_layers):
        layer = {"layer_id": i, "device_count": num_cards}
        device_list = []
        for j in range(num_cards):
            # 将 1*4 的行矩阵转换为列表
            device = {"device_id": j, "device_expert": deployment[i][j].tolist()}
            device_list.append(device)
        layer["device_list"] = device_list
        layer_list.append(layer)
    data["layer_list"] = layer_list

    file_name = os.path.join(output_path, f"{file_name}.json")
    try:
        with ms_open(file_name, 'w') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"写入文件 {deployment} 时出错: {e}") from e


def dump_tables_a3(deploy_fp, d2e_tables_obj_list, n_devices, n_share_expert_devices):
    mean_obj = 0
    for layer_idx, d2e_tables_obj in enumerate(d2e_tables_obj_list):
        mean_obj += d2e_tables_obj[1][layer_idx]
    layer_list = []
    for layer_idx, d2e_tables_obj in enumerate(d2e_tables_obj_list):
        d2e_tables = d2e_tables_obj[0]
        device_list = [
            {"device_id": d, "device_expert": [0]} 
            for d in range(n_share_expert_devices)
        ]
        device_list.extend([
            {"device_id": d + n_share_expert_devices, "device_expert": d2e_tables[layer_idx][d].tolist()}
            for d in range(n_devices)
        ])
        layer_list.append({
            "layer_id": layer_idx,
            "device_count": n_devices + n_share_expert_devices,
            "device_list": device_list
        })
    json_data = {
        "moe_layer_count": len(layer_list),
        "layer_list": layer_list
    }
    with ms_open(deploy_fp, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


def save_dataframes(prefill_final_df, decode_final_df, output_dir):
    """Save prefill and decode dataframes to CSV files if they are not empty."""
    if prefill_final_df is not None and not prefill_final_df.empty:
        prefill_final_df.to_csv(os.path.join(output_dir, 'prefill_info.csv'), index=False)
    if decode_final_df is not None and not decode_final_df.empty:
        decode_final_df.to_csv(os.path.join(output_dir, 'decode_info.csv'), index=False)


def get_csv_dimensions(input_csv_path):
    """
    读取CSV文件并返回其行数（第0维）和列数（第1维）。

    参数:
        input_csv_path (str): CSV文件路径。

    返回:
        tuple: (行数, 列数) 的元组，失败时返回 (None, None)。
    """
    dimensions = (None, None)
    try:
        df = pd.read_csv(input_csv_path)
        if df.empty:
            raise ValueError(f"文件包含空数据 {input_csv_path}")
        row_count = df.shape[0]  # 第0维（行数）58
        col_count = df.shape[1]  # 第1维（列数）256
        dimensions = (row_count, col_count)
        
        logger.info(f"Successfully read: {input_csv_path}.")
        logger.info(f"The input file: Number of row={row_count}, Number of columns={col_count}.")
        
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"文件无有效数据 {input_csv_path}") from e
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV格式解析失败 {input_csv_path}") from e
    except Exception as e:
        raise RuntimeError(f"未知错误") from e
    
    return dimensions


def process_c2lb(args, output_dir):
    """Process C2LB algorithm and save results based on available data files.
    
    支持场景:
    - 只有 decode_info.csv 时: 处理 decode 数据。
    - 只有 prefill_info.csv 时: 处理 prefill 数据。
    - 两者都存在时: 同时处理 decode 和 prefill 数据。
    
    参数:
    - args: 包含算法参数的命名空间。
    - output_dir: 输出结果的目录路径。
    """
    logger.info("Generating Files with the C2lb Algorithm for Static Scenarios")
    decode_df, prefill_df = load_expert_popularity_csv(output_dir)

    if decode_df is not None:
        try:
            num_original_expert = decode_df.shape[1] 
            decode_np = decode_df.to_numpy()
            global_deployment = lb_and_intra_layer_affinity_redundancy_deploy(
                decode_np, 
                args.num_redundancy_expert, 
                args.num_npus, 
                num_original_expert, 
            )
            save_matrix_to_json(output_dir, DECODE_FILE_NAME, global_deployment)
            logger.info(f"C2LB processed decode data -> {DECODE_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process decode data: {str(e)}") from e

    if prefill_df is not None:
        try:
            num_original_expert = prefill_df.shape[1] 
            prefill_np = prefill_df.to_numpy()
            global_deployment = lb_and_intra_layer_affinity_redundancy_deploy(
                prefill_np,  
                args.num_redundancy_expert, 
                args.num_npus, 
                num_original_expert,
            )
            save_matrix_to_json(output_dir, PREFILL_FILE_NAME, global_deployment)
            logger.info(f"C2LB processed prefill data -> {PREFILL_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process prefill data: {str(e)}") from e
    
    if decode_df is None and prefill_df is None:
        raise FileNotFoundError("No valid decode/prefill data found in", output_dir)
    

def process_speculative_moe(args, file_names, output_dir):
    """
    处理 Speculative MOE 算法，根据传入的 file_names 动态处理以下场景：
    - 只有 decode_info.csv 时: 处理 decode 数据。
    - 只有 prefill_info.csv 时: 处理 prefill 数据。
    - 两者都存在时: 同时处理 decode 和 prefill 数据。

    参数:
        args (Namespace): 包含算法参数的对象。
        file_names (list): 需要处理的 CSV 文件名列表（例如 ["decode_info.csv", "prefill_info.csv"]）。
        output_dir (str): 输出结果的目录路径。
    """
    logger.info("Generating Files with the Speculative Moe Algorithm for Static Scenarios")
    input_output_map = {
        "decode_info.csv": "decode_global_deployment.json",
        "prefill_info.csv": "prefill_global_deployment.json"
    }   

    for input_file in file_names:
        if input_file not in input_output_map:
            logger.warning(f"忽略不支持的文件 {input_file}")
            continue

        input_csv_path = get_csv_path(output_dir, input_file)
        dimensions = get_csv_dimensions(input_csv_path)
        output_file_name = input_output_map[input_file]
        output_path = os.path.join(output_dir, output_file_name)
        num_layer = dimensions[0]
        num_original_expert = dimensions[1]
        try:
            results = speculative_moe_algo_multi_process(
                args.num_npus,
                args.num_nodes,
                num_layer,
                num_original_expert,
                args.num_redundancy_expert,
                input_csv_path,
            )
            dump_tables(output_path, results, args.num_npus)
            logger.info(f"Speculative MOE processing is complete: {input_file} -> {output_file_name}.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The input file does not exist {input_csv_path}.") from e
        except Exception as e:
            raise FileNotFoundError(f"An exception occurred while processing {input_file}: {str(e)}.") from e


def process_dynamic_c2lb(args, output_dir):
    """Process dynamic C2LB algorithm and save results based on available data files.
    
    支持场景:
    - 只有 decode_info.csv 时: 处理 decode 数据。
    - 只有 prefill_info.csv 时: 处理 prefill 数据。
    - 两者都存在时: 同时处理 decode 和 prefill 数据。
    
    参数:
    - args: 包含算法参数的命名空间。
    - output_dir: 输出结果的目录路径。
    """
    logger.info("Generating Initialization Files with the C2lb Algorithm for Dynamic Scenarios")
    decode_df, prefill_df = load_expert_popularity_csv(output_dir)

    if decode_df is not None:
        try:
            decode_np = decode_df.to_numpy()
            global_deployment = lb_redundancy_deploy_for_dynamic(
                decode_np, 
                args.num_redundancy_expert, 
                args.num_nodes, 
                args.num_npus
            )
            save_matrix_to_json(output_dir, DECODE_FILE_NAME, global_deployment)
            logger.info(f"dynamic C2LB processed decode data -> {DECODE_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process decode data: {str(e)}") from e

    if prefill_df is not None:
        try:
            prefill_np = prefill_df.to_numpy()
            global_deployment = lb_redundancy_deploy_for_dynamic(
                prefill_np,  
                args.num_redundancy_expert, 
                args.num_nodes, 
                args.num_npus
            )
            save_matrix_to_json(output_dir, PREFILL_FILE_NAME, global_deployment)
            logger.info(f"dynamic C2LB processed prefill data -> {PREFILL_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process decode data: {str(e)}") from e

    if decode_df is None and prefill_df is None:
        raise FileNotFoundError(f"No valid decode/prefill data found in {output_dir}")


def process_speculative_moe_a3(args, file_names, output_dir):
    """
    处理 Speculative MOE A3算法，根据传入的 file_names 动态处理以下场景：
    - 只有 decode_info.csv 时: 处理 decode 数据。
    - 只有 prefill_info.csv 时: 处理 prefill 数据。
    - 两者都存在时: 同时处理 decode 和 prefill 数据。

    参数:
        args (Namespace): 包含算法参数的对象。
        file_names (list): 需要处理的 CSV 文件名列表（例如 ["decode_info.csv", "prefill_info.csv"]）。
        output_dir (str): 输出结果的目录路径。
    """
    logger.info("Generating Files with the Speculative Moe A3 Algorithm for Static Scenarios")
    input_output_map = {
        "decode_info.csv": "decode_global_deployment.json",
        "prefill_info.csv": "prefill_global_deployment.json"
    }   

    for input_file in file_names:
        if input_file not in input_output_map:
            logger.warning(f"Ignore unsupported files {input_file}.")
            continue

        input_csv_path = get_csv_path(output_dir, input_file)
        dimensions = get_csv_dimensions(input_csv_path)
        output_file_name = input_output_map[input_file]
        output_path = os.path.join(output_dir, output_file_name)
        num_layer = dimensions[0]
        num_original_expert = dimensions[1]
        try:
            results = speculative_moe_algo_multi_process_a3(
                args.num_npus,
                args.num_nodes,
                num_layer,
                num_original_expert,
                args.num_redundancy_expert,
                input_csv_path,
            )
            dump_tables_a3(output_path, results, args.num_npus, args.share_expert_devices)
            logger.info(f"Speculative MOE A3 processing is complete: {input_file} -> {output_file_name}.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The input file does not exist {input_csv_path}.") from e
        except Exception as e:
            raise FileNotFoundError(f"An exception occurred while processing {input_file}: {str(e)}.") from e


def process_c2lb_a3(args, output_dir):
    """Process C2LB A3 algorithm and save results based on available data files.
    
    支持场景:
    - 只有 decode_info.csv 时: 处理 decode 数据。
    - 只有 prefill_info.csv 时: 处理 prefill 数据。
    - 两者都存在时: 同时处理 decode 和 prefill 数据。
    
    参数:
    - args: 包含算法参数的命名空间。
    - output_dir: 输出结果的目录路径。
    """
    logger.info("Generating Files with the C2lb A3 Algorithm for Static Scenarios")
    decode_df, prefill_df = load_expert_popularity_csv(output_dir)

    if decode_df is not None:
        try:
            num_original_expert = decode_df.shape[1] 
            decode_np = decode_df.to_numpy()
            global_deployment = lb_and_intra_layer_affinity_redundancy_deploy_a3(
                decode_np, 
                args.num_redundancy_expert, 
                args.num_npus, 
                num_original_expert, 
            )
            save_matrix_to_json_a3(output_dir, DECODE_FILE_NAME, global_deployment)
            logger.info(f"C2LB A3 processed decode data -> {DECODE_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process decode data: {str(e)}") from e

    if prefill_df is not None:
        try:
            num_original_expert = prefill_df.shape[1] 
            prefill_np = prefill_df.to_numpy()
            global_deployment = lb_and_intra_layer_affinity_redundancy_deploy_a3(
                prefill_np,  
                args.num_redundancy_expert, 
                args.num_npus, 
                num_original_expert,
            )
            save_matrix_to_json_a3(output_dir, PREFILL_FILE_NAME, global_deployment)
            logger.info(f"C2LB A3 processed prefill data -> {PREFILL_FILE_NAME}.json")
        except Exception as e:
            raise RuntimeError(f"Failed to process prefill data: {str(e)}") from e
    
    if decode_df is None and prefill_df is None:
        raise FileNotFoundError("No valid decode/prefill data found in", output_dir)


def check_file_type(folder_path):
    """
    Check the folder for files containing 'prefill' or 'decode' in their names.
    Returns a tuple of:
    - detected file types ('prefill', 'decode', or both).
    - number of 'prefill' files.
    - number of 'decode' files.
    """
    file_types = set()
    prefill_count = 0
    decode_count = 0
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".csv"):
            if PREFILL in file_name.lower():
                file_types.add(PREFILL)
                prefill_count += 1
            if DECODE in file_name.lower():
                file_types.add(DECODE)
                decode_count += 1
    
    return file_types, prefill_count, decode_count


def select_algorithm(args, file_names):
    # 静态场景下 A2 c2lb 算法
    if args.algorithm == ALGORITHM_C2LB and args.device_type == A2:
        process_c2lb(args, output_dir=args.output_dir)
    # 静态场景下 A2 speculative_moe 算法
    elif args.algorithm == ALGORITHM_SPECULATIVE_MOE and args.device_type == A2:
        process_speculative_moe(args, file_names=file_names, output_dir=args.output_dir)
    # 动态场景下 A2 c2lb 算法
    elif args.algorithm == ALGORITHM_DYNAMIC_C2LB and args.device_type == A2:
        process_dynamic_c2lb(args, output_dir=args.output_dir)
    # 静态场景下 A3 c2lb 算法
    elif args.algorithm == ALGORITHM_C2LB and args.device_type == A3:
        if args.share_expert_devices == 0:
            process_c2lb_a3(args, output_dir=args.output_dir)
        else:
            raise ValueError("Input incorrect share expert devices parameters.")
    # 静态场景下 A3 speculative_moe 算法
    elif args.algorithm == ALGORITHM_SPECULATIVE_MOE and args.device_type == A3:
        process_speculative_moe_a3(args, file_names=file_names, output_dir=args.output_dir)
    else:
        raise ValueError("Please enter valid parameters")


def load_balancing(args):
    """
    Main function to perform load balancing based on the specified algorithm.
    """
    # Check the file types in the input path
    file_types, prefill_count, decode_count = check_file_type(args.expert_popularity_csv_load_path)

    # Print the file types and their count
    logger.info(f"Detected file types: {', '.join(file_types)}")
    logger.info(f"Total 'prefill' files: {prefill_count}")
    logger.info(f"Total 'decode' files: {decode_count}")

    # Merge CSV columns for prefill and decode
    prefill_final_df = None   
    decode_final_df = None
    if PREFILL in file_types:
        prefill_final_df = merge_csv_columns(args.expert_popularity_csv_load_path, PREFILL)
    if DECODE in file_types:
        decode_final_df = merge_csv_columns(args.expert_popularity_csv_load_path, DECODE)
    
     # Save merged dataframes to CSV files if they are not empty
    save_dataframes(prefill_final_df, decode_final_df, args.output_dir)

    # Define file names for further processing
    file_names = []
    if PREFILL in file_types:
        file_names.append("prefill_info.csv")
    if DECODE in file_types:
        file_names.append("decode_info.csv")

    select_algorithm(args, file_names)