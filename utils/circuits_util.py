import traceback
from pathlib import Path
import re
from collections import defaultdict

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from config import ConfigSingleton
from utils.file.file_util import read_all
args = ConfigSingleton().get_config()
'''
从代码中抽取 qubit nn 依赖关系
适用于 QCIS 格式的指令
'''
def qubits_nn_constrain(circuit_name):
    nn = [set() for _ in range(66)]
    path = Path(args.circuit_path) / circuit_name
    str = read_all(path)
    str =  reassign_qxx_labels(str)
    # 使用正则表达式匹配CZ指令
    pattern = re.compile(r'^CZ\s+(Q\d{1,2})\s+(Q\d{1,2})$')

    # 字典用于统计每个Qxx Qxx组合的出现次数
    counts = defaultdict(int)

    # 遍历每一行
    for line in str.strip().split('\n'):
        match = pattern.match(line.strip())
        if match:
            q1, q2 = match.groups()
            key = f"{q1} {q2}"
            counts[key] += 1
            nn[int(q1[1:])].add(int(q2[1:]))
    # 打印统计结果
    # for key, count in counts.items():
    #     print(f"{key} 出现 {count} 次")
    return nn
    # # 准备数据用于绘制饼状图
    # labels = list(counts.keys())
    # sizes = list(counts.values())


def reassign_qxx_labels(code):
    # 使用正则表达式匹配所有的 Qxx 指令
    qxx_pattern = re.compile(r'Q(\d{1,2})')
    matches = qxx_pattern.findall(code)

    unique_qxx = sorted(set(matches), key=lambda x: int(x))
    qxx_mapping = {old: new for new, old in enumerate(unique_qxx)}

    # 定义替换函数
    def replace_qxx(match):
        old_qxx = match.group(1)
        new_qxx = qxx_mapping[old_qxx]
        return f'Q{new_qxx}'

    # 使用正则表达式替换原代码中的 Qxx 指令
    new_code = qxx_pattern.sub(replace_qxx, code)

    return new_code



simulator = AerSimulator()
def score_layout():
    pass
def count_gates(circuit:QuantumCircuit, initial_layout,coupling_map, gates=['swap'],) -> int:
    try:
        compiled_circuit = transpile(circuits=circuit,
                                     coupling_map=coupling_map,
                                     backend=simulator)

        ops = compiled_circuit.count_ops()
        if len(gates) == 0:
            return  sum(ops.values())
        else:
            return  sum(ops[g] for g in gates if g in ops)
    except Exception as e:
        traceback.print_exc()
        return -1







