import traceback
from pathlib import Path
import re
from collections import defaultdict

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from config import ConfigSingleton
from core import chip
from utils.code_conversion import QCIS_2_QASM
from utils.file.file_util import read_all
from utils.points_util import coordinate2adjacent
simulator = AerSimulator()
args = ConfigSingleton().get_config()
'''
从代码中抽取 qubit nn 依赖关系
适用于 QCIS 格式的指令
return  [{1, 2}, {3}, {3, 4}, set(),set()]
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
    print('reassign_qxx_labels')
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

'''
* virtual to physical::
[0, 3, 5]  # virtual qubits are ordered (in addition to named)
'''
def swap_counts(circuit,initial_layout):
    return count_gates(circuit,initial_layout,coupling_map=chip.COUPLING_MAP)

def count_gates(circuit:QuantumCircuit, layout,coupling_map) -> int:
    initial_layout = {}
    for i, v in enumerate(layout):
        initial_layout[circuit.qubits[i]] = v
    #print(initial_layout)
    try:
        compiled_circuit = transpile(circuits=circuit,
                                     initial_layout=initial_layout,
                                     coupling_map=coupling_map,
                                     optimization_level=0,
                                     backend=simulator)
        ops = compiled_circuit.count_ops()
        if 'swap' in ops.keys():
            return ops['swap']
        else:
            return 0
    except Exception as e:
        traceback.print_exc()
        return -1


def compare_depth(circuit:QuantumCircuit,layout):
    ################################################################3
    path = Path(args.circuit_path) / 'XEB_9_qubits_8_cycles_circuit.txt'
    RE_LABEL_CIRCUIT = reassign_qxx_labels(read_all(path))
    QASM_STR = QCIS_2_QASM(RE_LABEL_CIRCUIT)
    simulator = AerSimulator()
    circuit = QuantumCircuit.from_qasm_str(qasm_str=QASM_STR)
    # circuit.draw('mpl').show()
    layout = [16,24,18,26,32,25,30,31,17]
    initial_layout = {}
    for i, v in enumerate(layout):
        initial_layout[circuit.qubits[i]] = v
    points = [(x, y) for x in range(7) for y in range(7)]
    adjacency_list = coordinate2adjacent(points)
    ct1 = transpile(circuits=circuit, coupling_map=adjacency_list, initial_layout=initial_layout, optimization_level=1,
                    backend=simulator)
    ct2 = transpile(circuits=circuit, coupling_map=adjacency_list, optimization_level=1, backend=simulator)
    d1 = ct1.depth()
    d2 = ct2.depth()

    print(f'd1={d1},d2={d2}')

if __name__ == '__main__':
    path = Path(args.circuit_path) / 'XEB_5_qubits_8_cycles_circuit.txt'
    RE_LABEL_CIRCUIT = reassign_qxx_labels(read_all(path))
    QASM_STR = QCIS_2_QASM(RE_LABEL_CIRCUIT)
    circuit = QuantumCircuit.from_qasm_str(qasm_str=QASM_STR)
    print(swap_counts(circuit,[27,34,33,38,39]))
    #print(qubits_nn_constrain('XEB_5_qubits_8_cycles_circuit.txt'))



