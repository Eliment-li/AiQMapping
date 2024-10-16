# QCIS 转换为 QASM
from cqlib.utils import QcisToQasm, QasmToQcis
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
simulator = AerSimulator()

from utils.file.file_util import read_all


def QCIS_2_QASM(qcis_circuit):
    qasm_str = QcisToQasm.convert_qcis_to_qasm(qcis_circuit)
    return qasm_str


def QASM_2_QCIS(qasm_str):

    qcis_str = QasmToQcis().convert_qasm_to_qcis(qasm_str)
    print('转换后的QCIS指令为:')
    print(qcis_str)


def Json2QCIS(json_str):
    # 定义输入数据
    data = {
        "exp_codes": [
            "H Q0\nH Q1\nCZ Q0 Q1\nB Q0 Q1 Q2\nH Q1\nH Q2\nCZ Q1 Q2\nH Q2\nB Q0 Q1 Q2\nM Q0 Q1 Q2",
            "H Q0\nH Q1\nCZ Q0 Q1\nB Q0 Q1 Q2\nH Q1\nH Q2\nCZ Q1 Q2\nH Q2\nB Q0 Q1 Q2\nX Q0\nX Q1\nX Q2\nM Q0 Q1 Q2",
        ]
    }

    # 提取 exp_codes 并打印每个指令
    for i, code in enumerate(data['exp_codes']):
        print(f"Processed exp_code {i + 1}:")
        for line in code.split('\n'):
            print(line)
        print()  # 打印空行以分隔不同的 exp_code

if __name__ == '__main__':
    # circuit = read_all('data/circuits/xeb/xeb3/XEB_3_qubits_8_cycles_circuit.txt')
    # qasm_str = QCIS_2_QASM(circuit)
    # circuit = QuantumCircuit.from_qasm_str(qasm_str=qasm_str)
    # print(qasm_str)
    # result = simulator.run(circuit).result()
    # print(result.get_counts(0))

    Json2QCIS('')