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


if __name__ == '__main__':
    circuit = read_all('data/test_set/xeb/xeb3/XEB_3_qubits_8_cycles_circuit.txt')
    qasm_str = QCIS_2_QASM(circuit)
    circuit = QuantumCircuit.from_qasm_str(qasm_str=qasm_str)
    print(qasm_str)
    result = simulator.run(circuit).result()
    print(result.get_counts(0))