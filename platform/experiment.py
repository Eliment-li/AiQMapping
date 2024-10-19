from datetime import datetime
# create_lab 方法创建了一个实验集，并返回了该实验集的唯一标识 lab_id
lab_id = platform.create_lab(name=f'lab.{datetime.now().strftime("%Y%m%d%H%M%S")}', remark='test_collection')

exp_id = platform.save_experiment(lab_id=lab_id, circuit=circuit.qcis, name=f'exp.{datetime.now().strftime("%Y%m%d%H%M%S")}')
query_id_single = platform.run_experiment(exp_id=exp_id, num_shots=5000)
print(f'query_id: {query_id_single}')

print(lab_id)