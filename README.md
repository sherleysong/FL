# Multi Factor Federated Learning

1. you must change the path like this:
```
data=pd.read_excel('/Users/admin/project1/order.xlsx')
poi_data=pd.read_excel('/Users/admin/project1/poi_effect.xlsx')
weather_data=pd.read_excel('/Users/admin/project1/weather_0.xlsx')
```

2. the main hardware and sofeware env:

```
Memory  32 GB
Pycharm 2022.3.2 (Community Edition)
Anaconda  2022.10
Python  3.9.13
TensorFlow 2.11.0
```

3. `main.py` provides several different ways to split data, you can change the way here:
```
# Generate index lists. change different methods to split orders
# 重要，根据不同方式划分数据集。
client_data_list = split_client_datasetnoniid2(num_clients, len_dataset)
print("split_client_datasetnoniid2")
```

In the paper, we make `split_client_datasetFixedTotal` for baseline , `split_client_datasetFixed` for IID,  `split_client_datasetnoniid2`  for Non-IID. 

4. the proposed ControlFedAvg when make and control client A : 

 We make client A as the controlled client by IID data with 5 Days, then 
`ControlFedAvg.py` shows FedAvg result;
`ControlFedAvg2.py` shows one ControlFedAvg , which gives client A double weight.
