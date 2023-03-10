from bootstrap_auc import get_outputs, bootstrap

output1 = r"C:\Users\dongm\Desktop\影像组学的demo数据/outputs.pickle"
output2 = r"C:\Users\dongm\Desktop\影像组学的demo数据/stat.pickle"


test_targets, test_probability = get_outputs(output1, output2)

data = np.vstack([test_targets, test_probability]).T
result = bootstrap(data, 1000, 0.95, get_auc)
print(f"95CI={result[0]}-{result[1]}")