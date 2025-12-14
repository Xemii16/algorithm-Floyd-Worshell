import json

with open("experiment_70.99.json", "r") as file:
    results = json.load(file)
    print("n: " + ",".join(list(map(lambda result: str(result["n"]), results))))
    print("density: " + ",".join(list(map(lambda result: str(result["density"]), results))))
    print("algorithm_work_time: " + ",".join(list(map(lambda result: str(result["algorithm_work_time"]), results))))
