import json
from collections import defaultdict


def aggregate_results(file_path):
    # extract results
    data = defaultdict(list)
    with open(file_path, "r") as f:
        for l in f:
            if not (l.startswith("ST_MODEL_ACC") or l.startswith("STD_MODEL_ACC")):
                continue
            result, target_task = l.strip().split("--------", 1)
            target_task = json.loads(target_task)
            composed_model_size = float(result.rsplit("~", 1)[1].strip().split(")", 1)[0])

            composed_model_acc = float(result.split("- COM_MODEL_ACC", 1)[1].split(" ")[1])
            # composed_model_acc = float(result.split("STD_MODEL_ACC", 1)[1].split(" ")[1])
            # composed_model_acc = float(result.split("- NotFineTuned__MODULE_ACC", 1)[1].split(" ")[1].split(")", 1)[0])
            # composed_model_acc = float(result.split("- FineTuned__MODULE_ACC", 1)[1].split(" ")[1])
            # st_model_acc = float(result.split("ST_MODEL_ACC", 1)[1].split(" ")[1])
            # composed_model_acc = float(result.split("- COM_MODEL_ACC", 1)[1].split(" ")[1])
            # st_model_acc = float(result.split("STD_MODEL_ACC", 1)[1].split(" ")[1])
            # if st_model_acc - composed_model_acc > 3.0:
            #     print(composed_model_acc, st_model_acc)

            data[len(target_task)].append((target_task, composed_model_acc, composed_model_size))
    sorted_keys = sorted(data.keys())

    # aggregate results
    avg_acc_list = []
    for k in sorted_keys:
        v = data[k]
        avg_acc = sum(map(lambda x: x[1], v)) / len(v)
        avg_acc_list.append(avg_acc)
        avg_size = sum(map(lambda x: x[2], v)) / len(v)
        print(f"{len(v)} [{k}-class] ACC: {round(avg_acc, 2)} - SIZE: {round(avg_size, 2)}")

    print("\n----\nAVG_ACC", round(sum(avg_acc_list)/len(avg_acc_list), 2))


if __name__ == '__main__':
    model_composition_result_file_path = "[MODA]/module_evaluation/modularization_eval.vgg16_cifar10.thres0.1.out"
    # model_composition_result_file_path = "[MwT]/src/module.mobilenet_cifar10.last.out"
    print(model_composition_result_file_path, "\n")
    aggregate_results(model_composition_result_file_path)
