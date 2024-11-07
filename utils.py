import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

def get_bboxes(
    loader,  # 数据加载器，包含数据集的批次数据
    model,   # 使用的模型，进行预测
    iou_threshold,  # 用于NMS的交并比阈值
    threshold,  # 用于决定是否保留预测的阈值
    pred_format="cells",  # 预测数据的格式
    box_format="midpoint",  # 边界框的格式
    device="cuda",  # 运算设备，如 'cuda' 或 'cpu'
):
    all_pred_boxes = []  # 存储所有预测框
    all_true_boxes = []  # 存储所有真实框

    # 确保模型在评估模式，这通常会禁用Dropout等
    model.eval()
    train_idx = 0  # 训练索引，用于跟踪不同批次的数据

    # 遍历加载器中的所有批次
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)  # 将输入数据移至计算设备
        labels = labels.to(device)  # 将标签数据移至计算设备

        with torch.no_grad():  # 关闭梯度计算，因为这里只进行推理
            predictions = model(x)  # 对当前批次的数据进行预测

        batch_size = x.shape[0]  # 获取批次大小
        true_bboxes = cellboxes_to_boxes(labels)  # 将真实标签转换为边界框格式
        bboxes = cellboxes_to_boxes(predictions)  # 将预测结果转换为边界框格式

        # 处理每个样本的预测和真实标签
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )  # 应用非最大抑制，过滤掉一些重叠的边界框

            # 添加过滤后的预测框到结果列表
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            # 添加真实边界框到结果列表，只有当置信度大于阈值时才添加
            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1  # 更新训练索引

    model.train()  # 恢复模型到训练模式
    return all_pred_boxes, all_true_boxes  # 返回所有预测和真实边界框

def convert_cellboxes(predictions, S=7):
    """
    将 YOLO 输出的边界框从基于每个网格单元的比例转换为基于整个图像的比例。
    此函数假定模型的单次前向传递输出包括置信度分数和标准化到各个网格单元的边界框坐标。

    参数:
        predictions (tensor): YOLO 模型输出的张量，形状为 [batch_size, S*S*(B*5+C)]
        S (int): 网格的大小 (SxS)

    返回:
        tensor: 转换后的预测，其中边界框以整个图像的比例表示。
    """

    # 确保预测数据在 CPU 上，以便进行运算
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]

    # 将预测数据重新塑形为更易管理的格式
    predictions = predictions.reshape(batch_size, 7, 7, 30)

    # 提取每个单元格的两个预测边界框
    bboxes1 = predictions[..., 21:25]  # 第一个边界框
    bboxes2 = predictions[..., 26:30]  # 第二个边界框

    # 提取边界框的置信度分数
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    # unsqueeze(0)这个操作在张量的最前面添加一个新的维度，将原来的 [batch_size, 7, 7] 形状变为 [1, batch_size, 7, 7]
    #torch.cat(..., dim=0) 函数将两个张量沿着第一个维度（dim=0，新添加的维度）合并。因为每个张量在 unsqueeze 之后的形状为 [1, batch_size, 7, 7]，合并后的张量形状变为 [2, batch_size, 7, 7]。
    # 这表示对于每个网格单元，现在有两个置信度分数，分别对应两个不同的预测边界框。
    
    # 确定每个单元格中得分最高的边界框
    best_box = scores.argmax(0).unsqueeze(-1)  # 形状为 [batch_size, 7, 7, 1]
    #scores.argmax(0)：这个操作沿着维度0（边界框置信度得分维度）查找最大值的索引[batch_size, 7, 7]
    
    
    # 选择每个单元格置信度得分最高的边界框
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2

    # 计算网格单元格索引，用于修正边界框预测
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    #.unsqueeze(-1)是在张量的最后一个维度上增加一个维度，从而将形状从 [batch_size, 7, 7] 转换为 [batch_size, 7, 7, 1]。
    #这一步是必须的，因为在后续的计算中，经常需要这个额外的维度来进行广播操作（broadcasting），使得单个维度上的值能够自动扩展以匹配其他操作数的相应维度。

    # 通过调整单元格索引计算网格上 x 和 y 的绝对位置
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))

    # 将宽度和高度标准化为整个图像的比例
    w_y = 1 / S * best_boxes[..., 2:4]

    # 将 x, y, 宽度和高度组合成一个张量
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)

    # 使用 argmax 确定得分最高的类别
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)

    # 提取选定边界框的置信度分数
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)

    # 将类别、置信度和边界框组合成最终的预测张量
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds



def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes
