import os
from typing import Optional,Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.metric import AverageMeter, BoundaryScoreMeter, ScoreMeter
def train(
        train_loader: DataLoader,
        model,
        criterion_cls: nn.Module,
        criterion_bound: nn.Module,
        lambda_bound_loss: float,
        optimizer,
        epoch: int,
        device: str,):
    losses = AverageMeter("Loss",":.4e")
    model = model.to(device)
    model.train()
    for i, sample in enumerate(train_loader):
        x = sample["feature"]
        t = sample["label"]
        b = sample["boundary"]
        mask = sample["mask"]
        x = x.to(device)
        t = t.to(device)
        b = b.to(device)

        batch_size= x.shape[0]
        print(f" x shpe {x.shape} target shape {t.shape} boundary shape {b.shape} mask shape {mask.shape}")
        output_cls,output_bound= model(x, mask)

        loss = 0.0

        if isinstance(output_cls, list):
            n =len(output_cls)
            for out in output_cls:
                print('output ', out.shape)
                loss+=criterion_cls(out,t,x)/n

        else:
            loss+=criterion_cls(output_cls,t,x)

        if isinstance(output_bound,list):
            n= len(output_bound)
            for out in output_bound:
                print('output ',out.shape)
                loss+=lambda_bound_loss * criterion_bound(out,b,mask)/n
        else:
            loss+=lambda_bound_loss*criterion_bound(output_bound,b,mask)

        losses.update(loss.item(),batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg

def validate(
        val_loader:DataLoader,
        model:nn.Module,
        criterion_cls:nn.Module,
        criterion_bound:nn.Module,
        lambda_bound_loss:float,
        device:str,
        dataset: str,
        dataset_dir: str,
        action_dict,
        iou_thresholds: Tuple[float],
        boundary_th: float,
        tolerance: int,
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        losses = AverageMeter("Loss", ":.4e")
        scores_cls = ScoreMeter(
            # id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
            id2class_map= action_dict,
            iou_thresholds=iou_thresholds,
        )
        scores_bound = BoundaryScoreMeter(
            tolerance=tolerance, boundary_threshold=boundary_th
        )

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            for sample in val_loader:
                x = sample["feature"]
                t = sample["label"]
                b = sample["boundary"]
                mask = sample["mask"]

                x = x.to(device)
                t = t.to(device)
                b = b.to(device)
                mask = mask.to(device)

                batch_size = x.shape[0]

                # compute output and loss
                output_cls, output_bound = model(x)

                loss = 0.0
                loss += criterion_cls(output_cls, t, x)
                loss += criterion_bound(output_bound, b, mask)

                # measure accuracy and record loss
                losses.update(loss.item(), batch_size)

                # calcualte accuracy and f1 score
                output_cls = output_cls.to("cpu").data.numpy()
                output_bound = output_bound.to("cpu").data.numpy()

                t = t.to("cpu").data.numpy()
                b = b.to("cpu").data.numpy()
                mask = mask.to("cpu").data.numpy()

                # update score
                scores_cls.update(output_cls, t, output_bound, mask)
                scores_bound.update(output_bound, b, mask)

        cls_acc, edit_score, segment_f1s = scores_cls.get_scores()
        bound_acc, precision, recall, bound_f1s = scores_bound.get_scores()

        return (
            losses.avg,
            cls_acc,
            edit_score,
            segment_f1s,
            bound_acc,
            precision,
            recall,
            bound_f1s,
        )


def evaluate(
    val_loader: DataLoader,
    model: nn.Module,
    device: str,
    boundary_th: float,
    dataset: str,
    dataset_dir: str,
    action_dict,
    iou_thresholds: Tuple[float],
    tolerance: float,
    result_path: str,
    refinement_method: Optional[str] = None,
) -> None:
    postprocessor = PostProcessor(refinement_method, boundary_th)

    scores_before_refinement = ScoreMeter(
        # id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        id2class_map= action_dict,
        iou_thresholds=iou_thresholds,
    )

    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        # id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        id2class_map=action_dict,
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for sample in val_loader:
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            # compute output and loss
            output_cls, output_bound = model(x)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            x = x.to("cpu").data.numpy()
            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )

            # update score
            scores_before_refinement.update(output_cls, t)
            scores_bound.update(output_bound, b, mask)
            scores_after_refinement.update(refined_output_cls, t)

    print("Before refinement:", scores_before_refinement.get_scores())
    print("Boundary scores:", scores_bound.get_scores())
    print("After refinement:", scores_after_refinement.get_scores())

    # save logs
    scores_before_refinement.save_scores(
        os.path.join(result_path, "test_as_before_refine.csv")
    )
    scores_before_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_before_refinement.csv")
    )

    scores_bound.save_scores(os.path.join(result_path, "test_br.csv"))

    scores_after_refinement.save_scores(
        os.path.join(result_path, "test_as_after_majority_vote.csv")
    )
    scores_after_refinement.save_confusion_matrix(
        os.path.join(result_path, "test_c_matrix_after_majority_vote.csv")
    )

import numpy as np
from utils.metric import PostProcessor
import matplotlib.pyplot as plt

def predict(
    loader: DataLoader,
    model: nn.Module,
    device: str,
    result_path: str,
    boundary_th: float,
) -> None:
    save_dir = os.path.join(result_path, "predictions")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    postprocessor = PostProcessor("refinement_with_boundary", boundary_th)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for sample in loader:
            x = sample["feature"]
            t = sample["label"]
            path = sample["feature_path"][0]
            name = os.path.basename(path)
            mask = sample["mask"].numpy()

            x = x.to(device)

            # compute output and loss
            output_cls, output_bound = model(x)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()

            refined_pred = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )

            pred = output_cls.argmax(axis=1)

            np.save(os.path.join(save_dir, name[:-4] + "_pred.npy"), pred[0])
            np.save(
                os.path.join(save_dir, name[:-4] + "_refined_pred.npy"), refined_pred[0]
            )
            np.save(os.path.join(save_dir, name[:-4] + "_gt.npy"), t[0])

            # make graph for boundary regression
            output_bound = output_bound[0, 0]
            h_axis = np.arange(len(output_bound))
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plt.tick_params(labelbottom=False, labelright=False, labeltop=False)
            plt.ylim(0.0, 1.0)
            ax.set_yticks([0, boundary_th, 1])
            ax.spines["right"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.plot(h_axis, output_bound, color="#e46409")
            plt.savefig(os.path.join(save_dir, name[:-4] + "_boundary.png"))
            plt.close(fig)

def save_checkpoint(
        result_path:str,
        epoch:int,
        model:nn.Module,
        optimizer:optim.Optimizer,
        best_loss:float,
):
    save_states={
        "epoch":epoch,
        "state_dict":model.state_dict(),
        "optimizer":optimizer.state_dict(),
        best_loss:best_loss,
    }
    torch.save(save_states,os.path.join(result_path,"checkpoint.pth"))

def resume(
        result_path:str,
        model:nn.Module,
        optimizer:optim.Optimizer,
):
    resume_path=os.path.join(result_path,"checkpoint.pth")
    print("loading checkpoint {}".format(result_path))

    checkpoint= torch.load(resume_path, map_location=lambda storage,loc:storage)
    begin_epoch= checkpoint["epoch"]
    best_loss= checkpoint["best_loss"]
    model.load_state_dict(checkpoint["state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer"])

    return begin_epoch, model,optimizer,best_loss

def get_optimizer(
    optimizer_name: str,
    model: nn.Module,
    learning_rate: float,
    momentum: float = 0.9,
    dampening: float = 0.0,
    weight_decay: float = 0.0001,
    nesterov: bool = True,
) -> optim.Optimizer:

    assert optimizer_name in ["SGD", "Adam"]
    print(f"{optimizer_name} will be used as an optimizer.")

    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )

    return optimizer

# ===== class weight =======

def get_class_weight(num_class,dataframe):
    nums = [0 for i in range(num_class)]
    bounds = [0 for i in range(3)]
    for idx, row in dataframe.iterrows():
        labels = os.path.join(os.path.dirname(row['feature_path']), 'labels.npy')
        boundaries = os.path.join(os.path.dirname(row['feature_path']), 'boundaries.npy')
        label = np.load(labels)
        boundary = np.load(boundaries)
        num, cnt = np.unique(label, return_counts=True)
        b, ct = np.unique(boundary, return_counts=True)
        for n, c in zip(num, cnt):
            nums[n] += c
        for i, j in zip(b, ct):
            # print(i,j)
            bounds[i] += j

    print(nums, bounds)

    class_num = torch.tensor(nums)
    total = class_num.sum().item()
    frequency = class_num.float() / total
    median = torch.median(frequency)
    class_weight = median / frequency
    pos_num = torch.tensor(bounds)
    totalb = pos_num.sum().item()
    frequencyb = pos_num.float() / totalb
    medianb = torch.median(frequencyb)
    pos_weight = medianb / frequencyb
    print(class_weight,pos_weight)
    return class_weight,pos_weight



#
# nums = [0 for i in range(8)]
# bounds = [0 for i in range(3)]
# print(nums, bounds)
# for idx, row in dataframe.iterrows():
#     labels = os.path.join(os.path.dirname(row['feature_path']), 'labels.npy')
#     boundaries = os.path.join(os.path.dirname(row['feature_path']), 'boundaries.npy')
#     label = np.load(labels)
#     boundary = np.load(boundaries)
#     num, cnt = np.unique(label, return_counts=True)
#     b, ct = np.unique(boundary, return_counts=True)
#     for n, c in zip(num, cnt):
#         nums[n] += c
#     for i, j in zip(b, ct):
#         # print(i,j)
#         bounds[i] += j
#
# print(nums, bounds)
#
# class_num = torch.tensor(nums)
# total = class_num.sum().item()
# frequency = class_num.float() / total
# median = torch.median(frequency)
# class_weight = median / frequency
# print(class_weight)
#
# pos_ratio = bounds[1] / sum(nums)
# pos_weight = 1 / pos_ratio
#
# pos_num = torch.tensor(bounds)
# totalb = pos_num.sum().item()
# frequencyb = pos_num.float() / totalb
# medianb = torch.median(frequencyb)
# pos_weight = medianb / frequencyb
# print(pos_weight)