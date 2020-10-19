import logging
import os
from collections import OrderedDict
import torch
import cv2
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.comm import get_world_size, is_main_process
import time 
import datetime 
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_setup, launch
from detectron2.evaluation import print_csv_format, inference_context
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from contact_hands_two_stream.engine import custom_argument_parser, CustomPredictor
from contact_hands_two_stream.config import add_contacthands_config
from contact_hands_two_stream.modeling import (
    first_stream_rcnn,
    second_stream_rcnn,
    first_stream_roi_head,
    second_stream_roi_head,
)
from contact_hands_two_stream.data.build import custom_train_loader, custom_test_loader
from detectron2.utils.logger import log_every_n_seconds
from contact_hands_two_stream.evaluation import PascalVOCContactHandsEvaluator
from datasets import *

logger = logging.getLogger("detectron2")

def get_evaluator(cfg, dataset_name, output_folder=None):

    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = [PascalVOCContactHandsEvaluator(dataset_name)]
    
    if len(evaluator_list) == 0:
        raise NotImplementedError("no Evaluator for the dataset {}".format(dataset_name))
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    
    return DatasetEvaluators(evaluator_list)    

def inference_second_stream(model, inputs, height, width):
    im = cv2.imread(inputs[0]['file_name'])
    im = cv2.resize(im, (width, height))
    outputs = model(im)   
    return outputs 

def inference_first_stream(model1, model2, data_loader, evaluator):
    
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model1), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            
            height = inputs[0]['image'].shape[1]
            width = inputs[0]['image'].shape[2]
            second_stream_outputs = inference_second_stream(model2, inputs, height, width)
            
            outputs = model1(inputs, second_stream_outputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            evaluator.process(inputs, outputs)

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def do_train(cfg1, model1, model2, resume=False):

    model1.train()
    optimizer = build_optimizer(cfg1, model1)
    scheduler = build_lr_scheduler(cfg1, optimizer)

    checkpointer = DetectionCheckpointer(
        model1, cfg1.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg1.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg1.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg1.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg1.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg1.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = custom_train_loader(cfg1)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()
            
            height = data[0]['image'].shape[1]
            width = data[0]['image'].shape[2]
            second_stream_outputs = inference_second_stream(model2, data, height, width)

            loss_dict = model1(data, second_stream_outputs)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg1.TEST.EVAL_PERIOD > 0
                and iteration % cfg1.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg1, model1, model2)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)
    
def do_test(cfg1, model1, model2):

    results = OrderedDict()
    for dataset_name in cfg1.DATASETS.TEST:
        data_loader = custom_test_loader(cfg1, dataset_name)
        evaluator = get_evaluator(
            cfg1, dataset_name, os.path.join(cfg1.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_first_stream(model1, model2, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

        
def setup_stream1(args):

    cfg = get_cfg()
    add_contacthands_config(cfg)
    cfg.merge_from_file(args.first_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    return cfg 

def main(args):
    cfg1 = setup_stream1(args)
    model1 = build_model(cfg1)
    logger.info("Model:\n{}".format(model1))

    cfg2 = get_cfg()
    cfg2.merge_from_file("./configs/second_stream.yaml")
    cfg2.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
    model2 = CustomPredictor(cfg2)

    if args.eval_only:
        DetectionCheckpointer(model1, save_dir=cfg1.OUTPUT_DIR).resume_or_load(
            cfg1.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg1, model1, model2)

    distributed = comm.get_world_size() > 1
    if distributed:
        model1 = DistributedDataParallel(
            model1, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg1, model1, model2, resume=args.resume)
    return do_test(cfg1, model1, model2)

if __name__ == "__main__":
    args = custom_argument_parser().parse_args()
    print("Command line args:", args)
    launch(
        main, 
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
