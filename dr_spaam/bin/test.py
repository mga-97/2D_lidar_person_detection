import argparse
import yaml
import torch

from dr_spaam.dataset import get_dataloader
from dr_spaam.pipeline.pipeline import Pipeline
from dr_spaam.model.get_model import get_model

def run_evaluation(model, pipeline, cfg):
    """ val_loader = get_dataloader(
        split="val",
        batch_size=1,
        num_workers=1,
        shuffle=False,
        dataset_cfg=cfg["dataset"],
    )
    pipeline.evaluate(model, val_loader, tb_prefix="VAL") """

    test_loader = get_dataloader(
        split="test",
        batch_size=1,
        num_workers=1,
        shuffle=False,
        dataset_cfg=cfg["dataset"],
    )
    pipeline.evaluate(model, test_loader, tb_prefix="TEST")


if __name__ == "__main__":
    # Run benchmark to select fastest implementation of ops.
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg", type=str, required=True, help="configuration of the experiment"
    )
    parser.add_argument("--ckpt", type=str, required=False, default=None)
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
        cfg["pipeline"]["Logger"]["backup_list"].append(args.cfg)
        
    model = get_model(cfg["model"])
    model.cuda()

    pipeline = Pipeline(model, cfg["pipeline"])

    pipeline.load_ckpt(model, args.ckpt)

    # dirty fix to avoid repeatative entries in cfg file
    cfg["dataset"]["mixup_alpha"] = cfg["model"]["mixup_alpha"]

    run_evaluation(model, pipeline, cfg)

    pipeline.close()
