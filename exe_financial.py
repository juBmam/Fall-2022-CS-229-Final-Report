import argparse
import torch
import datetime
import json
import yaml
import os
import time

from main_model import CSDI_financial
from dataset_financial import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="/content/drive/My Drive/CSDI-main/config/base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument("--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])")
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)
path = "/content/drive/My Drive/CSDI-main/config/base.yaml"
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

foldername = "/content/drive/My Drive/CSDI-main/save/financial_fold" + str(args.nfold) + "_" + current_time + "/"

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


train_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = CSDI_financial(config, args.device).to(args.device)

if args.modelfolder == "":
    print("creating new model")
    start_time = time.time()
    train(
        model,
        config["train"],
        train_loader,
        #valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    print("reusing old model")
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
print("--- %s seconds ---" % (time.time() - start_time))