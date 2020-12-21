import torch
import numpy as np

from Tools.logger import save_context, Logger, CheckpointIO
from Tools import FLAGS, load_config, utils_torch

# from library import loss_gan
from library import inputs, data_iters
from library.trainer import trainer_byol
from library import evaluator

KEY_ARGUMENTS = load_config(FLAGS.config_file)
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)

seed_bias = int(FLAGS.seed_bias)
torch.manual_seed(1234 + seed_bias)
torch.cuda.manual_seed(1235 + seed_bias)
np.random.seed(1236 + seed_bias)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS.device = device

onlineModel, optim_o = inputs.get_model()
targetModel, optim_t = inputs.get_model()
onlineModel = torch.nn.DataParallel(onlineModel).to(device)
targetModel = torch.nn.DataParallel(targetModel).to(device)
utils_torch.update_average(targetModel, onlineModel, 0.0)
scheduler_o = inputs.get_scheduler(optim_o)
scheduler_t = inputs.get_scheduler(optim_t)
ema_fun = inputs.get_ema_coefficiency()
optim = inputs.OptimizerWrapper([optim_o, optim_t])
scheduler = inputs.SchedulerWrapper([scheduler_o, scheduler_t])
augmentWrapper = data_iters.get_data_augmentation(aug=True, toTensor=False)
augmentWrapper = torch.nn.DataParallel(augmentWrapper).to(device)

checkpoint_io = CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(onlineModel=onlineModel, targetModel=targetModel, optim_o=optim_o, optim_t=optim_t)
logger = Logger(log_dir=SUMMARIES_FOLDER)

if FLAGS.old_model != "NotApply":
    print("Loading old model")
    checkpoint_io.load_file(FLAGS.old_model)
else:
    print("No old model, train from scratch")

trainer_dict = {"byol": trainer_byol}
trainer = trainer_dict[FLAGS.method].Trainer(onlineModel, targetModel, optim, ema_fun, augmentWrapper)
evaluater = evaluator.Evaluator(onlineModel, targetModel)

print_interv, eval_interv, sample_interv = 20, 400, 1000
iters, iters_per_epoch = data_iters.get_dataloader()
iters_total = iters_per_epoch * FLAGS.training.nepoch

x, y = iters.__next__()
logger.add_imgs(x[:100, :3], "OriginImage")
text_logger.info("Start Training")
for iter_num in range(iters_total):
    x, y = iters.__next__()
    return_dict = trainer.step(x.to(device), y.to(device), iter_num)
    logger.addvs("Training", return_dict, iter_num)
    scheduler.step()

    if iter_num % print_interv == 0:
        prefix = "Iter {}/{}".format(iter_num, iters_total)
        logger.log_info(prefix, text_logger.info, ["Training"])

    if iter_num % eval_interv == 0:
        iters_test = data_iters.get_dataloader(train=False, infinity=False, train_aug=False)
        return_dict = evaluater(iters_test)
        logger.addvs("Testing", return_dict, iter_num)
        prefix = "Iter {}".format(iter_num)
        logger.log_info(prefix, text_logger.info, ["Testing"])

    if iter_num > 0 and (iter_num % 10000 == 0 or iter_num == (iters_total - 1)):
        checkpoint_io.save("Model{:08d}.pth".format(iter_num))
        logger.save()
