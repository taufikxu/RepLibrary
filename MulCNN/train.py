import torch
import numpy as np

from Tools.logger import save_context, Logger, CheckpointIO
from Tools import FLAGS, load_config, utils_torch

# from library import loss_gan
from library import inputs, data_iters
from library import trainer
from library import evaluator, models

KEY_ARGUMENTS = load_config(FLAGS.config_file)
text_logger, MODELS_FOLDER, SUMMARIES_FOLDER = save_context(__file__, KEY_ARGUMENTS)

torch.manual_seed(1234)
torch.cuda.manual_seed(1235)
np.random.seed(1236)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FLAGS.device = device

model, optim = inputs.get_model()
model = torch.nn.DataParallel(model).to(device)
scheduler = inputs.get_scheduler(optim)

checkpoint_io = CheckpointIO(checkpoint_dir=MODELS_FOLDER)
checkpoint_io.register_modules(model=model, optim=optim)
logger = Logger(log_dir=SUMMARIES_FOLDER)

print_interv, eval_interv = 20, 400
iters, iters_per_epoch = data_iters.get_dataloader()
iters_total = iters_per_epoch * FLAGS.training.nepoch

trainer = trainer.Trainer(model, optim, iters)
evaluater = evaluator.Evaluator(model)
print(model)

x, y = iters.__next__()
logger.add_imgs(x[:100, :3], "OriginImage")
text_logger.info("Start Training")
for iter_num in range(iters_total):
    x, y = iters.__next__()
    return_dict = trainer.step()
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

checkpoint_io.save("Model_final.pth")
logger.save()
